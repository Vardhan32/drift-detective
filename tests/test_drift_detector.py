"""Tests for the KS drift detector."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_incoming_data, generate_reference_data
from src.drift_detector import benjamini_hochberg, detect_drift


def test_normal_scenario_no_drift():
    ref = generate_reference_data(n_rows=1000, seed=42)
    inc = generate_incoming_data(scenario="normal", n_rows=1000, seed=999)
    out = detect_drift(ref, inc)
    assert out["n_features"] == 5
    assert out["n_flagged"] == 0
    assert out["overall_drift"] is False


def test_severe_scenario_detects_drift():
    ref = generate_reference_data(n_rows=1000, seed=42)
    inc = generate_incoming_data(scenario="severe", n_rows=1000, seed=999)
    out = detect_drift(ref, inc)
    assert out["overall_drift"] is True
    assert out["n_flagged"] >= 2
    assert "ks_statistic" in out["results"].columns
    assert "p_value_adj" in out["results"].columns
    assert "cohens_d" in out["results"].columns


def test_moderate_scenario_partial_drift():
    ref = generate_reference_data(n_rows=1000, seed=42)
    inc = generate_incoming_data(scenario="moderate", n_rows=1000, seed=999)
    out = detect_drift(ref, inc)
    assert out["n_flagged"] >= 1


def test_constant_columns_no_crash():
    ref = pd.DataFrame({c: np.ones(50) * 10.0 for c in
                        ["cpu_usage", "memory_usage", "disk_usage",
                         "request_latency_ms", "error_rate"]})
    same = ref.copy()
    out = detect_drift(ref, same)
    assert out["n_flagged"] == 0
    shifted = ref.copy()
    shifted["cpu_usage"] = 99.0
    out2 = detect_drift(ref, shifted)
    assert out2["results"].set_index("feature").loc["cpu_usage", "drifted"]


def test_missing_values_handled():
    ref = generate_reference_data(n_rows=200, seed=3)
    inc = generate_incoming_data(scenario="normal", n_rows=200, seed=4)
    ref.loc[ref.index[:10], "cpu_usage"] = np.nan
    inc.loc[inc.index[:10], "error_rate"] = np.nan
    out = detect_drift(ref, inc)
    assert out["n_features"] == 5


def test_small_sample_raises():
    ref = generate_reference_data(n_rows=200, seed=3).head(5)
    inc = generate_incoming_data(scenario="normal", n_rows=200, seed=4)
    with pytest.raises(ValueError):
        detect_drift(ref, inc)


def test_invalid_inputs_raise():
    ref = generate_reference_data(n_rows=100, seed=1)
    inc = generate_incoming_data(scenario="normal", n_rows=100, seed=2)
    with pytest.raises(ValueError):
        detect_drift(ref, inc, alpha=0.0)
    with pytest.raises(ValueError):
        detect_drift(pd.DataFrame(), inc)
    with pytest.raises(ValueError):
        detect_drift(ref, inc, features=["nonexistent"])
    bad = ref.copy()
    bad["cpu_usage"] = "not-a-number"
    with pytest.raises(ValueError):
        detect_drift(bad, inc)
    inf_df = inc.copy()
    inf_df.loc[inf_df.index[0], "cpu_usage"] = np.inf
    # inf values are dropped, still enough samples -> no raise
    out = detect_drift(ref, inf_df)
    assert out["n_features"] == 5


def test_bh_correction():
    adj = benjamini_hochberg(np.array([0.01, 0.04, 0.5]))
    assert len(adj) == 3
    assert (adj >= np.array([0.01, 0.04, 0.5])).all()
    assert (adj <= 1.0).all()


def test_bh_exact_values_unsorted_input():
    # Hand-computed: sorted p = [0.01, 0.04, 0.5], m = 3
    # adj = [0.01*3/1, 0.04*3/2, 0.5*3/3] = [0.03, 0.06, 0.5]
    adj = benjamini_hochberg(np.array([0.5, 0.01, 0.04]))
    np.testing.assert_allclose(adj, [0.5, 0.03, 0.06], rtol=1e-9)


def test_all_nan_column_raises():
    ref = generate_reference_data(n_rows=100, seed=1)
    inc = generate_incoming_data(scenario="normal", n_rows=100, seed=2)
    inc["cpu_usage"] = np.nan
    with pytest.raises(ValueError):
        detect_drift(ref, inc)


def test_one_sided_constant_column_no_crash():
    ref = generate_reference_data(n_rows=100, seed=1)
    inc = generate_incoming_data(scenario="normal", n_rows=100, seed=2)
    ref["disk_usage"] = 55.0
    out = detect_drift(ref, inc)
    assert out["n_features"] == 5
    assert "drifted" in out["results"].columns
