"""Tests for the synthetic data generator."""

import numpy as np
import pandas as pd

from src.data_generator import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    generate_incoming_data,
    generate_reference_data,
)


def test_reference_columns_and_row_count():
    df = generate_reference_data(n_rows=1000, seed=42)
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 1000


def test_feature_ranges():
    df = generate_reference_data(n_rows=2000, seed=1)
    assert df["cpu_usage"].between(0, 100).all()
    assert df["memory_usage"].between(0, 100).all()
    assert df["disk_usage"].between(0, 100).all()
    assert (df["request_latency_ms"] >= 10).all()
    assert df["error_rate"].between(0, 1).all()


def test_labels_binary_and_both_classes():
    df = generate_reference_data(n_rows=1000, seed=42)
    assert set(df[LABEL_COLUMN].unique()) <= {0, 1}
    assert df[LABEL_COLUMN].nunique() == 2
    # sane balance: positive rate between 5% and 50%
    rate = (df[LABEL_COLUMN] == 1).mean()
    assert 0.05 < rate < 0.50


def test_reproducibility():
    a = generate_reference_data(n_rows=500, seed=123)
    b = generate_reference_data(n_rows=500, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_reference_data(n_rows=500, seed=1)
    b = generate_reference_data(n_rows=500, seed=2)
    assert not a.equals(b)


def test_scenarios_have_expected_columns_and_labels():
    for scenario in ("normal", "moderate", "severe"):
        df = generate_incoming_data(scenario=scenario, n_rows=300, seed=7)
        assert list(df.columns) == ALL_COLUMNS
        assert set(df[LABEL_COLUMN].unique()) <= {0, 1}


def test_normal_close_to_reference_but_severe_shifted():
    ref = generate_reference_data(n_rows=2000, seed=42)
    normal = generate_incoming_data(scenario="normal", n_rows=2000, seed=999)
    severe = generate_incoming_data(scenario="severe", n_rows=2000, seed=999)
    # normal means should be close to reference means
    for col in FEATURE_COLUMNS:
        assert abs(normal[col].mean() - ref[col].mean()) < 5.0 or col == "error_rate"
    # severe scenario shifts cpu and latency substantially
    assert severe["cpu_usage"].mean() - ref["cpu_usage"].mean() > 10
    assert severe["request_latency_ms"].mean() - ref["request_latency_ms"].mean() > 40


def test_moderate_between_normal_and_severe():
    normal = generate_incoming_data(scenario="normal", n_rows=2000, seed=5)
    moderate = generate_incoming_data(scenario="moderate", n_rows=2000, seed=5)
    severe = generate_incoming_data(scenario="severe", n_rows=2000, seed=5)
    assert moderate["cpu_usage"].mean() > normal["cpu_usage"].mean()
    assert severe["cpu_usage"].mean() > moderate["cpu_usage"].mean()


def test_invalid_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        generate_reference_data(n_rows=0)
    with pytest.raises(ValueError):
        generate_incoming_data(scenario="unknown", n_rows=100)
    with pytest.raises(ValueError):
        generate_incoming_data(scenario="normal", n_rows=-5)
