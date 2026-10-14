"""Tests for experiment-parameter validation (sidebar + URL query params)."""

import pandas as pd

from src.data_generator import FEATURE_COLUMNS, LABEL_COLUMN
from src.model import split_data
from src.params import (
    DEFAULT_ROWS,
    DEFAULT_SEED,
    ROW_MAX,
    ROW_MIN,
    resolve_params,
)


def test_valid_params_pass_through_without_warnings():
    params, warnings = resolve_params(
        {"scenario": "severe", "n_reference": "500", "n_incoming": 800,
         "seed": "7", "alpha": "0.05"}
    )
    assert warnings == []
    assert params == {
        "scenario": "severe",
        "n_reference": 500,
        "n_incoming": 800,
        "seed": 7,
        "alpha": 0.05,
    }


def test_empty_raw_gives_defaults():
    params, warnings = resolve_params(None)
    assert params["scenario"] == "moderate"
    assert params["n_reference"] == DEFAULT_ROWS
    assert params["seed"] == DEFAULT_SEED
    assert warnings == []


def test_invalid_scenario_falls_back_with_warning():
    params, warnings = resolve_params({"scenario": "bogus"})
    assert params["scenario"] == "moderate"
    assert any("bogus" in w for w in warnings)


def test_row_counts_clamped_with_warnings():
    params, warnings = resolve_params({"n_reference": "5", "n_incoming": "99999"})
    assert params["n_reference"] == ROW_MIN
    assert params["n_incoming"] == ROW_MAX
    assert len(warnings) >= 2


def test_non_numeric_rows_use_default_with_warning():
    params, warnings = resolve_params({"n_reference": "abc"})
    assert params["n_reference"] == DEFAULT_ROWS
    assert any("n_reference" in w for w in warnings)


def test_small_samples_produce_noise_warning():
    params, warnings = resolve_params({"n_reference": "50", "n_incoming": "60"})
    assert params["n_reference"] == 50
    assert any("Small sample" in w for w in warnings)


def test_invalid_seed_and_alpha_fall_back():
    params, warnings = resolve_params({"seed": "-3", "alpha": "0.9"})
    assert params["seed"] == DEFAULT_SEED
    assert params["alpha"] == 0.05
    assert len(warnings) == 2


def _ten_rows(positive: int) -> pd.DataFrame:
    data = {
        "cpu_usage": [50.0] * 10,
        "memory_usage": [60.0] * 10,
        "disk_usage": [55.0] * 10,
        "request_latency_ms": [120.0] * 10,
        "error_rate": [0.03] * 10,
        LABEL_COLUMN: [1] * positive + [0] * (10 - positive),
    }
    return pd.DataFrame(data)[FEATURE_COLUMNS + [LABEL_COLUMN]]


def test_ten_balanced_rows_split_at_minimum():
    X_train, X_test, y_train, y_test = split_data(_ten_rows(5))
    assert len(X_train) == 8 and len(X_test) == 2


def test_ten_single_class_rows_cannot_split():
    import pytest

    with pytest.raises(ValueError):
        split_data(_ten_rows(10))
