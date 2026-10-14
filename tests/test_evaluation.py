"""Tests for evaluation helpers."""

import pytest

from src.data_generator import LABEL_COLUMN, generate_incoming_data, generate_reference_data
from src.evaluation import compare_metrics, evaluate_model
from src.model import split_data, train_model


def _trained():
    df = generate_reference_data(n_rows=500, seed=42)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    return model, X_test, y_test


def test_evaluate_returns_real_metrics():
    model, X_test, y_test = _trained()
    m = evaluate_model(model, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0
    assert m["n_samples"] == len(y_test)
    assert len(m["confusion_matrix"]) == 2
    assert m["accuracy"] > 0.7


def test_evaluate_invalid_inputs():
    model, X_test, y_test = _trained()
    with pytest.raises(ValueError):
        evaluate_model(model, X_test.head(0), y_test.head(0))
    with pytest.raises(ValueError):
        evaluate_model(model, X_test, y_test.head(10))


def test_compare_metrics():
    model, X_test, y_test = _trained()
    ref_m = evaluate_model(model, X_test, y_test)
    inc = generate_incoming_data(scenario="severe", n_rows=300, seed=11)
    from src.data_generator import FEATURE_COLUMNS

    inc_m = evaluate_model(model, inc[FEATURE_COLUMNS], inc[LABEL_COLUMN])
    cmp_df = compare_metrics(ref_m, inc_m)
    assert list(cmp_df.columns) == ["metric", "reference", "incoming", "delta"]
    assert len(cmp_df) == 4
