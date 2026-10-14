"""Tests for model training pipeline."""

import pandas as pd
import pytest

from src.data_generator import FEATURE_COLUMNS, LABEL_COLUMN, generate_reference_data
from src.model import class_balance, get_feature_importance, split_data, train_model


@pytest.fixture()
def ref_df():
    return generate_reference_data(n_rows=500, seed=42)


def test_split_stratified_sizes(ref_df):
    X_train, X_test, y_train, y_test = split_data(ref_df, test_size=0.2, random_state=42)
    assert len(X_train) == 400
    assert len(X_test) == 100
    # stratification keeps positive rate close
    assert abs((y_train == 1).mean() - (y_test == 1).mean()) < 0.05


def test_class_balance(ref_df):
    bal = class_balance(ref_df[LABEL_COLUMN])
    assert bal["n_samples"] == 500
    assert set(bal["counts"].keys()) == {0, 1}
    assert 0.0 < bal["positive_rate"] < 1.0


def test_train_and_predict(ref_df):
    X_train, X_test, y_train, y_test = split_data(ref_df)
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(pd.Series(preds).unique()) <= {0, 1}
    assert (preds == y_test.values).mean() > 0.7  # synthetic rule is learnable


def test_feature_importance(ref_df):
    X_train, _, y_train, _ = split_data(ref_df)
    model = train_model(X_train, y_train)
    imp = get_feature_importance(model)
    assert list(imp.columns) == ["feature", "importance"]
    assert set(imp["feature"]) == set(FEATURE_COLUMNS)
    assert abs(imp["importance"].sum() - 1.0) < 1e-6


def test_invalid_inputs():
    with pytest.raises(ValueError):
        split_data(pd.DataFrame(), test_size=0.2)
    with pytest.raises(ValueError):
        train_model(pd.DataFrame(), pd.Series(dtype=int))
    df = generate_reference_data(n_rows=20, seed=0)
    one_class = df.copy()
    one_class[LABEL_COLUMN] = 0
    with pytest.raises(ValueError):
        split_data(one_class)
    X = one_class[FEATURE_COLUMNS]
    with pytest.raises(ValueError):
        train_model(X, one_class[LABEL_COLUMN])
