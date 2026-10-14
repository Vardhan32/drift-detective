"""Model training pipeline: split, train, importances.

Uses a RandomForestClassifier baseline:
  - robust on small tabular data, no feature scaling needed
  - natively gives feature_importances_ for the explainability panel
  - deterministic with random_state

Preprocessing: median imputation via a Pipeline fitted on training data
only, so missing values at prediction time are handled without leaking
test statistics into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_generator import FEATURE_COLUMNS, LABEL_COLUMN

DEFAULT_RANDOM_STATE = 42


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the label column."""
    if df.empty:
        raise ValueError("input DataFrame is empty")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"missing label column {LABEL_COLUMN!r}")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if len(df) < 10:
        raise ValueError("need at least 10 rows to split")
    if df[LABEL_COLUMN].nunique() < 2:
        raise ValueError("need at least 2 classes to stratify")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def class_balance(y: pd.Series) -> dict:
    """Return per-class counts and positive rate."""
    counts = y.value_counts().to_dict()
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "n_samples": int(len(y)),
        "positive_rate": float((y == 1).mean()) if len(y) else 0.0,
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = 200,
) -> Pipeline:
    """Fit median-imputation + RandomForest pipeline on training data only."""
    if len(X_train) == 0:
        raise ValueError("X_train is empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different lengths")
    if y_train.nunique() < 2:
        raise ValueError("y_train must contain at least 2 classes")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importance(
    model: Pipeline, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """Return sorted feature importances from the fitted forest."""
    names = feature_names or FEATURE_COLUMNS
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if not hasattr(clf, "feature_importances_"):
        raise ValueError("model does not expose feature_importances_")
    importances = np.asarray(clf.feature_importances_, dtype=float)
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
