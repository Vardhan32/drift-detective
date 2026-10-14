"""Evaluation helpers: metrics from actual predictions, no hard-coding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_generator import FEATURE_COLUMNS

KNOWN_LABELS = [0, 1]
REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1")


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute accuracy/precision/recall/F1 + confusion matrix.

    Uses zero_division=0 so small samples or a missing predicted class
    produce 0.0 instead of raising. The confusion matrix is always 2x2
    (labels=[0, 1]) even when a class is absent from the evaluation data.
    """
    if len(X) == 0:
        raise ValueError("evaluation data is empty")
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")
    if isinstance(X, pd.DataFrame):
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"evaluation data is missing feature columns: {missing}")
    y_pred = model.predict(X)
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": np.asarray(confusion_matrix(y, y_pred, labels=KNOWN_LABELS)).tolist(),
    }


def compare_metrics(reference: dict, incoming: dict) -> pd.DataFrame:
    """Row-per-metric table of reference vs incoming and absolute delta."""
    for name, metrics in (("reference", reference), ("incoming", incoming)):
        absent = [m for m in REQUIRED_METRICS if m not in metrics]
        if absent:
            raise ValueError(f"{name} metrics are missing keys: {absent}")
    rows = []
    for metric in REQUIRED_METRICS:
        ref_v = float(reference[metric])
        inc_v = float(incoming[metric])
        rows.append(
            {
                "metric": metric,
                "reference": round(ref_v, 4),
                "incoming": round(inc_v, 4),
                "delta": round(inc_v - ref_v, 4),
            }
        )
    return pd.DataFrame(rows)
