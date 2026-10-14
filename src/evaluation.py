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


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute accuracy/precision/recall/F1 + confusion matrix.

    Uses zero_division=0 so small samples or a missing predicted class
    produce 0.0 instead of raising.
    """
    if len(X) == 0:
        raise ValueError("evaluation data is empty")
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")
    y_pred = model.predict(X)
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": np.asarray(confusion_matrix(y, y_pred)).tolist(),
    }


def compare_metrics(reference: dict, incoming: dict) -> pd.DataFrame:
    """Row-per-metric table of reference vs incoming and absolute delta."""
    rows = []
    for metric in ("accuracy", "precision", "recall", "f1"):
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
