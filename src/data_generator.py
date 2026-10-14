"""Reproducible synthetic server-monitoring data generator.

Reference distributions (all independent draws, then clipped):
  - cpu_usage:          Normal(mean=50, sd=15), clipped to [0, 100] (%)
  - memory_usage:       Normal(mean=60, sd=12), clipped to [0, 100] (%)
  - disk_usage:         Normal(mean=55, sd=15), clipped to [0, 100] (%)
  - request_latency_ms: Normal(mean=120, sd=40), clipped to [10, 2000] (ms)
  - error_rate:         Normal(mean=0.03, sd=0.02), clipped to [0, 1] (fraction)

Label rule (documented synthetic rule, NOT real incidents):
  z_cpu = (cpu_usage - 50) / 15
  z_mem = (memory_usage - 60) / 12
  z_lat = (request_latency_ms - 120) / 40
  z_err = (error_rate - 0.03) / 0.02
  risk_score = 0.5 * z_cpu + 0.5 * z_mem + 0.7 * z_lat + 0.6 * z_err
  risk_label = 1 if risk_score > 1.0 else 0

This gives roughly 15-25% positive (high-risk) rows on reference data.
The same rule is applied to every dataset (reference and all incoming
scenarios) so labelled evaluation comparisons are consistent.

Incoming scenarios:
  - "normal":   same distributions as reference (different seed only).
  - "moderate": cpu mean 50->60, latency mean 120->150, error mean 0.03->0.045.
  - "severe":   cpu mean 50->72, memory mean 60->78, disk mean 55->70,
                latency mean 120->220 (sd 40->60), error mean 0.03->0.09.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "request_latency_ms",
    "error_rate",
]
LABEL_COLUMN = "risk_label"
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

VALID_SCENARIOS = ("normal", "moderate", "severe")


def _assign_labels(df: pd.DataFrame) -> pd.Series:
    """Apply the documented synthetic label rule."""
    z_cpu = (df["cpu_usage"] - 50.0) / 15.0
    z_mem = (df["memory_usage"] - 60.0) / 12.0
    z_lat = (df["request_latency_ms"] - 120.0) / 40.0
    z_err = (df["error_rate"] - 0.03) / 0.02
    score = 0.5 * z_cpu + 0.5 * z_mem + 0.7 * z_lat + 0.6 * z_err
    return (score > 1.0).astype(int)


def _sample_reference(rng: np.random.Generator, n: int) -> pd.DataFrame:
    cpu = np.clip(rng.normal(50, 15, n), 0, 100)
    mem = np.clip(rng.normal(60, 12, n), 0, 100)
    disk = np.clip(rng.normal(55, 15, n), 0, 100)
    lat = np.clip(rng.normal(120, 40, n), 10, 2000)
    err = np.clip(rng.normal(0.03, 0.02, n), 0, 1)
    df = pd.DataFrame(
        {
            "cpu_usage": cpu,
            "memory_usage": mem,
            "disk_usage": disk,
            "request_latency_ms": lat,
            "error_rate": err,
        }
    )
    df[LABEL_COLUMN] = _assign_labels(df)
    return df


def generate_reference_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate the labelled reference dataset."""
    if not isinstance(n_rows, int) or n_rows <= 0:
        raise ValueError("n_rows must be a positive integer")
    rng = np.random.default_rng(seed)
    return _sample_reference(rng, n_rows)


def generate_incoming_data(
    scenario: str = "normal", n_rows: int = 1000, seed: int = 100
) -> pd.DataFrame:
    """Generate labelled incoming data for one drift scenario.

    Labels use the same rule as the reference data so that evaluation
    comparisons are like-for-like. In a real deployment incoming labels
    would usually be delayed or unavailable; here they exist only to let
    us measure what happened to accuracy under synthetic drift.
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {VALID_SCENARIOS}, got {scenario!r}")
    if not isinstance(n_rows, int) or n_rows <= 0:
        raise ValueError("n_rows must be a positive integer")
    rng = np.random.default_rng(seed)

    if scenario == "normal":
        return _sample_reference(rng, n_rows)

    if scenario == "moderate":
        cpu = np.clip(rng.normal(60, 15, n_rows), 0, 100)
        mem = np.clip(rng.normal(60, 12, n_rows), 0, 100)
        disk = np.clip(rng.normal(55, 15, n_rows), 0, 100)
        lat = np.clip(rng.normal(150, 40, n_rows), 10, 2000)
        err = np.clip(rng.normal(0.045, 0.02, n_rows), 0, 1)
    else:  # severe
        cpu = np.clip(rng.normal(72, 15, n_rows), 0, 100)
        mem = np.clip(rng.normal(78, 12, n_rows), 0, 100)
        disk = np.clip(rng.normal(70, 15, n_rows), 0, 100)
        lat = np.clip(rng.normal(220, 60, n_rows), 10, 2000)
        err = np.clip(rng.normal(0.09, 0.03, n_rows), 0, 1)

    df = pd.DataFrame(
        {
            "cpu_usage": cpu,
            "memory_usage": mem,
            "disk_usage": disk,
            "request_latency_ms": lat,
            "error_rate": err,
        }
    )
    df[LABEL_COLUMN] = _assign_labels(df)
    return df
