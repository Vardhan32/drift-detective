"""Feature-level drift detection with the two-sample KS test.

For each numeric feature we compare reference vs incoming and report:
  counts, means, stds, median change, KS statistic, raw p-value,
  Benjamini-Hochberg adjusted p-value, Cohen's d effect size, drift flag.

Drift flag rule (transparent, not p-value-only):
  drifted = (adjusted_p < alpha AND |cohens_d| >= 0.2) OR (|cohens_d| >= 0.5)

Rationale: with large n even tiny shifts give small p-values, so we also
require a minimum effect size; conversely a huge effect size is flagged
even if the corrected p-value narrowly misses the threshold.

Overall status: "drift detected" if >= 1 feature is flagged.

Statistical notes (also shown in the dashboard + methodology doc):
  - H0 of the KS test: the two samples come from the same distribution.
  - A p-value is P(data this extreme | H0), NOT P(H0 is true).
  - Large n -> small differences become "significant".
  - Significance != operational importance.
  - Drift alone does not prove accuracy dropped; check labelled metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_generator import FEATURE_COLUMNS

MIN_SAMPLES = 10
DEFAULT_ALPHA = 0.05
SMALL_EFFECT = 0.2
LARGE_EFFECT = 0.5


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # cumulative minimum from the largest p-value downwards
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def detect_drift(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Compare distributions per feature. Returns summary + per-feature table."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if reference is None or incoming is None:
        raise ValueError("reference and incoming must be DataFrames")
    if reference.empty or incoming.empty:
        raise ValueError("reference and incoming must be non-empty")

    features = list(features) if features else list(FEATURE_COLUMNS)
    for col in features:
        if col not in reference.columns or col not in incoming.columns:
            raise ValueError(f"missing expected feature column: {col!r}")

    raw_rows: list[dict] = []
    for col in features:
        ref = reference[col]
        inc = incoming[col]
        if not pd.api.types.is_numeric_dtype(ref) or not pd.api.types.is_numeric_dtype(inc):
            raise ValueError(f"feature {col!r} must be numeric")
        ref_vals = pd.to_numeric(ref, errors="coerce").to_numpy(dtype=float)
        inc_vals = pd.to_numeric(inc, errors="coerce").to_numpy(dtype=float)
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        inc_vals = inc_vals[np.isfinite(inc_vals)]
        if len(ref_vals) < MIN_SAMPLES or len(inc_vals) < MIN_SAMPLES:
            raise ValueError(
                f"feature {col!r} needs at least {MIN_SAMPLES} valid samples "
                f"(got ref={len(ref_vals)}, incoming={len(inc_vals)})"
            )
        if np.all(ref_vals == ref_vals[0]) and np.all(inc_vals == inc_vals[0]):
            # both constant: identical -> no drift; different -> definite shift
            # (Cohen's d is undefined on zero variance, so flag directly)
            if ref_vals[0] == inc_vals[0]:
                ks_stat, p_value, constant_shift = 0.0, 1.0, False
            else:
                ks_stat, p_value, constant_shift = 1.0, 0.0, True
        else:
            constant_shift = False
            try:
                res = ks_2samp(ref_vals, inc_vals)
                ks_stat, p_value = float(res.statistic), float(res.pvalue)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"KS test failed for {col!r}: {exc}") from exc
            if not np.isfinite(p_value):
                p_value = 1.0
            if not np.isfinite(ks_stat):
                ks_stat = 0.0

        raw_rows.append(
            {
                "feature": col,
                "n_reference": int(len(ref_vals)),
                "n_incoming": int(len(inc_vals)),
                "mean_reference": float(np.mean(ref_vals)),
                "mean_incoming": float(np.mean(inc_vals)),
                "std_reference": float(np.std(ref_vals, ddof=1)) if len(ref_vals) > 1 else 0.0,
                "std_incoming": float(np.std(inc_vals, ddof=1)) if len(inc_vals) > 1 else 0.0,
                "median_change": float(np.median(inc_vals) - np.median(ref_vals)),
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "cohens_d": _cohens_d(ref_vals, inc_vals),
                "constant_shift": constant_shift,
            }
        )

    adj = benjamini_hochberg(np.array([r["p_value"] for r in raw_rows]))
    for row, ap in zip(raw_rows, adj):
        row["p_value_adj"] = float(ap)
        row["alpha"] = float(alpha)
        d = abs(row["cohens_d"])
        row["drifted"] = bool(
            row.pop("constant_shift")
            or (ap < alpha and d >= SMALL_EFFECT)
            or (d >= LARGE_EFFECT)
        )

    results = pd.DataFrame(raw_rows)
    n_flagged = int(results["drifted"].sum())
    return {
        "results": results,
        "n_features": len(results),
        "n_flagged": n_flagged,
        "overall_drift": n_flagged > 0,
        "status": "drift detected" if n_flagged > 0 else "no significant drift",
        "alpha": float(alpha),
    }
