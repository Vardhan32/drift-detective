"""Validation for experiment parameters (sidebar + URL query params).

Minimums mirror the real component requirements, not guesses:
  - rows >= 10: split_data needs >= 10 rows and detect_drift needs
    MIN_SAMPLES (10) valid samples per feature.
  - rows <= 3000: keeps local runs fast; matches the sidebar sliders.
  - rows < 100: allowed, but flagged as noisy (metrics wobble and the
    10-row sample may contain only one class, which cannot be split).
  - seed in [0, 100000]; alpha kept inside the sidebar slider range.

Never fabricates data: invalid values fall back to safe defaults and every
fallback produces a warning string the dashboard displays.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VALID_SCENARIOS = ("normal", "moderate", "severe")
DEFAULT_SCENARIO = "moderate"
ROW_MIN = 10
ROW_MAX = 3000
SMALL_N = 100
DEFAULT_ROWS = 1000
SEED_MIN = 0
SEED_MAX = 100000
DEFAULT_SEED = 42
ALPHA_MIN = 0.005
ALPHA_MAX = 0.10
DEFAULT_ALPHA = 0.05


def _to_int(value: Any, default: int) -> tuple[int, bool]:
    try:
        return int(float(value)), True
    except (TypeError, ValueError):
        return default, False


def _to_float(value: Any, default: float) -> tuple[float, bool]:
    try:
        return float(value), True
    except (TypeError, ValueError):
        return default, False


def resolve_params(raw: Mapping[str, Any] | None) -> tuple[dict, list[str]]:
    """Validate raw (URL query) values. Returns (params, warnings)."""
    raw = dict(raw) if raw else {}
    warnings: list[str] = []

    scenario = str(raw.get("scenario", DEFAULT_SCENARIO)).lower()
    if scenario not in VALID_SCENARIOS:
        warnings.append(
            f"Ignoring unrecognized scenario {scenario!r}; using {DEFAULT_SCENARIO!r}."
        )
        scenario = DEFAULT_SCENARIO

    params: dict = {"scenario": scenario}
    for key in ("n_reference", "n_incoming"):
        n, ok = _to_int(raw.get(key, DEFAULT_ROWS), DEFAULT_ROWS)
        if not ok:
            warnings.append(f"Ignoring non-numeric {key}={raw.get(key)!r}; using {DEFAULT_ROWS}.")
            n = DEFAULT_ROWS
        elif n < ROW_MIN:
            warnings.append(
                f"{key}={n} is below the {ROW_MIN}-row minimum for a reliable "
                f"split and drift test; using {ROW_MIN}."
            )
            n = ROW_MIN
        elif n > ROW_MAX:
            warnings.append(f"{key}={n} exceeds the {ROW_MAX}-row cap; using {ROW_MAX}.")
            n = ROW_MAX
        params[key] = n

    seed, ok = _to_int(raw.get("seed", DEFAULT_SEED), DEFAULT_SEED)
    if not ok or not SEED_MIN <= seed <= SEED_MAX:
        warnings.append(f"Ignoring invalid seed={raw.get('seed')!r}; using {DEFAULT_SEED}.")
        seed = DEFAULT_SEED
    params["seed"] = seed

    alpha, ok = _to_float(raw.get("alpha", DEFAULT_ALPHA), DEFAULT_ALPHA)
    if not ok or not ALPHA_MIN <= alpha <= ALPHA_MAX:
        warnings.append(
            f"Ignoring alpha={raw.get('alpha')!r}; using {DEFAULT_ALPHA} "
            f"(allowed {ALPHA_MIN}–{ALPHA_MAX})."
        )
        alpha = DEFAULT_ALPHA
    params["alpha"] = alpha

    if params["n_reference"] < SMALL_N or params["n_incoming"] < SMALL_N:
        warnings.append(
            f"Small sample sizes (reference={params['n_reference']}, "
            f"incoming={params['n_incoming']}): metrics will be noisy, drift "
            "p-values less reliable, and tiny samples may contain only one "
            "class, which cannot be split. 300+ rows per side is recommended."
        )
    return params, warnings
