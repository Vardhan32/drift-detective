# Methodology

This document describes exactly what the Drift Detective prototype does. It is kept
in sync with `src/` — if the code changes, update this file too.

## 1. Synthetic data (`src/data_generator.py`)

Reference distributions (independent draws, then clipped):

| feature | distribution | clip |
|---|---|---|
| `cpu_usage` (%) | Normal(50, 15) | [0, 100] |
| `memory_usage` (%) | Normal(60, 12) | [0, 100] |
| `disk_usage` (%) | Normal(55, 15) | [0, 100] |
| `request_latency_ms` (ms) | Normal(120, 40) | [10, 2000] |
| `error_rate` (fraction) | Normal(0.03, 0.02) | [0, 1] |

Deterministic via `numpy.random.default_rng(seed)`. Default: 1,000 reference rows
(seed 42). Same generator, new seed = "normal" incoming data.

Label rule (synthetic, not real incidents):

```
z_cpu = (cpu_usage - 50) / 15
z_mem = (memory_usage - 60) / 12
z_lat = (request_latency_ms - 120) / 40
z_err = (error_rate - 0.03) / 0.02
risk_score = 0.5*z_cpu + 0.5*z_mem + 0.7*z_lat + 0.6*z_err
risk_label = 1 if risk_score > 1.0 else 0
```

Positive (high-risk) rate on reference data is roughly 15–25%. The same rule is
applied to reference and all incoming scenarios so labelled comparisons are
like-for-like. There is no target leakage: labels are derived only from the five
feature values in the same row.

## 2. Drift scenarios

| scenario | changes vs reference |
|---|---|
| normal | none (fresh seed only) |
| moderate | cpu mean 50→60; latency mean 120→150; error mean 0.03→0.045 |
| severe | cpu 50→72; memory 60→78; disk 55→70; latency 120→220, sd 40→60; error 0.03→0.09 |

## 3. Train/test design (`src/model.py`, `src/evaluation.py`)

- `train_test_split(test_size=0.2, stratify=y, random_state=42)` on reference data.
- Model: `Pipeline(SimpleImputer(median) → RandomForestClassifier(n_estimators=200))`.
  Random forests need no scaling; the imputer handles missing values. The pipeline
  is fitted on training data only.
- Metrics from `sklearn.metrics` on actual predictions: accuracy, precision,
  recall, F1 (`zero_division=0`), confusion matrix.
- Synthetic metrics reflect the artificial label rule, not real-world accuracy.

## 4. Drift detection (`src/drift_detector.py`)

Per numeric feature, two-sample Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`):

- H0: reference and incoming samples come from the same distribution.
- KS statistic in [0, 1]: max gap between empirical CDFs.
- p-value = P(difference this large | H0), not P(H0 is true).

Reported per feature: sample counts, means, stds, median change, KS statistic,
raw p-value, Benjamini–Hochberg adjusted p-value, Cohen's d, flag.

Benjamini–Hochberg (implemented manually, no extra dependency): sort p-values
p(1)≤…≤p(m), adjusted p(i) = p(i)·m/i, then cumulative minimum from the top.
Controls the false-discovery rate across the 5 features.

Flag rule: `drifted = (adj_p < alpha AND |d| >= 0.2) OR (|d| >= 0.5)`.
Overall: drift if ≥ 1 feature flagged. Minimum 10 valid (finite, non-NaN) samples
per feature; infinite values dropped; constant columns handled explicitly;
non-numeric or missing columns raise `ValueError`.

## 5. Evaluation under drift

The reference-trained model is evaluated (never retrained) on labelled incoming
data per scenario. Dashboard shows reference held-out metrics, incoming metrics,
absolute deltas, and the drift table side by side. No claim is made about real
production behaviour.

## 6. Limitations

- Independent Gaussian features: no correlations, seasonality, or regimes.
- Fixed label rule: demonstrates data drift only, not concept drift.
- KS test is univariate and insensitive to some multivariate shifts.
- Large n makes trivial shifts "significant" — hence the effect-size gate.
- Labels for incoming data are a synthetic convenience; production labels are
  usually delayed or missing.
