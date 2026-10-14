# Interview Guide — Drift Detective

## 30-second explanation

"I built Drift Detective, an educational prototype that monitors a server-risk
classifier. It generates synthetic monitoring data, trains a random forest, then
simulates normal, moderate, and severe incoming-data drift. A Streamlit dashboard
shows per-feature KS drift tests alongside labelled accuracy metrics, keeping
distribution change and performance change as two separate questions."

## 2-minute technical explanation

"The reference data has five numeric features — CPU, memory, disk, latency, error
rate — drawn from documented Gaussian distributions, with a fixed z-score label
rule so experiments are reproducible. I split the reference set with a stratified
80/20 split and train a median-imputation plus RandomForest pipeline, fitted on
training data only. For each incoming scenario I run a two-sample KS test per
feature with a Benjamini–Hochberg correction, and I only flag a feature when the
adjusted p-value passes alpha *and* Cohen's d reaches 0.2, so large-sample noise
doesn't cry wolf. The same frozen model is then scored on labelled incoming data
and the dashboard shows reference vs incoming accuracy, precision, recall, F1 and
confusion matrices next to the drift table, with Plotly histograms and feature
importances for explanation. Everything is computed live — no hard-coded numbers —
and pytest covers the generator, model, detector, and evaluation."

## Architecture walkthrough

```
synthetic data (data_generator.py)
  → stratified split + RandomForest pipeline (model.py)
  → metrics from predictions (evaluation.py)
  → per-feature KS + BH + effect size (drift_detector.py)
  → Streamlit dashboard (app.py): overview / metrics / drift / explainability
```

`src/` has no Streamlit imports, so the statistics are testable without the UI.
`app.py` is a thin presentation layer over cached data/model functions.

## Library choices

- scikit-learn RandomForest: strong tabular baseline, no scaling, importances.
- SciPy `ks_2samp`: non-parametric, no distribution assumption.
- Manual BH correction: avoids a statsmodels dependency for 10 lines of code.
- Streamlit + Plotly: fastest credible interactive dashboard.
- pytest: standard test runner.

## Key distinctions

- Data drift (inputs changed) vs concept drift (feature→label rule changed).
  This project simulates data drift only; the label rule is fixed.
- Drift detection (do the distributions differ?) vs performance evaluation
  (did accuracy drop?). Drift alone proves nothing about accuracy.

## Leakage prevention

Labels derive from same-row features only; split is stratified before training;
the pipeline (imputer + forest) fits on train only; incoming data never touches
training; no retraining after drift in the main experiment.

## Metrics

Accuracy = overall correctness; precision = of predicted-risk, how many truly
risk; recall = of true-risk, how many caught; F1 = harmonic mean. Confusion
matrices show the error pattern behind the averages. `zero_division=0` keeps
small-sample edge cases from crashing.

## Limitations & future work

Synthetic independent Gaussians; univariate KS only; fixed rule; no retraining
loop; no production concerns (latency, missing labels, alerting). Next steps:
multivariate drift (e.g. MMD), PSI comparison, calibration tracking, scheduled
batch evaluation.

## Likely questions (with answers)

1. **What problem does this solve?** Detecting when incoming data stops
   resembling training data, so silent model degradation gets noticed.
2. **Why synthetic data?** Reproducible, dependency-free, lets reviewers run the
   full experiment locally without downloads or credentials.
3. **Why RandomForest?** Robust tabular baseline, handles non-linearity without
   scaling, gives importances for the explainability panel.
4. **Why not logistic regression?** Valid alternative; forest fit the
   z-score-threshold rule better with less tuning, and importances are built in.
5. **What is the KS test?** Non-parametric comparison of two empirical CDFs;
   statistic = max vertical gap; sensitive to location and shape changes.
6. **What is the null hypothesis?** Both samples came from the same distribution.
7. **What does the p-value mean?** P(data this extreme | H0) — not P(H0 is true).
8. **Why adjust p-values?** Testing 5 features inflates false alarms; BH controls
   the false-discovery rate.
9. **Why require an effect size too?** Large n makes trivial shifts significant;
   Cohen's d ≥ 0.2 keeps flags operationally meaningful.
10. **Cohen's d here?** Standardised mean difference (ref − incoming) / pooled SD;
    0.2 small, 0.5 medium, 0.8 large by convention.
11. **How do you prevent leakage?** Stratified split first; pipeline fits on train
    only; incoming data only used for testing; labels are same-row features.
12. **Why stratify?** Keeps the ~20% positive rate equal in train and test so
    metrics aren't skewed by a lucky split.
13. **Drift but accuracy held — explain?** The shifted feature may be unimportant
    to the model, or the shift stayed on the same side of the decision boundary.
14. **Accuracy dropped but no drift flagged?** Possible concept drift or a
    multivariate shift the univariate KS missed — one reason both panels exist.
15. **How do you handle missing data?** Median imputer in the pipeline; detector
    drops NaN/non-finite per feature and requires ≥ 10 valid samples.
16. **Constant columns?** Handled explicitly: identical constants → no drift;
    different constants → drift, without calling the KS test on zero variance.
17. **How is this tested?** pytest: generator ranges/reproducibility/scenarios,
    model training/importances/invalid inputs, detector normal-vs-severe plus edge
    cases, evaluation metric bounds and comparison deltas.
18. **Why Streamlit?** Zero-frontend interactive dashboard; caching keeps
    regeneration fast; runs locally with no credentials.
