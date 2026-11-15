# Drift Detective: Technical Learning & Interview Readiness Guide

---

## 1. Project Overview & Problem Statement

### What problem does Drift Detective solve?
In production machine learning systems, models are trained on historical data under the assumption that future production data will follow the same probability distribution ($P_{\text{train}}(X) = P_{\text{deploy}}(X)$). However, in real-world environments, consumer habits shift, system loads fluctuate, sensor hardware degrades, and macroeconomic conditions change. This divergence is known as **Dataset Drift (Covariate Shift)**.

When incoming data drifts, a previously accurate model can quietly fail without throwing a single code exception or HTTP error. Ground-truth target labels ($y$) often arrive weeks or months later (e.g., loan defaults or churn), meaning teams cannot rely on live accuracy monitoring alone.

Drift Detective is an **explainable ML model-monitoring prototype** that demonstrates how to detect feature distribution drift *before* ground-truth labels are available. It generates reproducible synthetic server-monitoring telemetry, trains a Random Forest risk classifier, simulates three operational scenarios (Normal, Moderate, and Severe drift), detects per-feature distribution shifts using two-sample Kolmogorov-Smirnov (KS) tests, applies Benjamini-Hochberg False Discovery Rate correction, computes Cohen's d effect sizes, and provides an interactive Streamlit dashboard.

---

## 2. Architecture & Design Decisions

```
                           [ app.py (Streamlit UI) ]
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    [ src/data_generator.py ]  [ src/model.py ]    [ src/drift_detector.py ]
     Generates synthetic        Trains RandomForest   Runs 2-sample KS test
     Gaussian reference         pipeline with         per feature, computes
     and shifted incoming       stratified 80/20      BH adjusted p-values,
     telemetry                  train/test split      and Cohen's d effect size
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                           [ src/evaluation.py ]
                            Scoring & metrics
                            (Accuracy, Precision,
                             Recall, F1, Confusion Matrix)
```

### Why this architecture?
* **Separation of Drift from Performance:** A fundamental statistical truth in MLOps is that *distribution drift does not automatically mean model performance collapsed*. By keeping drift detection (`src/drift_detector.py`) separate from evaluation (`src/evaluation.py`), the project models real production conditions where drift flags trigger alerts for human review rather than naive knee-jerk model retraining.
* **Statistically Rigorous Multi-Hypothesis Testing:** Testing 5 or 50 features simultaneously multiplies False Positive rates. By including Benjamini-Hochberg FDR correction and effect-size gating ($|d| \ge 0.2$), it demonstrates mathematically grounded production MLOps practices.

---

## 3. Mathematical Foundations

### 1. Two-Sample Kolmogorov-Smirnov (KS) Test
The two-sample KS test is a non-parametric test comparing the empirical cumulative distribution functions (eCDFs) of two continuous datasets: $F_1(x)$ (reference) and $F_2(x)$ (incoming).
The KS statistic $D$ is the maximum vertical distance between the two eCDFs:
$$D = \sup_x |F_1(x) - F_2(x)|$$
Under the null hypothesis $H_0$, both samples are drawn from the same underlying distribution. If the p-value $< \alpha$, we reject $H_0$.

### 2. Benjamini-Hochberg (BH) False Discovery Rate Correction
When performing $m$ independent hypothesis tests (one for each feature), testing at significance level $\alpha = 0.05$ yields a family-wise error rate:
$$P(\ge 1\text{ false positive}) = 1 - (1 - \alpha)^m \approx 22.6\%\text{ for }m=5$$
The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR):
1. Sort raw p-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Compute adjusted critical values: $q_{(k)} = \frac{k}{m} \cdot Q$.
3. Compute adjusted p-values (Benjamini-Hochberg step-up):
   $$p_{\text{adj}(k)} = \min \left( 1, \min_{j \ge k} \left( \frac{m}{j} p_{(j)} \right) \right)$$

### 3. Cohen's d Effect Size
With large sample sizes ($N > 10,000$), the KS test will flag statistically significant p-values for trivial, practically meaningless distribution shifts (e.g., mean shifting from 50.0 to 50.05). Cohen's d quantifies the standardized difference between two group means:
$$d = \frac{\bar{x}_2 - \bar{x}_1}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$
* $|d| < 0.2$: Negligible shift.
* $0.2 \le |d| < 0.5$: Small practical shift.
* $0.5 \le |d| < 0.8$: Moderate shift.
* $|d| \ge 0.8$: Large shift.

**The Drift Flag Decision Rule:**
A feature is flagged as drifted if and only if:
$$\left(p_{\text{adj}} < \alpha \text{ AND } |d| \ge 0.2\right) \quad \text{OR} \quad |d| \ge 0.5$$

---

## 4. Key Files & Responsibilities

| File | Primary Responsibility |
|---|---|
| `src/params.py` | Validates user inputs (seed, alpha, row counts, scenario selection) and provides graceful fallbacks with visible warnings. |
| `src/data_generator.py` | Generates synthetic reference and shifted incoming datasets with documented Gaussians and fixed label boundaries. |
| `src/model.py` | Stratified 80/20 train/test split, median-imputation + RandomForest(200 trees) pipeline, global impurity feature importance. |
| `src/drift_detector.py` | Two-sample KS test calculation, Benjamini-Hochberg FDR correction, Cohen's d effect size, and decision rule. |
| `src/evaluation.py` | Classification metrics (Accuracy, Precision, Recall, F1, 2x2 confusion matrix preservation). |
| `app.py` | Streamlit interactive dashboard with scenario selectors, distribution overlays, and Plotly visualizations. |

---

## 5. Bugs Discovered & Fixed During Audit

1. **Misleading Explainability Scope:**
   * *Problem:* The original README and methodology described feature importance in a way that sounded like SHAP or local per-prediction explanations.
   * *Fix:* Clarified in `README.md` and `docs/methodology.md` that feature importance is global Gini impurity importance from scikit-learn's `RandomForestClassifier` (`feature_importances_`), not SHAP or local explanations.
2. **Missing `pyproject.toml` / PYTHONPATH Configuration:**
   * *Problem:* Pytest failed with `ModuleNotFoundError: No module named 'src'` when invoked outside the repository directory.
   * *Fix:* Added `pyproject.toml` with `pythonpath = ["."]` so imports resolve seamlessly regardless of current working directory.
3. **Missing Automated CI Workflow:**
   * *Problem:* Repository had zero GitHub Actions workflows.
   * *Fix:* Added `.github/workflows/ci.yml` running pytest across all PRs and pushes on Python 3.11.

---

## 6. 40 Technical Interview Questions & Answers

### Part A: Basic Fundamentals (15 Questions)

#### 1. What does Drift Detective do?
Drift Detective is an educational prototype and Streamlit dashboard that demonstrates statistical dataset drift detection in supervised machine learning. It compares incoming data against a training reference using two-sample Kolmogorov-Smirnov tests, Benjamini-Hochberg correction, and Cohen's d effect size to flag distribution shifts.

#### 2. What is the difference between Data Drift and Concept Drift?
* **Data Drift (Covariate Shift):** The input feature distribution changes ($P_{\text{deploy}}(X) \neq P_{\text{train}}(X)$), but the true mapping from $X$ to $y$ ($P(y|X)$) remains unchanged.
* **Concept Drift:** The statistical relationship between inputs and targets changes ($P_{\text{deploy}}(y|X) \neq P_{\text{train}}(y|X)$). For example, a transaction amount that was previously considered normal is now flagged as fraud due to changing fraud techniques.

#### 3. What is a Kolmogorov-Smirnov (KS) test?
The two-sample KS test is a non-parametric statistical hypothesis test that compares two continuous sample distributions. It computes the maximum vertical difference ($D$) between their cumulative distribution functions.

#### 4. What is the null hypothesis ($H_0$) of the KS test?
The null hypothesis states that both continuous samples were drawn from the identical continuous probability distribution.

#### 5. Why is the KS test described as "non-parametric"?
Because it makes no assumptions about the underlying parametric distribution (e.g., it does not assume data is normally distributed, Gaussian, exponential, or Poisson). It evaluates empirical ranks and cumulative probabilities.

#### 6. What does a p-value represent in this context?
The p-value is the probability of observing a difference at least as large as the computed KS statistic $D$ under the assumption that the reference and incoming data came from the same distribution. A small p-value ($< \alpha$) provides evidence to reject the null hypothesis.

#### 7. What is Cohen's d?
Cohen's d is an effect size metric that measures the standardized difference between two means, expressed in units of pooled standard deviation.

#### 8. Why do we need Cohen's d in addition to the p-value?
With large sample sizes (e.g., $N = 10,000$), statistical tests have extreme power and will yield tiny p-values ($p < 0.0001$) for minuscule, practically irrelevant differences. Cohen's d prevents false alarms by ensuring only meaningful shifts ($|d| \ge 0.2$) trigger drift warnings.

#### 9. What machine learning model is trained in this project?
A `RandomForestClassifier` with 200 trees, `min_samples_split=5`, and `random_state=42`, preceded by a `SimpleImputer(strategy="median")`.

#### 10. How are the synthetic features generated?
Five continuous features representing server health metrics are generated using Gaussian distributions (`np.random.normal`): `cpu_utilization`, `memory_pressure`, `disk_io_latency`, `network_throughput_ratio`, and `thread_queue_depth`.

#### 11. How is the binary risk label assigned?
The label `risk_label` (0 = healthy, 1 = at risk) is determined by a deterministic weighted z-score combination of the features plus a small logistic threshold, representing server incident risk.

#### 12. What scenarios does the project simulate?
- **Normal Scenario:** Incoming data drawn from the exact same distributions as reference (mean shift = 0).
- **Moderate Scenario:** Shifts means of 3 features (`cpu_utilization`, `memory_pressure`, `thread_queue_depth`) by moderate amounts ($+0.4\sigma$ to $+0.6\sigma$).
- **Severe Scenario:** Shifts all 5 features substantially ($+1.0\sigma$ to $+1.5\sigma$).

#### 13. How does the dashboard visualize distributions?
Using interactive Plotly histogram and KDE overlays that plot reference data in blue and incoming data in orange, showing shifts in the distribution curves.

#### 14. What tech stack is used?
Python 3.11+, Streamlit, Plotly, Scikit-learn, SciPy (`scipy.stats.ks_2samp`), NumPy, Pandas, and Pytest.

#### 15. How do you run the tests?
`pytest -v` from the project directory. All 39 tests pass deterministically.

---

### Part B: Intermediate Statistics & Engineering (15 Questions)

#### 16. What is the multiple testing problem (family-wise error rate)?
When testing multiple independent hypotheses simultaneously (e.g., testing 5 features at $\alpha = 0.05$), the probability of obtaining at least one false positive purely by chance increases: $1 - (1 - 0.05)^5 \approx 22.6\%$. As the number of features grows to 50 or 100, false positives are virtually guaranteed without correction.

#### 17. How does the Benjamini-Hochberg (BH) procedure differ from the Bonferroni correction?
Bonferroni controls the Family-Wise Error Rate (FWER) by dividing $\alpha$ by the number of tests: $\alpha' = \alpha / m$. It is extremely conservative and severely reduces statistical power, causing true drift to be missed. Benjamini-Hochberg controls the False Discovery Rate (FDR)—the expected proportion of false positives among all rejected null hypotheses—providing a balanced, powerful detection threshold.

#### 18. How does `src/drift_detector.py` implement Benjamini-Hochberg correction?
1. Extracts raw two-sample KS p-values for all features.
2. Sorts p-values in ascending order while tracking original indices.
3. Calculates adjusted p-values using the step-up formula: $p_{\text{adj}(k)} = \min(1, \frac{m}{k} \cdot p_{(k)})$.
4. Enforces monotonicity backwards: $p_{\text{adj}(k)} = \min(p_{\text{adj}(k)}, p_{\text{adj}(k+1)})$.
5. Re-maps adjusted p-values back to the original feature ordering.

#### 19. What is the explicit drift decision rule in `evaluate_drift()`?
A feature is flagged as drifted if:
`((adj_p < alpha) and (abs(d) >= 0.2)) or (abs(d) >= 0.5)`
This ensures that tiny sample fluctuations with small p-values do not trigger alerts unless practical effect size is meaningful ($|d| \ge 0.2$), while substantial shifts ($|d| \ge 0.5$) are flagged even if sample size is modest.

#### 20. Why does Drift Detective use global Random Forest feature importance rather than SHAP?
`RandomForestClassifier.feature_importances_` measures Mean Decrease in Impurity (Gini importance) computed during tree construction. It is fast and built-in, providing a global ranking of which features contributed most to split purity. SHAP computes Shapley values from cooperative game theory to explain individual predictions; computing SHAP values over thousands of samples adds substantial latency and overhead, which was unnecessary for this global prototype.

#### 21. What is the limitation of Gini impurity-based feature importance?
Gini importance is biased toward high-cardinality continuous features with many split opportunities, even if they are noise. It also only provides a global summary, not local per-row explanations, and does not indicate the direction of impact (positive or negative).

#### 22. How does `src/evaluation.py` preserve 2x2 confusion matrices when a class is absent?
In heavily drifted or small evaluation slices, an incoming dataset might contain only Class 0 (no positive instances). Standard `confusion_matrix(y_true, y_pred)` would collapse into a 1x1 scalar or 1x2 array, crashing downstream indexing. `evaluate_model()` passes `labels=[0, 1]` explicitly:
`confusion_matrix(y_true, y_pred, labels=[0, 1])`
guaranteeing a consistent 2x2 matrix shape `[[TN, FP], [FN, TP]]` at all times.

#### 23. Why is data scaling / normalization not required for Random Forest?
Decision trees make axis-aligned partition splits based on feature ordering (ranks), not distances. Multiplying a feature by 1,000 or shifting its mean does not change the relative ordering of values, so tree models are invariant to monotonic monotonic transformations.

#### 24. How does `src/params.py` handle invalid URL query parameters?
When users pass query parameters like `?scenario=unknown&rows=abc`, `parse_dashboard_params()` validates inputs: clamps row counts to safe bounds [100, 20000], falls back to default scenarios, and surfaces a visible warning in the Streamlit UI rather than crashing the application.

#### 25. What happens when a feature in the dataset is completely constant (zero variance)?
In `test_constant_columns_no_crash()`, if a column has identical values across all rows, variance is zero and Cohen's d denominator is zero. The detector handles zero variance by catching division by zero and returning $d = 0.0$ and $p = 1.0$, preventing runtime crashes.

#### 26. How does the data generator prevent data leakage during model training?
The model is trained strictly on the reference training split (80% of reference data). Feature imputer and model weights are frozen; the incoming data stream is strictly an out-of-sample evaluation set that the model has never seen.

#### 27. Why does drift in features not always cause accuracy to drop?
If a drifted feature has low predictive importance (e.g., `thread_queue_depth` had only 5% importance), or if the shift occurs within a region where the model already predicts the correct class with high margin, accuracy remains intact. Drift Detective explicitly demonstrates this in the Moderate scenario.

#### 28. How does `test_bh_correction` verify statistical correctness?
It tests synthetic p-values with known analytical Benjamini-Hochberg outputs, ensuring that adjusted p-values are strictly bounded between 0 and 1, monotonically increasing, and accurately match hand-calculated FDR adjustments.

#### 29. What is Population Stability Index (PSI), and how does it compare to KS?
PSI is an alternative drift metric common in banking and credit scoring. It bins features into deciles and calculates symmetric Kullback-Leibler divergence: $\sum (P_i - Q_i) \ln(P_i / Q_i)$. While PSI requires binning choices, KS is continuous and bin-free.

#### 30. How does `pyproject.toml` resolve import path issues?
By configuring `pythonpath = ["."]`, Pytest adds the root directory to `sys.path`, allowing `from src.data_generator import ...` to succeed whether tests are run from the repository root, CI, or an IDE.

---

### Part C: Advanced MLOps & Production Architecture (10 Questions)

#### 31. What is the fundamental difference between univariate and multivariate drift detection?
* **Univariate (KS test):** Tests each feature in isolation ($P(X_1), P(X_2)$). It cannot detect shifts in correlation between features. For example, if both $X_1$ and $X_2$ remain in their normal ranges, but $X_1$ is high whenever $X_2$ is high (violating their historical inverse relationship), univariate KS tests will miss the drift entirely.
* **Multivariate (MMD / PCA reconstruction):** Evaluates the joint distribution $P(X_1, X_2, \dots, X_n)$ using Maximum Mean Discrepancy, isolation forests, or autoencoder reconstruction errors.

#### 32. In a production pipeline processing 100,000 events per second, how would you implement drift monitoring?
Running exact two-sample KS tests on 100k events/sec is computationally prohibitive.
Production solution:
1. **Streaming Reservoir Sampling:** Maintain a rolling reservoir buffer (e.g., 5,000 samples) of incoming features.
2. **Approximate Quantile Sketches:** Use streaming algorithms like t-digest or KLL sketches to compute approximate eCDFs in sub-linear memory.
3. **Scheduled Micro-Batches:** Compute KS tests or Wasserstein distance once every hour or day against the baseline reference sketch.

#### 33. When should a drifted model be retrained, and when should it NOT be retrained?
* **Retrain:** When ground-truth labels confirm concept drift or performance degradation, and the new data distribution represents the new steady state.
* **Do NOT Retrain:**
  1. Temporary anomalies (e.g., Black Friday traffic surge or temporary upstream sensor glitch); retraining would overfit to temporary outliers.
  2. Data pipeline corruptions (e.g., an upstream schema change sending millimeters instead of meters); retraining on corrupt inputs cements the bug.

#### 34. What is Maximum Mean Discrepancy (MMD)?
MMD is a kernel-based statistical test that evaluates the distance between two probability distributions in a Reproducing Kernel Hilbert Space (RKHS). It computes distance between all pairwise kernel evaluations, detecting multivariate shifts without binning.

#### 35. Explain how concept drift can occur without any covariate drift.
Suppose a credit card fraud model monitors transaction amount and hour of day. Their distributions ($P(X)$) remain identical. However, fraudsters discover a new exploit that allows them to steal money using normal transaction amounts at normal hours. The probability of fraud given the inputs ($P(y=1|X)$) has dramatically increased, representing pure concept drift with zero data drift.

#### 36. How does Drift Detective handle sample size discrepancies between reference and incoming data?
The two-sample KS test statistic distribution accounts for differing sample sizes:
$$D_{\text{crit}} = c(\alpha) \sqrt{\frac{n_1 + n_2}{n_1 \cdot n_2}}$$
The SciPy implementation `scipy.stats.ks_2samp` natively incorporates both sample counts $n_1$ and $n_2$ when computing the exact asymptotic p-value.

#### 37. What is Model Calibration, and how does it relate to drift?
Model calibration measures whether predicted probabilities match true empirical probabilities (e.g., among cases assigned a 70% probability of risk, 70% actually experience an incident). Under dataset drift, predictions often remain ordered correctly (preserving ROC-AUC), but probabilities become severely miscalibrated, breaking downstream cost-benefit thresholds.

#### 38. What is the difference between Label Shift and Covariate Shift?
* **Covariate Shift:** $P(X)$ changes while $P(y|X)$ stays the same (e.g., more younger applicants apply, but default rates for any given age remain identical).
* **Label Shift (Prior Probability Shift):** $P(y)$ changes while $P(X|y)$ stays the same (e.g., during a depression, overall default rates rise, but the profile of someone who defaults remains the same).

#### 39. What are the key limitations of this prototype?
1. Synthetic data: Uses independent Gaussians; does not exhibit real-world correlations or seasonality.
2. Univariate only: Does not evaluate multivariate interaction drift.
3. Fixed label rule: Simulates covariate drift only, not concept drift.

#### 40. What are the key takeaways from this engineering overhaul?
* Accurate documentation: Never claim global impurity importance is SHAP or explainable AI.
* Clean configuration: Ensure `pyproject.toml` allows tests to run anywhere.
* Automated CI: GitHub Actions verification ensures continuous quality.

---

## 7. Summary Table: Verified Test Suite

| Test Module | Tests | Status | Scope |
|---|---|---|---|
| `tests/test_data_generator.py` | 9 | PASSED | Reference schema, feature ranges, label balance, reproducibility, scenario shifts |
| `tests/test_drift_detector.py` | 11 | PASSED | Normal/moderate/severe drift, zero-variance handling, Benjamini-Hochberg correction |
| `tests/test_evaluation.py` | 5 | PASSED | Metrics generation, confusion matrix 2x2 shape preservation under missing classes |
| `tests/test_model.py` | 5 | PASSED | Stratified splits, RF training & prediction, impurity feature importances |
| `tests/test_params.py` | 9 | PASSED | Dashboard query parameter validation, row count clamping, fallback warnings |
| **Total** | **39** | **ALL PASSED** | **100% test pass rate across 39 automated checks** |
