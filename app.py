"""Drift Detective — educational prototype dashboard.

Monitors a RandomForest server-risk classifier under synthetic data drift.
Distribution monitoring (KS tests) and labelled performance evaluation are
kept separate: drift does not automatically mean accuracy dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_generator import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    generate_incoming_data,
    generate_reference_data,
)
from src.drift_detector import DEFAULT_ALPHA, detect_drift
from src.evaluation import compare_metrics, evaluate_model
from src.model import get_feature_importance, split_data, train_model

st.set_page_config(page_title="Drift Detective", layout="wide")

st.title("Drift Detective")
st.caption(
    "An educational prototype that shows how a supervised model behaves when "
    "incoming data drifts away from its training reference. Synthetic data only — "
    "not a production monitoring platform."
)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Experiment controls")
    scenario = st.selectbox(
        "Incoming-data scenario",
        options=["normal", "moderate", "severe"],
        index=1,
        help="normal ≈ reference; moderate = some features shifted; severe = many features shifted.",
    )
    alpha = st.slider(
        "Significance threshold (alpha)",
        min_value=0.005,
        max_value=0.10,
        value=float(DEFAULT_ALPHA),
        step=0.005,
        help="Per-feature flag uses Benjamini-Hochberg adjusted p-values.",
    )
    n_reference = st.slider("Reference rows", 300, 3000, 1000, step=100)
    n_incoming = st.slider("Incoming rows", 300, 3000, 1000, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    st.info(
        "Same label rule is applied to every dataset so the "
        "reference-vs-incoming accuracy comparison is like-for-like."
    )

# ---------------- Data + model (all computed, nothing hard-coded) ----------------
@st.cache_data(show_spinner="Generating synthetic data...")
def _load_data(n_ref: int, n_inc: int, sc: str, sd: int):
    ref = generate_reference_data(n_rows=n_ref, seed=int(sd))
    inc = generate_incoming_data(scenario=sc, n_rows=n_inc, seed=int(sd) + 1000)
    return ref, inc


@st.cache_resource(show_spinner="Training reference model...")
def _train_cached(train_values: tuple, train_labels: tuple, seed: int):
    import pandas as _pd

    X = _pd.DataFrame([list(r) for r in train_values], columns=FEATURE_COLUMNS)
    y = _pd.Series(list(train_labels))
    return train_model(X, y, random_state=int(seed))


try:
    reference_df, incoming_df = _load_data(n_reference, n_incoming, scenario, seed)
except Exception as exc:
    st.error(f"Data generation failed: {exc}")
    st.stop()

try:
    X_train, X_test, y_train, y_test = split_data(reference_df, random_state=int(seed))
except Exception as exc:
    st.error(f"Train/test split failed: {exc}")
    st.stop()

model = _train_cached(
    tuple(map(tuple, X_train.to_numpy().tolist())),
    tuple(y_train.tolist()),
    int(seed),
)

try:
    ref_metrics = evaluate_model(model, X_test, y_test)
    inc_metrics = evaluate_model(model, incoming_df[FEATURE_COLUMNS], incoming_df[LABEL_COLUMN])
    comparison_df = compare_metrics(ref_metrics, inc_metrics)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

try:
    drift_out = detect_drift(reference_df, incoming_df, alpha=float(alpha))
    drift_table: pd.DataFrame = drift_out["results"]
except Exception as exc:
    st.error(f"Drift detection failed: {exc}")
    st.stop()

try:
    importances = get_feature_importance(model)
except Exception:
    importances = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": 0.0})

# ---------------- Overview ----------------
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scenario", scenario)
c2.metric("Reference rows", f"{len(reference_df):,}")
c3.metric("Incoming rows", f"{len(incoming_df):,}")
status = drift_out["status"]
c4.metric(
    "Overall drift status",
    status,
    delta=f"{drift_out['n_flagged']}/{drift_out['n_features']} features flagged",
    delta_color="inverse" if drift_out["overall_drift"] else "normal",
)
if drift_out["overall_drift"]:
    st.warning(
        f"Drift detected in {drift_out['n_flagged']} of {drift_out['n_features']} features "
        f"(alpha={alpha:.3f}, BH-adjusted). This flags a distribution change — "
        "check the labelled metrics below before concluding accuracy dropped."
    )
else:
    st.success(
        "No significant drift flagged. Incoming distributions look consistent with the reference."
    )

# ---------------- ML evaluation ----------------
st.subheader("Model evaluation: reference held-out vs incoming")
st.caption(
    "Same trained reference model, evaluated on two labelled sets. "
    "Metrics are computed from actual predictions each run."
)
m1, m2, m3, m4 = st.columns(4)
for col, key, label in zip(
    (m1, m2, m3, m4),
    ("accuracy", "precision", "recall", "f1"),
    ("Accuracy", "Precision", "Recall", "F1-score"),
):
    col.metric(
        f"Incoming {label}",
        f"{inc_metrics[key]:.3f}",
        delta=f"{inc_metrics[key] - ref_metrics[key]:+.3f} vs reference",
        delta_color="normal",
    )

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Reference vs incoming (absolute delta = incoming − reference)**")
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.markdown("**Reference confusion matrix (held-out test)**")
    st.dataframe(
        pd.DataFrame(
            ref_metrics["confusion_matrix"],
            index=["actual 0", "actual 1"],
            columns=["pred 0", "pred 1"],
        )
    )
with col_b:
    st.markdown("**Incoming confusion matrix (same model, drifted data)**")
    st.dataframe(
        pd.DataFrame(
            inc_metrics["confusion_matrix"],
            index=["actual 0", "actual 1"],
            columns=["pred 0", "pred 1"],
        )
    )
    st.caption(
        f"Reference n={ref_metrics['n_samples']}, "
        f"incoming n={inc_metrics['n_samples']}. "
        "Synthetic labels follow the documented rule; they do not represent real incidents, "
        "and these numbers say nothing about real-world accuracy."
    )

# ---------------- Drift analysis ----------------
st.subheader("Drift analysis (per feature)")
st.caption(
    "Two-sample Kolmogorov–Smirnov test per feature with Benjamini–Hochberg correction. "
    f"Flag rule: (adjusted p < {alpha:.3f} and |Cohen's d| ≥ 0.2) or (|Cohen's d| ≥ 0.5)."
)
display = drift_table.copy()
for c in ["mean_reference", "mean_incoming", "std_reference", "std_incoming",
          "median_change", "ks_statistic", "p_value", "p_value_adj", "cohens_d"]:
    display[c] = display[c].round(4)
st.dataframe(display, width="stretch", hide_index=True)

st.markdown("**Distribution comparison (reference vs incoming)**")
tabs = st.tabs(FEATURE_COLUMNS)
for tab, feat in zip(tabs, FEATURE_COLUMNS):
    with tab:
        combined = pd.DataFrame(
            {
                "value": pd.concat(
                    [reference_df[feat], incoming_df[feat]], ignore_index=True
                ),
                "source": (["reference"] * len(reference_df))
                + (["incoming"] * len(incoming_df)),
            }
        )
        fig = px.histogram(
            combined,
            x="value",
            color="source",
            barmode="overlay",
            opacity=0.55,
            nbins=40,
            title=f"{feat}: reference vs incoming ({scenario})",
        )
        st.plotly_chart(fig, width="stretch")
        row = drift_table.set_index("feature").loc[feat]
        st.write(
            f"KS={row['ks_statistic']:.3f}, raw p={row['p_value']:.4g}, "
            f"adjusted p={row['p_value_adj']:.4g}, Cohen's d={row['cohens_d']:+.3f} → "
            f"{'**drifted**' if row['drifted'] else 'no flag'}."
        )

# ---------------- Explainability ----------------
st.subheader("Explainability")
ce1, ce2 = st.columns(2)
with ce1:
    st.markdown("**Model feature importance (RandomForest, reference training data)**")
    fig_imp = px.bar(
        importances, x="feature", y="importance", title="Feature importance"
    )
    st.plotly_chart(fig_imp, width="stretch")
with ce2:
    st.markdown("**Most changed features (by |Cohen's d|)**")
    ranked = drift_table.reindex(
        drift_table["cohens_d"].abs().sort_values(ascending=False).index
    )[["feature", "cohens_d", "ks_statistic", "p_value_adj", "drifted"]]
    st.dataframe(ranked, width="stretch", hide_index=True)
    top = ranked.iloc[0]
    st.info(
        f"Largest shift: **{top['feature']}** (Cohen's d={top['cohens_d']:+.3f}). "
        "This is a distributional observation, not a causal claim — "
        "we cannot say this feature *caused* any accuracy change from this table alone."
    )

# ---------------- Educational section ----------------
with st.expander("Learn: what do these numbers mean?"):
    st.markdown(
        """
**Data drift** = the incoming feature distribution differs from the reference
training distribution. **Concept drift** = the relationship between features and
the label itself changes (this prototype keeps the label rule fixed, so it
demonstrates data drift only).

**KS test**: the two-sample Kolmogorov–Smirnov test compares the full shapes of
two samples. Its null hypothesis is that both came from the same distribution.
The KS statistic (0–1) is the largest gap between the two cumulative curves;
bigger means more different.

**p-values**: a p-value is the probability of seeing a difference this large
*if the null were true* — it is not the probability that the null is true.
With 1,000+ rows, tiny unimportant shifts can give tiny p-values, which is why
this app also requires a minimum effect size (Cohen's d ≥ 0.2) before flagging.

**Multiple testing**: testing 5 features at alpha=0.05 gives more than a 5%
chance of at least one false alarm. The Benjamini–Hochberg correction adjusts
p-values upward to control the false-discovery rate.

**Drift ≠ broken model**: drift says the inputs changed; only labelled
evaluation says whether predictions got worse. Always read the two sections
together. A drifted feature the model barely uses may not hurt accuracy, and
accuracy can fall (concept drift) with no visible feature drift.
"""
    )
