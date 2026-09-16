"""Overview page: purpose, data type and headline test metrics."""
from __future__ import annotations

import streamlit as st

from src import charts
from src.config import PREDICTION_TIME
from src.ui import dataset_badge, require_bundle, responsible_use_note, synthetic_warning


def render():
    bundle = require_bundle()
    meta = bundle.metadata
    sel, base = meta["test_metrics"]["selected"], meta["test_metrics"]["baseline"]

    st.markdown(
        f"""<div class="hero"><h1>🎓 EduPredict</h1>
        <p>AI-driven estimate of a student's final exam score (0–100) from information available
        before the exam, paired with supportive, rule-based study suggestions.</p>
        <span class="badge">{dataset_badge(meta)}</span></div>""",
        unsafe_allow_html=True,
    )
    synthetic_warning(meta)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", sel["model"].split(" (")[0], help=sel["model"])
    c2.metric("Test MAE", f"{sel['mae']:.2f} pts", delta=f"{sel['mae'] - base['mae']:.2f} vs baseline",
              delta_color="inverse")
    c3.metric("Test RMSE", f"{sel['rmse']:.2f} pts", delta=f"{sel['rmse'] - base['rmse']:.2f} vs baseline",
              delta_color="inverse")
    c4.metric("Test R²", f"{sel['r2']:.3f}", delta=f"{sel['r2'] - base['r2']:+.3f} vs baseline")
    st.caption(f"Metrics from a single evaluation on {meta['split']['test_rows']} held-out test records. "
               f"Trained {meta['trained_at_utc']} on {meta['split']['total_rows']:,} records.")

    left, right = st.columns([1, 1])
    with left:
        with st.container(border=True):
            st.subheader("What it does")
            st.markdown(
                "- **Individual prediction** – enter one student's week-10 information.\n"
                "- **Batch predictions** – upload a CSV and download results.\n"
                "- **Model insights** – compare models, inspect errors and feature importance.\n"
                "- **Support suggestions** – transparent rules that point to helpful next steps."
            )
            st.markdown(f"**Prediction time:** {PREDICTION_TIME}")
        responsible_use_note()
    with right:
        tp = meta["test_predictions"]
        st.plotly_chart(charts.actual_vs_predicted(tp["actual"], tp["predicted"]), width="stretch")

    left, right = st.columns(2)
    with left:
        h = meta["target_histogram"]
        st.plotly_chart(charts.target_histogram(h["counts"], h["edges"]), width="stretch")
    with right:
        st.plotly_chart(charts.metric_comparison(list(meta["test_metrics"].values()), "mae",
                                                 "Test MAE: selected model vs mean baseline (lower is better)"),
                        width="stretch")


render()
