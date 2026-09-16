"""Batch predictions page: CSV upload (in memory), validation report and export."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import charts
from src.config import DATA_DIR, FEATURE_NAMES, MAX_BATCH_ROWS
from src.data import generate_synthetic_data, make_prediction_sample
from src.predict import predict_batch
from src.recommendations import support_band
from src.ui import dataset_badge, require_bundle, responsible_use_note, thresholds


def _sample_csv_bytes() -> bytes:
    path = DATA_DIR / "sample_students.csv"
    if path.exists():
        return path.read_bytes()
    return make_prediction_sample(generate_synthetic_data(n_students=200)).to_csv(index=False).encode()


def render():
    bundle = require_bundle()
    st.title("Batch predictions")
    st.caption(f"{dataset_badge(bundle.metadata)} · Uploads are processed in memory and are not saved. "
               "The model is never retrained on uploaded files.")

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in FEATURE_NAMES) +
                    f"  \nOptional: `student_id` or any other columns (kept in the output). "
                    f"Blank cells are allowed. Up to {MAX_BATCH_ROWS:,} rows.")
        c2.download_button("Download sample CSV", _sample_csv_bytes(), "sample_students.csv", "text/csv",
                           icon=":material/download:", width="stretch")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.markdown('<p class="small-note">No file uploaded yet. Download the sample CSV above to try it.</p>',
                    unsafe_allow_html=True)
        return

    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read this file as CSV: {exc}")
        return

    results, report = predict_batch(bundle.pipeline, df)
    if results is None:
        if report.missing_columns:
            st.error("Missing required column(s): " + ", ".join(f"`{c}`" for c in report.missing_columns))
            st.caption("Found columns: " + ", ".join(f"`{c}`" for c in df.columns))
        for msg in report.general_errors:
            st.error(msg)
        return

    if report.issues:
        st.warning(f"{len(report.invalid_rows)} of {report.n_rows} row(s) contain invalid values and were "
                   "skipped. Row numbers start at 0 (the first data row after the header).")
        with st.expander("Show invalid values", expanded=True):
            st.dataframe(report.issues_frame(), hide_index=True, width="stretch")

    if results.empty:
        st.error("No valid rows to predict.")
        return

    lo, hi = thresholds()
    results["support_band"] = [support_band(s, lo, hi).label for s in results["predicted_final_exam_score"]]
    st.success(f"Predicted {len(results)} student(s).")

    counts = results["support_band"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average estimate", f"{results['predicted_final_exam_score'].mean():.1f}")
    m2.metric("Additional support suggested", int(counts.get("Additional support suggested", 0)))
    m3.metric("Progressing", int(counts.get("Progressing", 0)))
    m4.metric("Strong progress", int(counts.get("Strong progress", 0)))

    st.plotly_chart(charts.batch_distribution(results["predicted_final_exam_score"], lo, hi),
                    width="stretch")
    st.dataframe(results, hide_index=True, width="stretch")
    st.download_button("Download predictions CSV", results.to_csv(index=False).encode(),
                       "edupredict_predictions.csv", "text/csv", type="primary", icon=":material/download:")
    responsible_use_note()


render()
