"""Shared Streamlit helpers used by every page (model loading, notices, thresholds)."""
from __future__ import annotations

import streamlit as st

from src.config import DEFAULT_STRONG_THRESHOLD, DEFAULT_SUPPORT_THRESHOLD, MODELS_DIR
from src.data import generate_synthetic_data
from src.predict import SETUP_COMMANDS, ModelNotFoundError, load_bundle

CSS = """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 16px; color: #fff;
             background: linear-gradient(120deg, #4338CA 0%, #0EA5E9 100%); margin-bottom: 1.2rem;}
      .hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 2rem;}
      .hero p {margin: 0; opacity: .95; font-size: 1.02rem;}
      .badge {display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-size: .8rem;
              font-weight: 600; margin-top: .6rem; background: rgba(255,255,255,.2); color: #fff;}
      .band {padding: .9rem 1.1rem; border-radius: 12px; border-left: 6px solid; background: #F8FAFC;}
      .band h3 {margin: 0 0 .25rem 0; font-size: 1.15rem;}
      .small-note {color: #64748B; font-size: .85rem;}
    </style>
"""


@st.cache_resource(show_spinner="Loading model…")
def get_bundle():
    """Loaded once per server process; never retrained on user interaction."""
    return load_bundle(MODELS_DIR)


def build_demo_model() -> None:
    """Explicit, user-triggered build of the SYNTHETIC demo model (only if artifacts are missing)."""
    from src.train import save_artifacts, train_and_evaluate

    df = generate_synthetic_data()
    pipeline, metadata = train_and_evaluate(df, dataset_type="synthetic",
                                            dataset_name="students_synthetic (generated in app)")
    save_artifacts(pipeline, metadata, MODELS_DIR)


def require_bundle():
    try:
        return get_bundle()
    except ModelNotFoundError as exc:
        st.error("**Model artifact not found.** The app needs a trained model before it can predict.")
        st.code(SETUP_COMMANDS, language="bash")
        st.caption(str(exc).splitlines()[0])
        if st.button("Build the synthetic demo model now (seed 42, a few seconds)", type="primary"):
            with st.spinner("Generating synthetic data and training…"):
                build_demo_model()
            get_bundle.clear()
            st.rerun()
        st.stop()


def thresholds() -> tuple[float, float]:
    lo = st.session_state.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD)
    hi = st.session_state.get("strong_threshold", DEFAULT_STRONG_THRESHOLD)
    if lo >= hi:  # invalid slider combination -> fall back to defaults
        return DEFAULT_SUPPORT_THRESHOLD, DEFAULT_STRONG_THRESHOLD
    return lo, hi


def dataset_badge(meta: dict) -> str:
    return "SYNTHETIC training data" if meta.get("dataset_type") == "synthetic" else "REAL training data"


def synthetic_warning(meta: dict) -> None:
    if meta.get("dataset_type") == "synthetic":
        st.warning("This model was trained on **synthetic (computer-generated) data**. Its metrics show "
                   "how well it learned the invented patterns, not how accurate it would be for real students.",
                   icon="⚠️")


def responsible_use_note() -> None:
    st.info("Predictions are **estimates, not guarantees**. They must not be used to decide admissions, "
            "grades or disciplinary action. Use them only to start supportive conversations.", icon="ℹ️")
