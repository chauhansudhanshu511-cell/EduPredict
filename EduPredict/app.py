"""EduPredict — Streamlit entry point (page config, sidebar and navigation).

Run with:  streamlit run app.py
Each page lives in views/; shared helpers are in src/ui.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src.config import DEFAULT_STRONG_THRESHOLD, DEFAULT_SUPPORT_THRESHOLD  # noqa: E402
from src.predict import ModelNotFoundError  # noqa: E402
from src.ui import CSS, dataset_badge, get_bundle  # noqa: E402

st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- navigation & sidebar
pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True, url_path="overview"),
    st.Page("views/individual_prediction.py", title="Individual Prediction", icon=":material/person_search:", url_path="predict"),
    st.Page("views/batch_predictions.py", title="Batch Predictions", icon=":material/table_view:", url_path="batch"),
    st.Page("views/model_insights.py", title="Model Insights", icon=":material/insights:", url_path="insights"),
    st.Page("views/about.py", title="About", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🎓 EduPredict")
    try:
        _meta = get_bundle().metadata
        st.caption(f"Model: {_meta['model_name']}  \n{dataset_badge(_meta)}")
    except ModelNotFoundError:
        st.caption("Model not trained yet")
    with st.expander("Support band thresholds"):
        st.slider("Additional support below", 0.0, 95.0, DEFAULT_SUPPORT_THRESHOLD, 1.0, key="support_threshold")
        st.slider("Strong progress from", 5.0, 100.0, DEFAULT_STRONG_THRESHOLD, 1.0, key="strong_threshold")
        if st.session_state["support_threshold"] >= st.session_state["strong_threshold"]:
            st.error("The support threshold must be lower than the strong-progress threshold. Defaults are used.")
        st.caption("Presentation rules only — not validated predictions.")
    st.caption("Estimates, not guarantees. Not for admissions, grading or discipline.")

nav.run()
