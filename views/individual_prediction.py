"""Individual prediction page: form, estimate, support band and study suggestions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.config import EXAMPLE_STUDENT, FEATURES
from src.predict import predict_one
from src.recommendations import study_suggestions, support_band
from src.ui import dataset_badge, require_bundle, responsible_use_note, thresholds
from src.validation import validate_single


def _init_form_state():
    for spec in FEATURES:
        st.session_state.setdefault(f"in_{spec.name}", None)


def _load_example():
    for name, value in EXAMPLE_STUDENT.items():
        st.session_state[f"in_{name}"] = int(value) if name == "missed_assignments" else float(value)
    st.session_state.pop("last_prediction", None)


def _clear_form():
    for spec in FEATURES:
        st.session_state[f"in_{spec.name}"] = None
    st.session_state.pop("last_prediction", None)


def render():
    bundle = require_bundle()
    st.title("Individual prediction")
    st.caption(f"{dataset_badge(bundle.metadata)} · Enter information recorded up to week 10. "
               "Blank fields are allowed and filled with typical training values.")
    _init_form_state()

    b1, b2, _ = st.columns([1.4, 1, 3])
    b1.button("Load example student", on_click=_load_example, icon=":material/person:")
    b2.button("Clear form", on_click=_clear_form, icon=":material/refresh:")

    with st.form("student_form", border=True):
        for start in range(0, len(FEATURES), 2):  # row by row, so mobile order stays logical
            cols = st.columns(2)
            for col, spec in zip(cols, FEATURES[start:start + 2]):
                with col:
                    if spec.integer:
                        st.number_input(spec.label, min_value=int(spec.min_value), max_value=int(spec.max_value),
                                        step=1, key=f"in_{spec.name}", help=spec.help,
                                        placeholder="Leave blank if unknown")
                    else:
                        st.number_input(spec.label, min_value=float(spec.min_value),
                                        max_value=float(spec.max_value), step=0.5, key=f"in_{spec.name}",
                                        help=spec.help, placeholder="Leave blank if unknown")
        submitted = st.form_submit_button("Predict final exam score", type="primary", width="stretch")

    if submitted:
        values = {spec.name: st.session_state[f"in_{spec.name}"] for spec in FEATURES}
        values = {k: (float("nan") if v is None else v) for k, v in values.items()}
        _, report = validate_single(values)
        if all(pd.isna(v) for v in values.values()):
            st.error("Please enter at least one value, or click **Load example student**.")
            return
        if not report.ok:
            st.error("Some inputs are invalid:")
            st.dataframe(report.issues_frame(), hide_index=True)
            return
        st.session_state["last_prediction"] = (values, predict_one(bundle.pipeline, values))

    if "last_prediction" not in st.session_state:
        st.markdown('<p class="small-note">Fill in the form or load the example student, then press '
                    '<b>Predict</b>.</p>', unsafe_allow_html=True)
        return

    values, score = st.session_state["last_prediction"]
    lo, hi = thresholds()
    band = support_band(score, lo, hi)
    mae = bundle.metadata["test_metrics"]["selected"]["mae"]

    st.divider()
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Estimated final exam score")
        st.plotly_chart(charts.score_gauge(score, lo, hi, band.color), width="stretch")
        st.caption(f"On the held-out test set the model was off by about **{mae:.1f} points** on average "
                   f"(MAE). Treat this as a rough estimate, not an exact score.")
    with right:
        st.subheader("Support band")
        st.markdown(f'<div class="band" style="border-color:{band.color}"><h3 style="color:{band.color}">'
                    f'{band.label}</h3>{band.message}</div>', unsafe_allow_html=True)
        st.caption(f"Bands are presentation rules applied to the estimate (below {lo:g}; {lo:g}–{hi:g}; "
                   f"{hi:g} and above). They are **not** separately validated predictions. "
                   "Adjust the thresholds in the sidebar.")
        st.subheader("Study suggestions")
        st.caption("Rule-based tips triggered by the inputs you entered. They are **not** model "
                   "explanations and do not show why the model produced this score.")
        for tip in study_suggestions(values):
            st.markdown(f"- {tip}")
    responsible_use_note()


render()
