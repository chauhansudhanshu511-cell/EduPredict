"""Model insights page: model comparison, error analysis, feature importance, limitations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.ui import require_bundle, synthetic_warning


def render():
    bundle = require_bundle()
    meta = bundle.metadata
    st.title("Model insights")
    synthetic_warning(meta)

    st.subheader("1 · Model selection on the validation set")
    st.caption(f"All candidates were fitted on {meta['split']['train_rows']} training records and compared on "
               f"{meta['split']['validation_rows']} validation records. Split: {meta['split']['strategy']}. "
               f"Selection rule: {meta['selection_metric']}.")
    val = pd.DataFrame(meta["validation_results"])
    val["selected"] = val["model"].eq(meta["model_name"]).map({True: "✅", False: ""})
    st.dataframe(val, hide_index=True, width="stretch")

    st.subheader("2 · Final evaluation on the untouched test set")
    test = pd.DataFrame(meta["test_metrics"].values())
    c1, c2 = st.columns([1, 1])
    c1.dataframe(test, hide_index=True, width="stretch")
    c1.caption("MAE: average absolute error in points. RMSE: penalises large errors more. "
               "R²: share of score variation explained (0 = no better than predicting the mean).")
    c2.plotly_chart(charts.metric_comparison(test.to_dict("records"), "rmse", "Test RMSE (lower is better)"),
                    width="stretch")

    tp = meta["test_predictions"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.actual_vs_predicted(tp["actual"], tp["predicted"]), width="stretch")
    c2.plotly_chart(charts.residuals(tp["actual"], tp["predicted"]), width="stretch")
    st.plotly_chart(charts.residual_histogram(tp["actual"], tp["predicted"]), width="stretch")
    st.caption("Predictions are clipped to 0–100, so errors near the ends of the scale can look compressed.")

    st.subheader("3 · Permutation feature importance")
    st.plotly_chart(charts.feature_importance(meta["permutation_importance"]["values"]), width="stretch")
    st.info("Importance shows how much the **model's** validation error grows when a feature's values are "
            "shuffled. It describes model behaviour, **not causation**: it does not prove that changing a "
            "feature would change a student's result. Correlated features can share or hide importance.")

    st.subheader("4 · Limitations and possible bias")
    st.markdown(
        "- **Synthetic data:** relationships were invented by the data generator, so good metrics here are "
        "expected and do not transfer to real classrooms.\n"
        "- **Missing context:** health, family responsibilities, work hours, disability accommodations, "
        "language background and course difficulty are not included.\n"
        "- **Measurement bias:** participation scores are subjective and study hours are self-reported.\n"
        "- **Historical bias:** a model trained on real records can reproduce past inequalities, e.g. "
        "lower attendance caused by transport or caring duties.\n"
        "- **Subgroup performance is unknown:** errors were not checked across demographic groups, because "
        "the dataset has no such fields. A real deployment would need a fairness audit.\n"
        "- **Distribution shift:** a new course, grading scheme or institution can make the model unreliable."
    )


render()
