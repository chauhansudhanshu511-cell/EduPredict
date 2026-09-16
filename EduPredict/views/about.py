"""About page: workflow, schema, algorithms, limitations and future work."""
from __future__ import annotations

import streamlit as st

from src.config import PREDICTION_TIME
from src.data import schema_table


def render():
    st.title("About EduPredict")
    st.markdown(
        f"""
**EduPredict** is a portfolio project that shows an honest, end-to-end machine-learning workflow for
estimating a student's final exam score and turning it into supportive guidance.

#### Prediction time and leakage prevention
{PREDICTION_TIME}

Leakage is also prevented by fitting imputation and scaling **inside each pipeline on training data only**,
splitting **by student** so repeat records never cross train/validation/test, selecting the model on the
**validation** set, and touching the **test** set only once.

#### Workflow
1. `scripts/generate_demo_data.py` creates a reproducible synthetic dataset (seed 42) with noise,
   missing values and repeat students.
2. `src/validation.py` checks required columns, numeric types and allowed ranges.
3. `src/train.py` splits the data 70 / 15 / 15, fits a mean baseline, Ridge regression and random
   forest candidates, selects the best by validation RMSE, evaluates once on test, computes permutation
   importance on validation, and saves `models/edupredict_model.joblib` + `models/metadata.json`.
4. `app.py` loads the saved artifacts once (cached) and serves predictions. It never retrains on uploads
   and never loads user-supplied model files.
        """
    )
    st.markdown("#### Dataset schema")
    st.dataframe(schema_table(), hide_index=True, width="stretch")

    st.markdown(
        """
#### Algorithms
- **Mean baseline** – always predicts the average training score. Any useful model must beat it.
- **Ridge regression** – linear model with L2 regularisation; fast, stable and easy to interpret.
- **Random forest** – averages many decision trees; captures non-linear effects and interactions.

#### Limitations
Trained on synthetic data by default; no fairness audit; missing important context about students;
support bands and suggestions are hand-written rules, not validated interventions.

#### Responsible use and privacy
Estimates only – never for admissions, grading or discipline. Uploaded CSVs are processed in memory and not
stored by the app. Do not upload personally identifying information you are not authorised to process.

#### Future improvements
Prediction intervals (e.g. quantile regression or conformal prediction) · subgroup error and fairness
analysis on real, consented data · probability calibration for band membership · model cards and data
drift monitoring · per-student explanations (e.g. SHAP) clearly separated from suggestions ·
teacher feedback loop to evaluate whether suggestions help.
        """
    )


render()
