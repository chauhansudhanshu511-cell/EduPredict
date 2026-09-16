"""Smoke tests that run the Streamlit pages headlessly with streamlit.testing."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import METADATA_FILENAME, MODEL_FILENAME, MODELS_DIR

APP = str(Path(__file__).resolve().parents[1] / "app.py")
needs_model = pytest.mark.skipif(
    not (MODELS_DIR / MODEL_FILENAME).exists() or not (MODELS_DIR / METADATA_FILENAME).exists(),
    reason="Model artifact not trained yet",
)


PAGE_TITLES = {
    "app.py": None,
    "views/overview.py": None,
    "views/individual_prediction.py": "Individual prediction",
    "views/batch_predictions.py": "Batch predictions",
    "views/model_insights.py": "Model insights",
    "views/about.py": "About EduPredict",
}


@needs_model
@pytest.mark.parametrize("page", list(PAGE_TITLES))
def test_pages_render_without_exceptions(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page != "app.py":
        at.switch_page(page)
        at.run()
    assert not at.exception
    if PAGE_TITLES[page]:
        assert [t.value for t in at.title] == [PAGE_TITLES[page]]
    else:
        assert any("EduPredict" in m.value for m in at.markdown)


@needs_model
def test_example_student_prediction_flow():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page("views/individual_prediction.py")
    at.run()
    next(b for b in at.button if b.label == "Load example student").click().run()
    next(b for b in at.button if "Predict" in b.label).click().run()
    assert not at.exception
    assert any("Study suggestions" in s.value for s in at.subheader)
