import numpy as np
import pandas as pd
import pytest

from src.config import EXAMPLE_STUDENT, FEATURE_NAMES, GROUP_COLUMN, TARGET
from src.data import generate_synthetic_data
from src.predict import (
    InvalidInputError,
    ModelNotFoundError,
    load_bundle,
    predict_batch,
    predict_one,
    predict_scores,
)
from src.recommendations import study_suggestions, support_band
from src.train import save_artifacts, split_data


def test_synthetic_data_is_reproducible_and_has_missing_values():
    a = generate_synthetic_data(n_students=300, seed=7)
    b = generate_synthetic_data(n_students=300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert len(generate_synthetic_data()) >= 2000
    assert a[FEATURE_NAMES].isna().any().any()
    assert a[TARGET].between(0, 100).all()


def test_split_keeps_students_in_one_partition(small_dataset):
    train, val, test, strategy = split_data(small_dataset, seed=1)
    ids = [set(part[GROUP_COLUMN]) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(train) + len(val) + len(test) == len(small_dataset)
    assert "grouped" in strategy


def test_selected_model_beats_baseline_and_metadata_complete(trained):
    _, meta = trained
    sel, base = meta["test_metrics"]["selected"], meta["test_metrics"]["baseline"]
    assert sel["model"] != "Mean baseline"
    assert sel["mae"] < base["mae"]
    for key in ("feature_names", "trained_at_utc", "dataset_type", "validation_results",
                "permutation_importance", "test_predictions"):
        assert key in meta
    assert meta["feature_names"] == FEATURE_NAMES


def test_pipeline_handles_missing_values(trained):
    pipeline, _ = trained
    row = dict(EXAMPLE_STUDENT)
    row["study_hours_per_week"] = np.nan
    row["participation_score"] = None
    score = predict_one(pipeline, row)
    assert 0 <= score <= 100


def test_prediction_shape_and_bounds(trained, small_dataset):
    pipeline, _ = trained
    preds = predict_scores(pipeline, small_dataset)
    assert preds.shape == (len(small_dataset),)
    assert np.all((preds >= 0) & (preds <= 100))
    extreme = pd.DataFrame([{f: 0 for f in FEATURE_NAMES} | {"missed_assignments": 30},
                            {f: 100 for f in FEATURE_NAMES} | {"missed_assignments": 0,
                                                                  "participation_score": 10,
                                                                  "study_hours_per_week": 60}])
    assert np.all((predict_scores(pipeline, extreme) >= 0) & (predict_scores(pipeline, extreme) <= 100))


def test_single_and_batch_predictions_match(trained, small_dataset):
    pipeline, _ = trained
    batch_input = small_dataset[[GROUP_COLUMN] + FEATURE_NAMES].head(15)
    results, report = predict_batch(pipeline, batch_input)
    assert report.ok and len(results) == 15
    for i, row in batch_input.reset_index(drop=True).iterrows():
        single = predict_one(pipeline, row[FEATURE_NAMES].to_dict())
        assert results.loc[i, "predicted_final_exam_score"] == pytest.approx(round(single, 1))


def test_batch_skips_invalid_rows_and_reports_missing_columns(trained):
    pipeline, _ = trained
    df = pd.DataFrame([EXAMPLE_STUDENT] * 3)
    df.loc[1, "attendance_pct"] = 150
    results, report = predict_batch(pipeline, df)
    assert report.invalid_rows == [1]
    assert list(results["row"]) == [0, 2]
    results, report = predict_batch(pipeline, df.drop(columns=["quiz_avg"]))
    assert results is None and report.missing_columns == ["quiz_avg"]


def test_invalid_single_input_raises(trained):
    pipeline, _ = trained
    with pytest.raises(InvalidInputError):
        predict_one(pipeline, EXAMPLE_STUDENT | {"quiz_avg": 101})


def test_save_and_load_roundtrip(trained, small_dataset, tmp_path):
    pipeline, meta = trained
    save_artifacts(pipeline, meta, tmp_path)
    bundle = load_bundle(tmp_path)
    X = small_dataset[FEATURE_NAMES].head(50)
    np.testing.assert_allclose(predict_scores(bundle.pipeline, X), predict_scores(pipeline, X))
    assert bundle.metadata["model_name"] == meta["model_name"]


def test_missing_artifact_gives_actionable_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        load_bundle(tmp_path)


def test_support_bands_and_suggestions():
    assert support_band(40).key == "support"
    assert support_band(70).key == "progressing"
    assert support_band(90).key == "strong"
    assert support_band(70, 72, 80).key == "support"
    with pytest.raises(ValueError):
        support_band(50, 80, 70)
    tips = study_suggestions(EXAMPLE_STUDENT | {"attendance_pct": 50, "missed_assignments": 6})
    assert any("Attendance" in t for t in tips) and any("missing" in t for t in tips)
    assert study_suggestions({f: 100 for f in FEATURE_NAMES} | {"missed_assignments": 0, "participation_score": 10,
                                                                "study_hours_per_week": 20})
