import numpy as np
import pandas as pd

from src.config import EXAMPLE_STUDENT, FEATURE_NAMES, TARGET
from src.validation import validate_frame, validate_single


def make_rows(n=3):
    return pd.DataFrame([EXAMPLE_STUDENT] * n)


def test_valid_frame_passes():
    clean, report = validate_frame(make_rows())
    assert report.ok
    assert list(clean[FEATURE_NAMES].dtypes.unique()) == [np.dtype("float64")]


def test_missing_required_columns_reported():
    df = make_rows().drop(columns=["quiz_avg", "attendance_pct"])
    _, report = validate_frame(df)
    assert not report.ok
    assert set(report.missing_columns) == {"quiz_avg", "attendance_pct"}


def test_column_names_are_normalised():
    df = make_rows()
    df.columns = [f"  {c.upper()} " for c in df.columns]
    _, report = validate_frame(df)
    assert report.ok


def test_out_of_range_and_non_numeric_values_flagged():
    df = make_rows(4).astype(object)
    df.loc[0, "attendance_pct"] = 120
    df.loc[1, "participation_score"] = -1
    df.loc[2, "quiz_avg"] = "abc"
    df.loc[3, "missed_assignments"] = 2.5
    _, report = validate_frame(df)
    assert report.invalid_rows == [0, 1, 2, 3]
    problems = report.issues_frame().set_index("row")["problem"]
    assert "outside allowed range" in problems[0]
    assert problems[2] == "not a number"
    assert problems[3] == "must be a whole number"


def test_blank_values_are_allowed_in_features():
    df = make_rows(2)
    df.loc[0, "study_hours_per_week"] = np.nan
    clean, report = validate_frame(df)
    assert report.ok
    assert np.isnan(clean.loc[0, "study_hours_per_week"])


def test_row_with_all_features_blank_is_invalid():
    _, report = validate_single({name: None for name in FEATURE_NAMES})
    assert report.invalid_rows == [0]


def test_target_required_for_training():
    df = make_rows()
    _, report = validate_frame(df, require_target=True)
    assert report.missing_columns == [TARGET]
    df[TARGET] = [50, np.nan, 101]
    _, report = validate_frame(df, require_target=True)
    assert report.invalid_rows == [1, 2]


def test_empty_file_and_row_limit():
    _, report = validate_frame(make_rows().iloc[0:0])
    assert not report.ok
    _, report = validate_frame(make_rows(5), max_rows=2)
    assert "Too many rows" in report.general_errors[0]
