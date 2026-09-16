"""Central configuration: feature schema, value ranges, paths and presentation rules.

Keeping these in one place means the data generator, validation, training,
prediction and the web app all agree on the same schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILENAME = "edupredict_model.joblib"
METADATA_FILENAME = "metadata.json"

RANDOM_SEED = 42

# Prediction time: every feature must be known at this point in the term.
PREDICTION_TIME = (
    "Week 10 of a 14-week term, about four weeks before the final exam. "
    "Only information recorded up to that point is used. The final exam score, "
    "final course grade and anything collected after week 10 are never used as inputs."
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    min_value: float
    max_value: float
    integer: bool
    help: str
    example: float


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("attendance_pct", "Attendance (%)", 0, 100, False,
                "Share of classes attended from week 1 to week 10.", 86.0),
    FeatureSpec("study_hours_per_week", "Average weekly study hours", 0, 60, False,
                "Self-reported independent study hours per week, averaged to week 10.", 7.5),
    FeatureSpec("previous_exam_score", "Previous exam score", 0, 100, False,
                "Most recent exam taken before the prediction point (e.g. the midterm).", 68.0),
    FeatureSpec("assignment_avg", "Assignment average", 0, 100, False,
                "Mean score of assignments graded by week 10.", 74.0),
    FeatureSpec("quiz_avg", "Quiz average", 0, 100, False,
                "Mean score of quizzes graded by week 10.", 70.0),
    FeatureSpec("missed_assignments", "Missed assignments", 0, 30, True,
                "Number of assignments not submitted by week 10.", 2),
    FeatureSpec("participation_score", "Participation score (0-10)", 0, 10, False,
                "Instructor-rated class participation on a 0-10 scale.", 6.0),
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURE_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}

TARGET = "final_exam_score"
GROUP_COLUMN = "student_id"  # optional; used to keep a student's records in one split
TARGET_MIN, TARGET_MAX = 0.0, 100.0

MAX_BATCH_ROWS = 10_000

# Support bands are PRESENTATION RULES applied to the predicted score.
# They are not separately validated predictions.
DEFAULT_SUPPORT_THRESHOLD = 60.0   # below this -> additional support suggested
DEFAULT_STRONG_THRESHOLD = 75.0    # at or above this -> strong progress

EXAMPLE_STUDENT: dict[str, float] = {f.name: f.example for f in FEATURES}
