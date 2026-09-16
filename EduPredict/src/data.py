"""Synthetic data generation and CSV loading.

The synthetic generator exists so the project runs without credentials or
private student records. Its relationships are invented for demonstration;
results on this data say nothing about real-world accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_BY_NAME,
    FEATURE_NAMES,
    GROUP_COLUMN,
    RANDOM_SEED,
    TARGET,
)

# Share of values set to missing per feature (self-reported fields are missing more often).
MISSING_RATES = {
    "attendance_pct": 0.02,
    "study_hours_per_week": 0.08,
    "previous_exam_score": 0.03,
    "assignment_avg": 0.03,
    "quiz_avg": 0.04,
    "missed_assignments": 0.02,
    "participation_score": 0.06,
}


def generate_synthetic_data(
    n_students: int = 2000,
    repeat_share: float = 0.25,
    seed: int = RANDOM_SEED,
    add_missing: bool = True,
) -> pd.DataFrame:
    """Create a reproducible synthetic dataset.

    Each student has hidden traits (ability, engagement). A share of students
    appear twice (e.g. two course terms), which is why training splits by
    ``student_id``.
    """
    rng = np.random.default_rng(seed)

    ability = rng.normal(0, 1, n_students)
    engagement = rng.normal(0, 1, n_students)
    student_ids = np.array([f"S{i:05d}" for i in range(1, n_students + 1)])

    # Students with a second record keep their traits, with some drift.
    n_repeat = int(n_students * repeat_share)
    repeat_idx = rng.choice(n_students, size=n_repeat, replace=False)
    idx = np.concatenate([np.arange(n_students), repeat_idx])
    term = np.concatenate([np.ones(n_students, dtype=int), np.full(n_repeat, 2)])
    ab = ability[idx] + np.where(term == 2, rng.normal(0.1, 0.3, len(idx)), 0)
    en = engagement[idx] + np.where(term == 2, rng.normal(0, 0.4, len(idx)), 0)
    n = len(idx)

    attendance = np.clip(82 + 9 * en + rng.normal(0, 7, n), 20, 100)
    study_hours = np.clip(np.exp(1.8 + 0.35 * en + 0.1 * ab + rng.normal(0, 0.45, n)), 0, 40)
    previous_exam = np.clip(66 + 12 * ab + 3 * en + rng.normal(0, 8, n), 0, 100)
    assignment_avg = np.clip(72 + 7 * ab + 7 * en + rng.normal(0, 8, n), 0, 100)
    quiz_avg = np.clip(68 + 10 * ab + 4 * en + rng.normal(0, 9, n), 0, 100)
    missed = np.clip(rng.poisson(np.exp(0.5 - 0.7 * en)), 0, 30)
    participation = np.clip(5.5 + 1.6 * en + rng.normal(0, 1.5, n), 0, 10)

    # Invented outcome with diminishing returns for study time, an attendance
    # penalty below 70% and student-level noise.
    final = (
        6
        + 0.38 * previous_exam
        + 0.17 * assignment_avg
        + 0.17 * quiz_avg
        + 0.10 * attendance
        + 6.0 * np.log1p(study_hours)
        - 1.3 * missed
        + 0.6 * participation
        - 0.25 * np.clip(70 - attendance, 0, None)
        + rng.normal(0, 6.5, n)
    )
    final = np.clip(final, 0, 100)

    df = pd.DataFrame(
        {
            GROUP_COLUMN: student_ids[idx],
            "term": term,
            "attendance_pct": attendance.round(1),
            "study_hours_per_week": study_hours.round(1),
            "previous_exam_score": previous_exam.round(1),
            "assignment_avg": assignment_avg.round(1),
            "quiz_avg": quiz_avg.round(1),
            "missed_assignments": missed.astype(float),
            "participation_score": participation.round(1),
            TARGET: final.round(1),
        }
    )

    if add_missing:
        for col, rate in MISSING_RATES.items():
            mask = rng.random(n) < rate
            df.loc[mask, col] = np.nan

    df["missed_assignments"] = df["missed_assignments"].astype("Int64")  # whole numbers, blanks allowed
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def make_prediction_sample(df: pd.DataFrame, n_rows: int = 25, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build an input-only sample CSV (no target) for batch prediction demos."""
    sample = df.sample(n=n_rows, random_state=seed)[FEATURE_NAMES].copy()
    sample.insert(0, GROUP_COLUMN, [f"DEMO{i:03d}" for i in range(1, n_rows + 1)])
    return sample.reset_index(drop=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and normalise column names (trim spaces, lower-case)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def schema_table() -> pd.DataFrame:
    """Human-readable schema used in docs and the About page."""
    rows = []
    for name in FEATURE_NAMES:
        spec = FEATURE_BY_NAME[name]
        rows.append(
            {
                "column": name,
                "type": "integer" if spec.integer else "number",
                "allowed range": f"{spec.min_value:g} - {spec.max_value:g}",
                "missing allowed": "yes",
                "description": spec.help,
            }
        )
    rows.append({"column": TARGET, "type": "number", "allowed range": "0 - 100",
                 "missing allowed": "no (training only)",
                 "description": "Final exam score. Training target only; never an input."})
    rows.append({"column": GROUP_COLUMN, "type": "text", "allowed range": "any",
                 "missing allowed": "optional column",
                 "description": "Student identifier used to keep repeat records in one split."})
    return pd.DataFrame(rows)
