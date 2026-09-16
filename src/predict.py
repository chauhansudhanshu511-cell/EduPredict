"""Load trusted model artifacts and make predictions.

Only artifacts produced by ``src/train.py`` in this project's ``models/``
folder are loaded. The app never accepts serialized models from users,
because unpickling untrusted files can execute arbitrary code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_NAMES,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    TARGET_MAX,
    TARGET_MIN,
)
from src.validation import validate_frame, validate_single

SETUP_COMMANDS = (
    "python scripts/generate_demo_data.py\n"
    "python -m src.train --data data/students_synthetic.csv --dataset-type synthetic"
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact is missing."""


class InvalidInputError(ValueError):
    """Raised when prediction inputs fail validation."""


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict

    @property
    def dataset_type(self) -> str:
        return self.metadata.get("dataset_type", "unknown")


def load_bundle(model_dir: str | Path = MODELS_DIR) -> ModelBundle:
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    meta_path = model_dir / METADATA_FILENAME
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotFoundError(
            f"No trained model found in '{model_dir}'. Expected '{MODEL_FILENAME}' and "
            f"'{METADATA_FILENAME}'. From the project root, run:\n{SETUP_COMMANDS}"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ModelNotFoundError(
            "The saved model was trained with a different feature schema. "
            f"Retrain it with:\n{SETUP_COMMANDS}"
        )
    pipeline = joblib.load(model_path)
    return ModelBundle(pipeline=pipeline, metadata=metadata)


def predict_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict scores for already-validated rows, clipped to the 0-100 scale."""
    X = features[FEATURE_NAMES].astype(float)
    return np.clip(pipeline.predict(X), TARGET_MIN, TARGET_MAX)


def predict_one(pipeline, values: dict) -> float:
    clean, report = validate_single(values)
    if not report.ok:
        raise InvalidInputError(report.summary())
    return float(predict_scores(pipeline, clean)[0])


def predict_batch(pipeline, df: pd.DataFrame):
    """Validate and predict a batch.

    Returns ``(results, report)``. ``results`` holds the valid rows with a
    ``predicted_final_exam_score`` column, or ``None`` if the file cannot be
    used at all (e.g. missing columns). Invalid rows are skipped and listed in
    ``report``.
    """
    clean, report = validate_frame(df, require_target=False)
    if report.missing_columns or report.general_errors:
        return None, report
    valid = clean.drop(index=report.invalid_rows)
    results = valid.copy()
    if len(valid):
        results["predicted_final_exam_score"] = predict_scores(pipeline, valid).round(1)
    else:
        results["predicted_final_exam_score"] = pd.Series(dtype=float)
    results.insert(0, "row", results.index.astype(int))
    return results.reset_index(drop=True), report
