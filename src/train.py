"""Train, select and evaluate EduPredict models.

Workflow
1. Validate the CSV against the schema (strict: invalid values stop training).
2. Split 70/15/15 into train/validation/test. If ``student_id`` exists, split by
   student so no student appears in more than one split.
3. Fit every candidate (mean baseline, Ridge, Random Forest) on TRAIN only.
   Imputation and scaling live inside each pipeline, so they are fitted on
   training data only.
4. Pick the candidate with the lowest validation RMSE.
5. Evaluate the selected model and the baseline ONCE on the untouched test set.
6. Save the pipeline (joblib) and metadata (JSON).

Usage:
    python -m src.train --data data/students_synthetic.csv --dataset-type synthetic
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_NAMES,
    GROUP_COLUMN,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    PREDICTION_TIME,
    RANDOM_SEED,
    TARGET,
    TARGET_MAX,
    TARGET_MIN,
    FEATURE_BY_NAME,
)
from src.data import load_csv
from src.validation import validate_frame


def build_candidates(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Candidate pipelines. Each includes its own preprocessing."""
    candidates: dict[str, Pipeline] = {
        "Mean baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]),
    }
    for alpha in (0.1, 1.0, 10.0):
        candidates[f"Ridge (alpha={alpha:g})"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
    for depth in (8, 14):
        candidates[f"Random forest (max_depth={depth})"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(n_estimators=150, max_depth=depth, min_samples_leaf=5,
                                            random_state=seed, n_jobs=1)),
        ])
    return candidates


def metrics(y_true, y_pred) -> dict[str, float]:
    y_pred = np.clip(y_pred, TARGET_MIN, TARGET_MAX)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED,
               val_size: float = 0.15, test_size: float = 0.15):
    """Reproducible train/validation/test split, grouped by student when possible."""
    idx = np.arange(len(df))
    if GROUP_COLUMN in df.columns and df[GROUP_COLUMN].notna().all():
        groups = df[GROUP_COLUMN].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss.split(idx, groups=groups))
        rel_val = val_size / (1 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr, va = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))
        train_idx, val_idx = trainval_idx[tr], trainval_idx[va]
        strategy = f"grouped by {GROUP_COLUMN}"
    else:
        trainval_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
        train_idx, val_idx = train_test_split(trainval_idx, test_size=val_size / (1 - test_size),
                                              random_state=seed)
        strategy = "random rows (no student_id column)"
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
            strategy)


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean, report = validate_frame(df, require_target=True, max_rows=None)
    if not report.ok:
        details = report.issues_frame().head(10).to_string(index=False) if report.issues else ""
        raise ValueError(f"Training data failed validation: {report.summary()}\n{details}")
    return clean


def train_and_evaluate(df: pd.DataFrame, dataset_type: str = "synthetic",
                       seed: int = RANDOM_SEED, dataset_name: str = "in-memory"):
    """Run the full workflow. Returns ``(selected_pipeline, metadata)``."""
    if dataset_type not in {"synthetic", "real"}:
        raise ValueError("dataset_type must be 'synthetic' or 'real'")
    df = prepare_training_frame(df)
    train, val, test, strategy = split_data(df, seed=seed)
    X_tr, y_tr = train[FEATURE_NAMES], train[TARGET]
    X_va, y_va = val[FEATURE_NAMES], val[TARGET]
    X_te, y_te = test[FEATURE_NAMES], test[TARGET]

    candidates = build_candidates(seed)
    validation_results = []
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        validation_results.append({"model": name, **metrics(y_va, pipe.predict(X_va))})

    trained = [r for r in validation_results if r["model"] != "Mean baseline"]
    best_name = min(trained, key=lambda r: r["rmse"])["model"]
    best = fitted[best_name]
    baseline = fitted["Mean baseline"]

    # Single final evaluation on the untouched test set.
    test_pred = np.clip(best.predict(X_te), TARGET_MIN, TARGET_MAX)
    test_metrics = {
        "selected": {"model": best_name, **metrics(y_te, test_pred)},
        "baseline": {"model": "Mean baseline", **metrics(y_te, baseline.predict(X_te))},
    }

    perm = permutation_importance(best, X_va, y_va, scoring="neg_mean_absolute_error",
                                  n_repeats=10, random_state=seed, n_jobs=1)
    importance = sorted(
        [{"feature": f, "label": FEATURE_BY_NAME[f].label,
          "importance_mean": round(float(m), 4), "importance_std": round(float(s), 4)}
         for f, m, s in zip(FEATURE_NAMES, perm.importances_mean, perm.importances_std)],
        key=lambda r: r["importance_mean"], reverse=True,
    )

    counts, edges = np.histogram(df[TARGET], bins=20, range=(0, 100))
    metadata = {
        "project": "EduPredict",
        "model_name": best_name,
        "dataset_type": dataset_type,
        "dataset_name": dataset_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "prediction_time": PREDICTION_TIME,
        "feature_names": FEATURE_NAMES,
        "feature_schema": {f: {"min": FEATURE_BY_NAME[f].min_value, "max": FEATURE_BY_NAME[f].max_value,
                               "integer": FEATURE_BY_NAME[f].integer} for f in FEATURE_NAMES},
        "target": TARGET,
        "split": {"strategy": strategy, "train_rows": len(train), "validation_rows": len(val),
                  "test_rows": len(test), "total_rows": len(df),
                  "unique_students": int(df[GROUP_COLUMN].nunique()) if GROUP_COLUMN in df else None},
        "missing_value_share": {f: round(float(df[f].isna().mean()), 4) for f in FEATURE_NAMES},
        "selection_metric": "validation RMSE (lower is better)",
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "permutation_importance": {"data": "validation", "scoring": "increase in MAE", "values": importance},
        "test_predictions": {"actual": [round(float(v), 2) for v in y_te],
                             "predicted": [round(float(v), 2) for v in test_pred]},
        "target_histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    return best, metadata


def save_artifacts(pipeline, metadata: dict, output_dir: str | Path = MODELS_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / MODEL_FILENAME, compress=3)
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the EduPredict model.")
    parser.add_argument("--data", required=True, help="CSV file with the documented schema")
    parser.add_argument("--dataset-type", choices=["synthetic", "real"], required=True,
                        help="Label shown in the app. Use 'real' only for genuine student records.")
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    df = load_csv(args.data)
    pipeline, metadata = train_and_evaluate(df, dataset_type=args.dataset_type, seed=args.seed,
                                            dataset_name=Path(args.data).name)
    out = save_artifacts(pipeline, metadata, args.output_dir)

    print(f"Split: {metadata['split']}")
    print("\nValidation results (used for model selection):")
    print(pd.DataFrame(metadata["validation_results"]).to_string(index=False))
    print(f"\nSelected model: {metadata['model_name']}")
    print("\nTest-set results (single final evaluation):")
    print(pd.DataFrame(metadata["test_metrics"].values()).to_string(index=False))
    print(f"\nSaved artifacts to {out.resolve()}")
    if args.dataset_type == "synthetic":
        print("Note: metrics on synthetic data do not establish real-world accuracy.")


if __name__ == "__main__":
    main()
