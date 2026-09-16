"""Schema and value-range validation for training and prediction data.

Missing values (blank cells) are allowed in feature columns because the model
pipeline imputes them. Values that are present must be numeric and in range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_BY_NAME,
    FEATURE_NAMES,
    MAX_BATCH_ROWS,
    TARGET,
    TARGET_MAX,
    TARGET_MIN,
)


@dataclass
class ValidationReport:
    missing_columns: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)  # {"row", "column", "value", "problem"}
    invalid_rows: list[int] = field(default_factory=list)
    general_errors: list[str] = field(default_factory=list)
    n_rows: int = 0

    @property
    def ok(self) -> bool:
        return not (self.missing_columns or self.issues or self.general_errors)

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=["row", "column", "value", "problem"])

    def summary(self) -> str:
        parts = []
        if self.missing_columns:
            parts.append("Missing required columns: " + ", ".join(self.missing_columns))
        parts.extend(self.general_errors)
        if self.issues:
            parts.append(f"{len(self.issues)} invalid value(s) in {len(self.invalid_rows)} row(s)")
        return "; ".join(parts) if parts else "No problems found"


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _check_column(series: pd.Series, name: str, lo: float, hi: float, integer: bool,
                  allow_missing: bool) -> tuple[pd.Series, list[dict]]:
    """Coerce a column to float and collect row-level problems."""
    issues: list[dict] = []
    raw = series
    blank = raw.isna() | raw.astype(str).str.strip().eq("")
    numeric = pd.to_numeric(raw.where(~blank), errors="coerce")

    not_numeric = ~blank & numeric.isna()
    for i in raw.index[not_numeric]:
        issues.append({"row": int(i), "column": name, "value": str(raw[i]), "problem": "not a number"})

    present = numeric.notna()
    finite = present & np.isfinite(numeric.fillna(0))
    for i in numeric.index[present & ~finite]:
        issues.append({"row": int(i), "column": name, "value": str(raw[i]), "problem": "not a finite number"})

    out_of_range = finite & ((numeric < lo) | (numeric > hi))
    for i in numeric.index[out_of_range]:
        issues.append({"row": int(i), "column": name, "value": str(raw[i]),
                       "problem": f"outside allowed range {lo:g}-{hi:g}"})

    if integer:
        non_int = finite & ~out_of_range & (numeric % 1 != 0)
        for i in numeric.index[non_int]:
            issues.append({"row": int(i), "column": name, "value": str(raw[i]), "problem": "must be a whole number"})

    if not allow_missing:
        for i in raw.index[blank]:
            issues.append({"row": int(i), "column": name, "value": "", "problem": "missing value not allowed"})

    cleaned = numeric.where(finite)
    return cleaned.astype(float), issues


def validate_frame(df: pd.DataFrame, require_target: bool = False,
                   max_rows: int | None = MAX_BATCH_ROWS) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a DataFrame against the feature schema.

    Returns ``(clean_df, report)``. ``clean_df`` keeps the original columns but
    feature (and target) columns are converted to floats. Rows listed in
    ``report.invalid_rows`` contain at least one invalid value.
    """
    report = ValidationReport(n_rows=len(df))
    df = normalise_columns(df).reset_index(drop=True)  # row numbers are 0-based positions

    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        report.general_errors.append("Duplicate column names: " + ", ".join(dupes))
        return df, report

    required = FEATURE_NAMES + ([TARGET] if require_target else [])
    report.missing_columns = [c for c in required if c not in df.columns]
    if report.missing_columns:
        return df, report

    if len(df) == 0:
        report.general_errors.append("The file contains no data rows.")
        return df, report
    if max_rows is not None and len(df) > max_rows:
        report.general_errors.append(f"Too many rows ({len(df):,}). The limit is {max_rows:,}.")
        return df, report

    clean = df.copy()
    for name in FEATURE_NAMES:
        spec = FEATURE_BY_NAME[name]
        clean[name], issues = _check_column(df[name], name, spec.min_value, spec.max_value,
                                            spec.integer, allow_missing=True)
        report.issues.extend(issues)

    if require_target:
        clean[TARGET], issues = _check_column(df[TARGET], TARGET, TARGET_MIN, TARGET_MAX,
                                              integer=False, allow_missing=False)
        report.issues.extend(issues)

    all_missing = clean[FEATURE_NAMES].isna().all(axis=1) & ~clean.index.isin(
        [i["row"] for i in report.issues])
    for i in clean.index[all_missing]:
        report.issues.append({"row": int(i), "column": "(all features)", "value": "",
                              "problem": "every input is blank"})

    report.invalid_rows = sorted({i["row"] for i in report.issues})
    return clean, report


def validate_single(values: dict) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate one student's inputs given as a dict."""
    return validate_frame(pd.DataFrame([values]), require_target=False, max_rows=1)
