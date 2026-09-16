"""Generate the SYNTHETIC demo dataset and the input-only sample CSV.

Usage (from the project root):
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --n-students 3000 --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, RANDOM_SEED  # noqa: E402
from src.data import generate_synthetic_data, make_prediction_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EduPredict data.")
    parser.add_argument("--n-students", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", default=str(DATA_DIR / "students_synthetic.csv"))
    parser.add_argument("--sample-out", default=str(DATA_DIR / "sample_students.csv"))
    args = parser.parse_args()

    df = generate_synthetic_data(n_students=args.n_students, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    make_prediction_sample(df, seed=args.seed).to_csv(args.sample_out, index=False)

    print(f"Wrote {len(df):,} SYNTHETIC records ({df['student_id'].nunique():,} students) to {args.out}")
    print(f"Wrote input-only sample to {args.sample_out}")


if __name__ == "__main__":
    main()
