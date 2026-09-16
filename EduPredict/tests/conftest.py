import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import generate_synthetic_data  # noqa: E402
from src.train import train_and_evaluate  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset():
    # Small but realistic dataset keeps the test suite fast and reproducible.
    return generate_synthetic_data(n_students=400, seed=123)


@pytest.fixture(scope="session")
def trained(small_dataset):
    pipeline, metadata = train_and_evaluate(small_dataset, dataset_type="synthetic", seed=123)
    return pipeline, metadata
