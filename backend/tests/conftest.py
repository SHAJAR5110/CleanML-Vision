"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `backend/` importable so tests can `from cleaner import ...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def dirty_df():
    """A small dataset deliberately containing every cleaning case we handle."""
    return pd.DataFrame({
        "id":      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name":    ["Alice", "Bob", "Charlie", "Dan", None, "Frank", "Gina", "Hank", "Ivy", "Jen"],
        "gender":  ["Male", "M", "male", "MALE", "Female", "F", "female", "FEMALE", "Male", None],
        "age":     [25, 30, np.nan, 22, 45, 200, 28, 31, 27, 26],          # outlier 200, missing 1
        "salary":  ["$1,200", "$2,500", "?", "$3,000", "$4,500", "$8,000", "$2,800", "$3,200", "$2,700", "$2,600"],
        "joined":  ["2020-01-15", "2019-06-22", "2021-03-10", "2020-11-01",
                    "2022-02-14", "2018-08-30", "2020-05-19", "2021-12-25",
                    "2019-09-12", "2022-07-04"],
        "const":   ["x"] * 10,                                              # constant
        "category":["A", "B", "A", "C", "B", "A", "B", "A", "C", "B"],
    })


@pytest.fixture
def clean_numeric_df():
    """Numeric-only DataFrame for tests that need pure numeric data."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, 100),
        "x2": rng.normal(10, 5, 100),
        "x3": rng.normal(-5, 2, 100),
        "y":  rng.integers(0, 2, 100),
    })


@pytest.fixture
def int64_df():
    """DataFrame whose numeric column uses pandas 3.0 nullable Int64 dtype."""
    return pd.DataFrame({
        "big": pd.array([100, 200, 150, 175, 220, 343361258, None, 500000000], dtype="Int64"),
        "g":   ["a", "a", "b", "a", "b", "a", "b", "a"],
    })


@pytest.fixture
def imbalanced_df():
    """For class-balancing tests: 100 minority + 10 majority."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "f1": rng.normal(0, 1, 110),
        "f2": rng.normal(0, 1, 110),
        "target": [0] * 100 + [1] * 10,
    })


@pytest.fixture
def flask_client():
    """Flask test client — exercises HTTP endpoints without spawning a server."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
