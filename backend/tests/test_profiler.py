"""Tests for cleaner.profiler — type inference, missing detection, quality score."""

import pandas as pd
import pytest

from cleaner.profiler import (
    NAN_STRINGS, _count_embedded_nans, _is_datetime_like, _is_numeric_string,
    profile_dataframe,
)


# ---------- type inference ----------

def test_numeric_column_detected_as_numeric(dirty_df):
    p = profile_dataframe(dirty_df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["age"]["inferred_type"] == "numeric"


def test_currency_strings_detected_as_numeric_string():
    s = pd.Series(["$1,200", "$2,500", "$3,000", "$4,500"])
    assert bool(_is_numeric_string(s))


def test_datetime_strings_detected_as_datetime():
    s = pd.Series(["2020-01-15", "2019-06-22", "2021-03-10"])
    assert bool(_is_datetime_like(s))


def test_constant_column_detected_as_constant(dirty_df):
    p = profile_dataframe(dirty_df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["const"]["inferred_type"] == "constant"
    assert "constant_column" in cols["const"]["warnings"]


def test_categorical_with_low_cardinality(dirty_df):
    p = profile_dataframe(dirty_df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["category"]["inferred_type"] == "categorical"
    assert cols["category"]["cardinality"] == "low_card"


def test_id_like_column_detected():
    df = pd.DataFrame({"customer_id": range(100), "value": range(100)})
    p = profile_dataframe(df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["customer_id"]["inferred_type"] == "id_like"
    assert "id_like_column" in cols["customer_id"]["warnings"]


# ---------- missing value detection ----------

def test_real_nan_counted_as_missing(dirty_df):
    p = profile_dataframe(dirty_df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["age"]["missing"] == 1
    assert cols["name"]["missing"] == 1


def test_embedded_nan_tokens_counted():
    s = pd.Series(["A", "?", "N/A", "B", "null", "-", "C", "missing"])
    assert _count_embedded_nans(s) >= 5


def test_nan_strings_includes_common_tokens():
    expected = {"?", "n/a", "null", "missing", "unknown", "na", "nan", ""}
    assert expected.issubset(NAN_STRINGS)


# ---------- outliers + skewness ----------

def test_outliers_detected(dirty_df):
    p = profile_dataframe(dirty_df)
    cols = {c["name"]: c for c in p["columns"]}
    # 200 is an obvious outlier vs ~25-45 cluster
    assert cols["age"]["outlier_count"] >= 1


def test_skewness_flagged_when_extreme():
    df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]})
    p = profile_dataframe(df)
    cols = {c["name"]: c for c in p["columns"]}
    assert "highly_skewed" in cols["x"]["warnings"]


# ---------- quality score ----------

def test_quality_score_in_range(dirty_df):
    p = profile_dataframe(dirty_df)
    assert 0 <= p["quality_score"] <= 100
    assert p["grade"] in ("A", "B", "C", "D", "F")


def test_clean_data_high_score():
    df = pd.DataFrame({"x": range(100), "y": range(100, 200)})
    p = profile_dataframe(df)
    assert p["quality_score"] >= 80
    assert p["grade"] in ("A", "B")


def test_terrible_data_low_score():
    df = pd.DataFrame({
        "a": [None] * 90 + [1] * 10,
        "b": ["x"] * 100,
        "c": list(range(50)) + [None] * 50,
    })
    p = profile_dataframe(df)
    assert p["quality_score"] < 60
