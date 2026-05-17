"""Tests for the per-column cleaning operations."""

import numpy as np
import pandas as pd

from cleaner import (
    datetime_fix, dtype_fix, duplicates, encoders, missing, outliers, scalers,
    text_clean,
)


# ==========================================================================
# MISSING VALUES
# ==========================================================================

def test_fill_mean(dirty_df):
    df, _, msg = missing.fill_mean(dirty_df, "age")
    assert df["age"].isna().sum() == 0
    assert "mean" in msg.lower()


def test_fill_median(dirty_df):
    df, _, msg = missing.fill_median(dirty_df, "age")
    assert df["age"].isna().sum() == 0
    assert df["age"].dtype == float


def test_fill_mode(dirty_df):
    df, _, _ = missing.fill_mode(dirty_df, "gender")
    assert df["gender"].isna().sum() == 0


def test_fill_constant_with_custom_value(dirty_df):
    df, _, _ = missing.fill_constant(dirty_df, "name", {"value": "Unknown"})
    assert "Unknown" in df["name"].values


def test_drop_rows_missing(dirty_df):
    n_before = len(dirty_df)
    df, _, _ = missing.drop_rows_missing(dirty_df, "age")
    assert len(df) == n_before - 1


def test_drop_column(dirty_df):
    df, _, _ = missing.drop_column(dirty_df, "const")
    assert "const" not in df.columns


def test_standardize_nan_converts_tokens():
    df = pd.DataFrame({"x": ["A", "?", "N/A", "B", "null"]})
    df2, _, _ = missing.standardize_nan(df, "x")
    assert df2["x"].isna().sum() == 3   # ?, N/A, null become NaN


def test_fill_all_smart_handles_int64(int64_df):
    """Regression test: pandas 3.0 nullable Int64 + float fill used to crash."""
    df, _, msg = missing.fill_all_smart(int64_df)
    assert df["big"].isna().sum() == 0
    assert df["g"].isna().sum() == 0


def test_fill_mean_handles_int64(int64_df):
    """Regression test for the Int64 + float fill error."""
    df, _, _ = missing.fill_mean(int64_df, "big")
    assert df["big"].isna().sum() == 0
    assert df["big"].dtype == float


# ==========================================================================
# OUTLIERS
# ==========================================================================

def test_iqr_remove(dirty_df):
    df, _, _ = outliers.iqr_remove(dirty_df, "age")
    # 200 should have been removed
    assert df["age"].max() < 200


def test_iqr_cap(dirty_df):
    df, _, _ = outliers.iqr_cap(dirty_df, "age")
    # values are capped, not dropped
    assert len(df) == len(dirty_df)
    assert df["age"].max() < 200


def test_iqr_cap_handles_int64(int64_df):
    """Regression test: nullable Int64 used to fail at clip()."""
    df, _, _ = outliers.iqr_cap(int64_df, "big")
    assert df["big"].dtype == float    # cast happened
    assert df["big"].max() < int64_df["big"].max()


def test_zscore_remove():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    df2, _, _ = outliers.zscore_remove(df, "x", {"threshold": 1.5})
    assert df2["x"].max() < 100


def test_isolation_forest_removes_some():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.concatenate([rng.normal(0, 1, 100), [50, -50, 100]]),
        "y": np.concatenate([rng.normal(0, 1, 100), [50, -50, 100]]),
    })
    df2, _, _ = outliers.isolation_forest_remove(df, column=None, params={"contamination": 0.05})
    assert len(df2) < len(df)


def test_dbscan_removes_noise():
    rng = np.random.default_rng(0)
    cluster = rng.normal(0, 1, (50, 2))
    outl = np.array([[20, 20], [-20, 20], [20, -20]])
    df = pd.DataFrame(np.vstack([cluster, outl]), columns=["x", "y"])
    df2, _, _ = outliers.dbscan_remove(df, params={"eps": 0.8, "min_samples": 3})
    assert len(df2) < len(df)


# ==========================================================================
# ENCODERS
# ==========================================================================

def test_one_hot_creates_columns():
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
    df2, _, _ = encoders.one_hot(df, "color")
    assert "color" not in df2.columns
    assert "color_red" in df2.columns
    assert "color_blue" in df2.columns


def test_label_encode_returns_integers():
    df = pd.DataFrame({"x": ["a", "b", "c", "a", "b"]})
    df2, _, _ = encoders.label_encode(df, "x")
    assert df2["x"].dtype.kind in ("i", "u")


def test_frequency_encode_returns_proportions():
    df = pd.DataFrame({"x": ["a"] * 6 + ["b"] * 4})
    df2, _, _ = encoders.frequency_encode(df, "x")
    vals = set(df2["x"].unique())
    assert vals == {0.6, 0.4}


def test_binary_encode_two_values():
    df = pd.DataFrame({"x": ["yes", "no", "yes", "no"]})
    df2, _, _ = encoders.binary_encode(df, "x")
    assert set(df2["x"].unique()) == {0, 1}


def test_binary_encode_refuses_three_values():
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    df2, _, msg = encoders.binary_encode(df, "x")
    assert "binary" in msg.lower() or "2 unique" in msg


def test_ordinal_encode_uses_order():
    df = pd.DataFrame({"size": ["small", "large", "medium", "small"]})
    df2, _, _ = encoders.ordinal_encode(df, "size", {"order": ["small", "medium", "large"]})
    # small=0, medium=1, large=2
    assert df2["size"].tolist() == [0, 2, 1, 0]


# ==========================================================================
# SCALERS
# ==========================================================================

def test_standard_scaler_zero_mean(clean_numeric_df):
    df, _, _ = scalers.standard(clean_numeric_df, "x1")
    assert abs(df["x1"].mean()) < 1e-6


def test_minmax_scaler_in_unit_range(clean_numeric_df):
    df, _, _ = scalers.minmax(clean_numeric_df, "x1")
    assert df["x1"].min() >= -1e-9
    assert df["x1"].max() <= 1 + 1e-9


def test_robust_scaler_runs(clean_numeric_df):
    df, _, _ = scalers.robust(clean_numeric_df, "x1")
    assert df["x1"].dtype == float


def test_log_transform_handles_negative():
    df = pd.DataFrame({"x": [-5, -1, 0, 1, 5, 10]})
    df2, _, _ = scalers.log_transform(df, "x")
    assert df2["x"].isna().sum() == 0   # shift applied


def test_standard_scaler_handles_int64(int64_df):
    """Regression test for the Int64 + scaler crash."""
    df, _, _ = scalers.standard(int64_df, "big")
    assert df["big"].dtype == float


# ==========================================================================
# DUPLICATES + STRUCTURE
# ==========================================================================

def test_drop_duplicate_rows():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
    df2, _, _ = duplicates.drop_duplicate_rows(df)
    assert len(df2) == 3


def test_drop_constant_columns(dirty_df):
    df, _, _ = duplicates.drop_constant_columns(dirty_df)
    assert "const" not in df.columns


def test_drop_high_missing_columns():
    df = pd.DataFrame({
        "good":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "bad":   [1] + [None] * 9,
    })
    df2, _, _ = duplicates.drop_high_missing_columns(df, params={"threshold": 0.5})
    assert "bad" not in df2.columns
    assert "good" in df2.columns


def test_drop_high_correlation():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 50)
    df = pd.DataFrame({"x": base, "y": base + rng.normal(0, 0.001, 50), "z": rng.normal(0, 1, 50)})
    df2, _, _ = duplicates.drop_high_correlation(df, params={"threshold": 0.95})
    assert df2.shape[1] < df.shape[1]


# ==========================================================================
# TEXT CLEANING
# ==========================================================================

def test_strip():
    df = pd.DataFrame({"x": ["  hello ", " world  "]})
    df2, _, _ = text_clean.strip(df, "x")
    assert df2["x"].tolist() == ["hello", "world"]


def test_lowercase():
    df = pd.DataFrame({"x": ["Hello", "WORLD"]})
    df2, _, _ = text_clean.lowercase(df, "x")
    assert df2["x"].tolist() == ["hello", "world"]


def test_remove_punctuation():
    df = pd.DataFrame({"x": ["Hello, World!"]})
    df2, _, _ = text_clean.remove_punctuation(df, "x")
    assert df2["x"].iloc[0] == "Hello World"


def test_remove_stopwords():
    df = pd.DataFrame({"x": ["The quick brown fox", "is the the the"]})
    df2, _, _ = text_clean.remove_stopwords(df, "x")
    assert "the" not in df2["x"].iloc[0].lower()


def test_word_count():
    df = pd.DataFrame({"x": ["hello world", "one two three", None]})
    df2, _, _ = text_clean.word_count(df, "x")
    assert "x_word_count" in df2.columns
    assert df2["x_word_count"].tolist() == [2, 3, 0]


def test_collapse_spaces():
    df = pd.DataFrame({"x": ["hello   world", "a  b"]})
    df2, _, _ = text_clean.collapse_spaces(df, "x")
    assert df2["x"].tolist() == ["hello world", "a b"]


# ==========================================================================
# DATETIME
# ==========================================================================

def test_parse_datetime():
    df = pd.DataFrame({"d": ["2020-01-15", "2021-06-22", "bad-date"]})
    df2, _, _ = datetime_fix.parse(df, "d")
    assert pd.api.types.is_datetime64_any_dtype(df2["d"])
    assert df2["d"].isna().sum() == 1   # 'bad-date' coerced to NaT


def test_extract_parts():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-15", "2021-06-22"])})
    df2, _, _ = datetime_fix.extract_parts(df, "d", {"parts": ["year", "month", "weekday"]})
    assert "d_year" in df2.columns
    assert "d_month" in df2.columns
    assert "d_weekday" in df2.columns
    assert df2["d_year"].tolist() == [2020, 2021]


# ==========================================================================
# DTYPE REPAIR
# ==========================================================================

def test_to_numeric_strips_currency():
    df = pd.DataFrame({"s": ["$1,200", "$2,500", "$3,000"]})
    df2, _, _ = dtype_fix.to_numeric(df, "s")
    assert df2["s"].tolist() == [1200, 2500, 3000]


def test_to_boolean_maps_yes_no():
    df = pd.DataFrame({"x": ["Yes", "no", "YES", "No"]})
    df2, _, _ = dtype_fix.to_boolean(df, "x")
    assert df2["x"].tolist() == [True, False, True, False]
