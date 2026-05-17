"""Tests for advanced modules: label_norm, validate, feature_eng, balance, reduce, merge."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cleaner import balance, feature_eng, label_norm, merge, reduce, splitter, validate


# ==========================================================================
# LABEL NORMALIZER
# ==========================================================================

def test_label_norm_detects_capitalization_groups():
    s = pd.Series(["Male", "male", "MALE", "Female", "female"])
    groups = label_norm.detect_groups(s, threshold=0.85)
    canonicals = {g["canonical"] for g in groups}
    # Should find at least Male/male/MALE as one group
    male_members = next((g["members"] for g in groups if any("male" in m["value"].lower() and len(m["value"]) > 1 for m in g["members"])), [])
    male_values = {m["value"] for m in male_members}
    assert {"Male", "male", "MALE"}.issubset(male_values)


def test_label_norm_apply_mapping_changes_values():
    df = pd.DataFrame({"g": ["Male", "male", "MALE", "Female"]})
    mapping = {"male": "Male", "MALE": "Male"}
    df2, _, msg = label_norm.apply_mapping(df, "g", {"mapping": mapping})
    assert df2["g"].value_counts().to_dict() == {"Male": 3, "Female": 1}


def test_label_norm_requires_mapping():
    df = pd.DataFrame({"g": ["A", "B"]})
    _, _, msg = label_norm.apply_mapping(df, "g", {})
    assert "mapping" in msg.lower()


# ==========================================================================
# VALIDATION
# ==========================================================================

def test_validate_less_than_flags_violations():
    df = pd.DataFrame({"start": [1, 5, 10, 20], "end": [3, 4, 15, 19]})
    df2, _, msg = validate.check(df, params={"rule": "less_than", "a": "start", "b": "end"})
    assert "_violates_less_than" in df2.columns
    assert df2["_violates_less_than"].sum() == 2   # rows 2 and 4


def test_validate_drop_violations_removes_rows():
    df = pd.DataFrame({"start": [1, 5, 10, 20], "end": [3, 4, 15, 19]})
    df2, _, msg = validate.drop_violations(df, params={"rule": "less_than", "a": "start", "b": "end"})
    assert len(df2) == 2


def test_validate_equal_rule():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 9, 4]})
    df2, _, _ = validate.check(df, params={"rule": "equal", "a": "a", "b": "b"})
    assert df2["_violates_equal"].sum() == 1


def test_validate_sum_equals():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "total": [3, 5, 8]})
    df2, _, _ = validate.check(df, params={
        "rule": "sum_equals", "cols": ["a", "b"], "total_col": "total",
    })
    assert df2["_violates_sum_equals"].sum() == 1   # row 1: 2+3 != 5? wait 2+3=5 yes; row 2: 3+4=7 != 8


# ==========================================================================
# FEATURE ENGINEERING
# ==========================================================================

def test_feature_eng_creates_bmi():
    df = pd.DataFrame({"weight": [70, 80], "height": [170, 180]})
    df2, _, msg = feature_eng.create(df, params={"name": "BMI", "formula": "weight / (height/100)**2"})
    assert "BMI" in df2.columns
    assert abs(df2["BMI"].iloc[0] - (70 / (1.70**2))) < 1e-6


def test_feature_eng_arithmetic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2, _, _ = feature_eng.create(df, params={"name": "sum", "formula": "a + b"})
    assert df2["sum"].tolist() == [5, 7, 9]


def test_feature_eng_requires_name_and_formula():
    df = pd.DataFrame({"x": [1]})
    _, _, msg = feature_eng.create(df, params={"name": "y"})
    assert "name" in msg.lower() or "formula" in msg.lower()


def test_feature_eng_invalid_formula_returns_error():
    df = pd.DataFrame({"x": [1, 2, 3]})
    _, _, msg = feature_eng.create(df, params={"name": "y", "formula": "import os"})
    assert "formula error" in msg.lower() or "error" in msg.lower()


# ==========================================================================
# SPLITTER
# ==========================================================================

def test_splitter_creates_train_test_pkls(tmp_path):
    df = pd.DataFrame({"x": range(100), "y": [i % 2 for i in range(100)]})
    splitter.split(df, params={
        "target": "y", "test_size": 0.25, "random_state": 0,
        "session_dir": str(tmp_path),
    })
    train = pd.read_pickle(tmp_path / "train.pkl")
    test = pd.read_pickle(tmp_path / "test.pkl")
    assert len(train) == 75
    assert len(test) == 25


def test_splitter_stratifies_by_target(tmp_path):
    df = pd.DataFrame({
        "x": range(100),
        "y": [0] * 80 + [1] * 20,    # 80/20 imbalance
    })
    splitter.split(df, params={
        "target": "y", "test_size": 0.2, "random_state": 0,
        "session_dir": str(tmp_path),
    })
    test = pd.read_pickle(tmp_path / "test.pkl")
    # Stratified — should preserve roughly the 80/20 split
    fracs = test["y"].value_counts(normalize=True)
    assert abs(fracs.get(0, 0) - 0.8) < 0.1


# ==========================================================================
# CLASS BALANCING
# ==========================================================================

def test_oversample_makes_classes_equal(imbalanced_df):
    df, _, _ = balance.oversample(imbalanced_df, params={"target": "target"})
    counts = df["target"].value_counts()
    assert counts.min() == counts.max()


def test_undersample_makes_classes_equal(imbalanced_df):
    df, _, _ = balance.undersample(imbalanced_df, params={"target": "target"})
    counts = df["target"].value_counts()
    assert counts.min() == counts.max()
    assert counts.iloc[0] == 10   # shrunk to minority size


def test_smote_balances_synthetic(imbalanced_df):
    df, _, _ = balance.smote(imbalanced_df, params={"target": "target"})
    counts = df["target"].value_counts()
    assert counts.min() == counts.max()


def test_smote_refuses_non_numeric_features():
    df = pd.DataFrame({"f": ["a", "b"] * 50, "target": [0] * 90 + [1] * 10})
    _, _, msg = balance.smote(df, params={"target": "target"})
    assert "numeric" in msg.lower()


# ==========================================================================
# DIMENSIONALITY REDUCTION
# ==========================================================================

def test_pca_reduces_dimensions(clean_numeric_df):
    df, _, _ = reduce.pca(clean_numeric_df, params={"n_components": 2, "target": "y"})
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    assert len(pc_cols) == 2
    assert "y" in df.columns   # target preserved


def test_variance_threshold_drops_constant():
    df = pd.DataFrame({
        "varying": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "constant": [5] * 10,
    })
    df2, _, _ = reduce.variance_threshold(df, params={"threshold": 0.0})
    assert "constant" not in df2.columns
    assert "varying" in df2.columns


def test_select_k_best_keeps_top_k():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (50, 6)), columns=list("abcdef"))
    df["y"] = (df["a"] + df["b"] > 0).astype(int)
    df2, _, _ = reduce.select_k_best(df, params={"target": "y", "k": 3, "score_func": "f_classif"})
    # 6 features + target → keep 3 features + target = 4 cols
    assert df2.shape[1] == 4
    assert "y" in df2.columns


# ==========================================================================
# MERGE
# ==========================================================================

def test_merge_inner_keeps_matching_rows():
    left = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})
    right = pd.DataFrame({"id": [1, 2, 5], "extra": ["x", "y", "z"]})
    df, _, _ = merge.merge_with(left, params={
        "other_df": right, "left_on": "id", "right_on": "id", "how": "inner",
    })
    assert len(df) == 2


def test_merge_left_keeps_all_left_rows():
    left = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    right = pd.DataFrame({"id": [1], "extra": ["x"]})
    df, _, _ = merge.merge_with(left, params={
        "other_df": right, "left_on": "id", "right_on": "id", "how": "left",
    })
    assert len(df) == 3
    assert df["extra"].isna().sum() == 2


def test_merge_outer_keeps_all_rows():
    left = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    right = pd.DataFrame({"id": [2, 3], "extra": ["x", "y"]})
    df, _, _ = merge.merge_with(left, params={
        "other_df": right, "left_on": "id", "right_on": "id", "how": "outer",
    })
    assert len(df) == 3


def test_merge_validates_columns():
    left = pd.DataFrame({"a": [1]})
    right = pd.DataFrame({"b": [1]})
    _, _, msg = merge.merge_with(left, params={
        "other_df": right, "left_on": "missing", "right_on": "b",
    })
    assert "not in main" in msg.lower() or "left_on" in msg
