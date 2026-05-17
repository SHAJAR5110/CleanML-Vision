"""Tests for the pipeline orchestrator + Magic Clean."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cleaner import magic, pipeline


def test_apply_op_routes_to_family(dirty_df):
    df, code, msg = pipeline.apply_op(dirty_df.copy(), {
        "family": "missing", "strategy": "median", "column": "age",
    })
    assert df["age"].isna().sum() == 0


def test_apply_op_rejects_unknown_family():
    with pytest.raises(ValueError):
        pipeline.apply_op(pd.DataFrame({"x": [1]}), {"family": "bogus", "strategy": "x"})


def test_history_round_trip(tmp_path):
    df = pd.DataFrame({"x": [1, None, 3]})
    df.to_pickle(tmp_path / "original.pkl")
    df.to_pickle(tmp_path / "current.pkl")
    pipeline.save_history(tmp_path, [])

    entry = {"op": {"family": "missing", "strategy": "median", "column": "x"},
             "code": "df['x'] = df['x'].fillna(df['x'].median())",
             "message": "filled"}
    pipeline.append_history(tmp_path, entry)
    loaded = pipeline.load_history(tmp_path)
    assert loaded == [entry]


def test_undo_replays_history_minus_last(tmp_path):
    """Undo should re-apply all ops except the most recent one."""
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
    df.to_pickle(tmp_path / "original.pkl")
    df.to_pickle(tmp_path / "current.pkl")

    pipeline.append_history(tmp_path, {
        "op": {"family": "missing", "strategy": "median", "column": "x"},
        "code": "", "message": "filled",
    })
    pipeline.append_history(tmp_path, {
        "op": {"family": "missing", "strategy": "drop_column", "column": "x"},
        "code": "", "message": "dropped",
    })

    df_after, h_after = pipeline.undo_last(tmp_path)
    assert "x" in df_after.columns          # last op (drop) undone
    assert df_after["x"].isna().sum() == 0  # but fill still applied
    assert len(h_after) == 1


def test_reset_restores_original(tmp_path):
    original = pd.DataFrame({"x": [1, 2, 3]})
    current = pd.DataFrame({"y": [4, 5]})
    original.to_pickle(tmp_path / "original.pkl")
    current.to_pickle(tmp_path / "current.pkl")
    pipeline.append_history(tmp_path, {
        "op": {"family": "missing", "strategy": "drop_column", "column": "x"},
        "code": "", "message": "",
    })

    df = pipeline.reset(tmp_path)
    assert df.equals(original)
    assert pipeline.load_history(tmp_path) == []


# ==========================================================================
# MAGIC CLEAN
# ==========================================================================

def test_magic_clean_makes_dataset_ml_ready(dirty_df):
    """Magic Clean should produce an entirely numeric/boolean DataFrame."""
    df, history = magic.run(dirty_df)
    assert len(history) > 0
    # After magic clean, all remaining columns should be numeric or bool
    for c in df.columns:
        dt = df[c].dtype
        assert pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]), \
            f"Column {c!r} has non-ML-ready dtype {dt}"


def test_magic_clean_drops_constant_columns(dirty_df):
    df, _ = magic.run(dirty_df)
    assert "const" not in df.columns


def test_magic_clean_no_missing_values(dirty_df):
    df, _ = magic.run(dirty_df)
    assert df.isna().sum().sum() == 0


def test_magic_clean_history_each_entry_has_code(dirty_df):
    """Generated code is needed for the reproducible-notebook feature."""
    _, history = magic.run(dirty_df)
    for h in history:
        assert "op" in h
        assert "message" in h
        # code may be empty for no-op ops but key must exist
        assert "code" in h
