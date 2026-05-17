"""Pipeline orchestrator + history tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import (
    balance, datetime_fix, dtype_fix, duplicates, encoders, feature_eng,
    label_norm, merge, missing, outliers, reduce, scalers, splitter,
    text_clean, validate,
)

# (module, family_name)
FAMILIES = {
    "missing":     missing,
    "outliers":    outliers,
    "encoders":    encoders,
    "scalers":     scalers,
    "duplicates":  duplicates,
    "text":        text_clean,
    "datetime":    datetime_fix,
    "dtype":       dtype_fix,
    "label_norm":  label_norm,
    "validate":    validate,
    "feature_eng": feature_eng,
    "splitter":    splitter,
    "balance":     balance,
    "reduce":      reduce,
    "merge":       merge,
}


def apply_op(df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, str, str]:
    """Apply a single operation. op = {family, strategy, column?, params?}"""
    family = op["family"]
    strategy = op["strategy"]
    column = op.get("column")
    params = op.get("params") or {}
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    return FAMILIES[family].apply(df, column, strategy, params)


def history_path(session_dir: Path) -> Path:
    return session_dir / "history.json"


def load_history(session_dir: Path) -> list[dict]:
    p = history_path(session_dir)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_history(session_dir: Path, history: list[dict]) -> None:
    history_path(session_dir).write_text(json.dumps(history, indent=2), encoding="utf-8")


def append_history(session_dir: Path, entry: dict) -> list[dict]:
    h = load_history(session_dir)
    h.append(entry)
    save_history(session_dir, h)
    return h


def undo_last(session_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Re-apply all ops except the last. Returns (df, new_history)."""
    h = load_history(session_dir)
    if not h:
        df = pd.read_pickle(session_dir / "original.pkl")
        return df, []
    h = h[:-1]
    df = pd.read_pickle(session_dir / "original.pkl")
    for entry in h:
        df, _, _ = apply_op(df, entry["op"])
    save_history(session_dir, h)
    df.to_pickle(session_dir / "current.pkl")
    return df, h


def reset(session_dir: Path) -> pd.DataFrame:
    df = pd.read_pickle(session_dir / "original.pkl")
    df.to_pickle(session_dir / "current.pkl")
    save_history(session_dir, [])
    return df
