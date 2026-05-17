"""Multi-CSV merge / join."""

from __future__ import annotations

import pandas as pd


def merge_with(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Join `df` with a second DataFrame.

    params:
        other_df: pandas DataFrame (required, injected by caller)
        left_on: column from main df (required)
        right_on: column from other df (required)
        how: 'inner' | 'outer' | 'left' | 'right' (default 'inner')
        suffixes: tuple of suffixes for overlapping cols, default ('_x', '_y')
        source_label: filename of the 2nd CSV (for the history message)
    """
    p = params or {}
    other = p.get("other_df")
    left_on = p.get("left_on")
    right_on = p.get("right_on")
    how = p.get("how", "inner")
    suffixes = tuple(p.get("suffixes") or ("_x", "_y"))
    source = p.get("source_label") or "other.csv"

    if other is None or not isinstance(other, pd.DataFrame):
        return df, "", "merge needs params.other_df (pandas DataFrame)."
    if not left_on or left_on not in df.columns:
        return df, "", f"left_on column '{left_on}' not in main dataset."
    if not right_on or right_on not in other.columns:
        return df, "", f"right_on column '{right_on}' not in second dataset."
    if how not in ("inner", "outer", "left", "right"):
        return df, "", f"invalid how={how!r}"

    before = len(df)
    new_df = df.merge(other, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
    after = len(new_df)
    added_cols = [c for c in new_df.columns if c not in df.columns]

    code = (
        f"# Load second CSV: source = {source!r}\n"
        f"other = pd.read_csv({source!r})\n"
        f"df = df.merge(other, how={how!r}, left_on={left_on!r}, right_on={right_on!r}, suffixes={suffixes!r})"
    )
    msg = (
        f"Merged with '{source}' on {left_on}={right_on} (how={how}): "
        f"{before} -> {after} rows, +{len(added_cols)} columns."
    )
    return new_df, code, msg


STRATEGIES = {"join": merge_with}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown merge strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
