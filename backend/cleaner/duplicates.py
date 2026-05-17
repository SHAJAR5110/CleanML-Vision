"""Duplicate row/column handling."""

from __future__ import annotations

import pandas as pd


def drop_duplicate_rows(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    subset = (params or {}).get("subset")
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    n = before - len(df)
    if subset:
        code = f"df = df.drop_duplicates(subset={subset!r}).reset_index(drop=True)"
    else:
        code = "df = df.drop_duplicates().reset_index(drop=True)"
    return df, code, f"Dropped {n} duplicate rows."


def drop_constant_columns(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constants)
    code = (
        f"_constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]\n"
        f"df = df.drop(columns=_constants)"
    )
    return df, code, f"Dropped {len(constants)} constant column(s): {constants}."


def drop_high_missing_columns(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    threshold = float((params or {}).get("threshold", 0.5))
    drops = [c for c in df.columns if df[c].isna().mean() > threshold]
    df = df.drop(columns=drops)
    code = (
        f"_drops = [c for c in df.columns if df[c].isna().mean() > {threshold}]\n"
        f"df = df.drop(columns=_drops)"
    )
    return df, code, f"Dropped {len(drops)} cols with >{threshold:.0%} missing: {drops}."


def drop_high_correlation(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    threshold = float((params or {}).get("threshold", 0.95))
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return df, "", "Need ≥2 numeric columns for correlation pruning."
    corr = num.corr().abs()
    to_drop = set()
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if corr.loc[a, b] > threshold:
                to_drop.add(b)
    df = df.drop(columns=list(to_drop))
    code = (
        f"_num = df.select_dtypes(include='number')\n"
        f"_corr = _num.corr().abs()\n"
        f"_drop = set()\n"
        f"for i, a in enumerate(_corr.columns):\n"
        f"    for b in _corr.columns[i+1:]:\n"
        f"        if _corr.loc[a, b] > {threshold}: _drop.add(b)\n"
        f"df = df.drop(columns=list(_drop))"
    )
    return df, code, f"Dropped {len(to_drop)} highly-correlated cols (>{threshold}): {sorted(to_drop)}."


STRATEGIES = {
    "drop_rows": drop_duplicate_rows,
    "drop_constants": drop_constant_columns,
    "drop_high_missing": drop_high_missing_columns,
    "drop_high_corr": drop_high_correlation,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown dedup strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
