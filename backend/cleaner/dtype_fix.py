"""Dtype repair — coerce strings to proper numeric/bool types."""

from __future__ import annotations

import pandas as pd


def to_numeric(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    s = df[column].astype("string").str.replace(r"[$,%\s]", "", regex=True)
    parsed = pd.to_numeric(s, errors="coerce")
    n_bad = int(parsed.isna().sum() - df[column].isna().sum())
    df[column] = parsed
    code = (
        f"df['{column}'] = pd.to_numeric("
        f"df['{column}'].astype('string').str.replace(r'[$,%\\s]', '', regex=True), errors='coerce')"
    )
    return df, code, f"Coerced '{column}' to numeric (stripped currency/symbols; {n_bad} new NaN)."


def to_boolean(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    norm = df[column].astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "yes": True, "no": False,
               "y": True, "n": False, "1": True, "0": False, "t": True, "f": False}
    df[column] = norm.map(mapping)
    code = (
        f"_map = {mapping!r}\n"
        f"df['{column}'] = df['{column}'].astype('string').str.strip().str.lower().map(_map)"
    )
    return df, code, f"Coerced '{column}' to boolean."


STRATEGIES = {
    "to_numeric": to_numeric,
    "to_boolean": to_boolean,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown dtype strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
