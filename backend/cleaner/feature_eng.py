"""Feature engineering — create new columns from formula expressions.

Uses pandas `df.eval()` which has a restricted, safe expression language:
arithmetic (+ - * / ** % //), comparison, logical, and a few functions.
Does NOT allow arbitrary Python (no import, attribute access, function calls).
"""

from __future__ import annotations

import pandas as pd


def create(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Create a new column from a formula.

    params:
        name: new column name (required)
        formula: expression like 'weight / (height/100)**2' (required)
    """
    p = params or {}
    name = p.get("name")
    formula = p.get("formula")
    if not name or not formula:
        return df, "", "feature_eng.create requires params.name and params.formula"

    df = df.copy()
    expr = f"`{name}` = {formula}"
    try:
        df = df.eval(expr)
    except Exception as e:
        return df, "", f"formula error: {e}"
    code = f"df = df.eval('{expr.replace(chr(92), chr(92)*2)}')"
    n_valid = int(df[name].notna().sum())
    return df, code, f"Created '{name}' = {formula}  ({n_valid} valid values)."


STRATEGIES = {
    "create": create,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown feature_eng strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
