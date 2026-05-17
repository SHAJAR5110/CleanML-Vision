"""Cross-field validation — flag or drop rows that fail a logical rule between columns."""

from __future__ import annotations

import pandas as pd


def _build_violation_mask(df: pd.DataFrame, rule: str, params: dict) -> pd.Series:
    if rule == "less_than":
        a, b = params["a"], params["b"]
        return ~(df[a] < df[b])
    if rule == "less_or_equal":
        a, b = params["a"], params["b"]
        return ~(df[a] <= df[b])
    if rule == "equal":
        a, b = params["a"], params["b"]
        return df[a] != df[b]
    if rule == "sum_equals":
        cols = params["cols"]
        total = params.get("total")
        target_col = params.get("total_col")
        s = df[cols].sum(axis=1)
        if target_col:
            return s != df[target_col]
        return s != float(total)
    if rule == "age_dob":
        # age_col should equal (today - dob_col).years (within ±1)
        age_col = params["age_col"]
        dob_col = params["dob_col"]
        ref = pd.Timestamp(params.get("reference") or "today")
        dob = pd.to_datetime(df[dob_col], errors="coerce")
        expected = ((ref - dob).dt.days / 365.25).round()
        return (df[age_col].astype(float) - expected).abs() > 1
    raise ValueError(f"unknown validation rule: {rule}")


def check(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Add a column `_violates_<rule>` (True for offending rows). Does not drop."""
    df = df.copy()
    p = params or {}
    rule = p.get("rule")
    if not rule:
        return df, "", "validate.check requires params.rule"
    mask = _build_violation_mask(df, rule, p)
    flag_col = p.get("flag_col") or f"_violates_{rule}"
    df[flag_col] = mask.astype(bool)
    n = int(mask.sum())
    code = f"# Cross-field validation '{rule}'; flags violations in '{flag_col}'"
    return df, code, f"Flagged {n} rows that violate rule '{rule}' in '{flag_col}'."


def drop_violations(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    p = params or {}
    rule = p.get("rule")
    if not rule:
        return df, "", "validate.drop_violations requires params.rule"
    mask = _build_violation_mask(df, rule, p)
    n = int(mask.sum())
    df = df[~mask].reset_index(drop=True)
    code = f"# Dropped rows that violated rule '{rule}'"
    return df, code, f"Dropped {n} rows that violated rule '{rule}'."


STRATEGIES = {
    "check": check,
    "drop_violations": drop_violations,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown validate strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
