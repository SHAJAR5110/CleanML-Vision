"""Magic Clean — one-click intelligent cleaner.

Strategy planning:
1. Standardize embedded NaN strings everywhere.
2. Drop fully-constant columns and columns with >70% missing.
3. Drop duplicate rows.
4. For each remaining column, choose per-type strategy:
   - id_like  -> drop (not predictive)
   - constant -> already dropped
   - datetime -> parse
   - boolean  -> coerce to bool
   - numeric:
        - if string-typed (e.g. '$1,200') -> coerce to numeric
        - fill missing with median
        - outliers: if >5% -> cap at IQR fence, else remove
        - if highly skewed -> log transform
   - categorical:
        - fill missing with mode
        - if low_card / binary -> one-hot
        - if medium_card / high_card -> frequency-encode
   - text:
        - drop (long free text not useful by default)
"""

from __future__ import annotations

import pandas as pd

from . import datetime_fix, dtype_fix, duplicates, encoders, missing, outliers, scalers, text_clean
from .pipeline import apply_op
from .profiler import profile_dataframe


def plan(df: pd.DataFrame) -> list[dict]:
    profile = profile_dataframe(df)
    ops: list[dict] = []

    # Step 1: standardize embedded NaN strings across all string columns
    if any(c["embedded_nan_count"] > 0 for c in profile["columns"]):
        ops.append({"family": "missing", "strategy": "standardize_nan_global"})

    # Step 2: drop fully-constant columns
    if any(c["cardinality"] == "constant" for c in profile["columns"]):
        ops.append({"family": "duplicates", "strategy": "drop_constants"})

    # Step 3: drop columns with very high missing (>70%)
    high_miss = [c for c in profile["columns"] if c["missing_pct"] > 70]
    if high_miss:
        ops.append({
            "family": "duplicates", "strategy": "drop_high_missing",
            "params": {"threshold": 0.7},
        })

    # Step 4: drop duplicate rows
    if profile["duplicate_rows"] > 0:
        ops.append({"family": "duplicates", "strategy": "drop_rows"})

    # Step 5: per-column ops
    drop_after = {c["name"] for c in high_miss} | {
        c["name"] for c in profile["columns"] if c["cardinality"] == "constant"
    }
    for c in profile["columns"]:
        name = c["name"]
        if name in drop_after:
            continue
        t = c["inferred_type"]
        missing_pct = c["missing_pct"]
        card = c["cardinality"]
        skew = c.get("skew")
        out_pct = (c.get("outlier_count", 0) / c["count"] * 100) if c["count"] else 0

        if t == "id_like":
            ops.append({"family": "missing", "strategy": "drop_column", "column": name})
            continue

        if t == "text":
            ops.append({"family": "missing", "strategy": "drop_column", "column": name})
            continue

        if t == "datetime":
            ops.append({"family": "datetime", "strategy": "parse", "column": name})
            ops.append({
                "family": "datetime", "strategy": "extract", "column": name,
                "params": {"parts": ["year", "month", "weekday"]},
            })
            ops.append({"family": "missing", "strategy": "drop_column", "column": name})
            continue

        if t == "boolean":
            ops.append({"family": "dtype", "strategy": "to_boolean", "column": name})
            if missing_pct > 0:
                ops.append({"family": "missing", "strategy": "mode", "column": name})
            continue

        if t == "numeric":
            # if it was a numeric-string ("$1,200"), coerce first
            if c["dtype"] not in ("int64", "float64", "Int64", "Float64", "int32", "float32"):
                ops.append({"family": "dtype", "strategy": "to_numeric", "column": name})
            if missing_pct > 0:
                ops.append({"family": "missing", "strategy": "median", "column": name})
            if out_pct > 0:
                strat = "iqr_cap" if out_pct > 5 else "iqr_remove"
                ops.append({"family": "outliers", "strategy": strat, "column": name})
            if skew is not None and abs(skew) > 2:
                ops.append({"family": "scalers", "strategy": "log", "column": name})
            continue

        if t == "categorical":
            if missing_pct > 0:
                ops.append({"family": "missing", "strategy": "mode", "column": name})
            if card in ("binary", "low_card"):
                ops.append({"family": "encoders", "strategy": "onehot", "column": name})
            else:  # medium_card, high_card
                ops.append({"family": "encoders", "strategy": "frequency", "column": name})
            continue

    return ops


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Execute the magic plan, returning (df, history).

    Each history entry: {op, code, message}.
    """
    plan_ops = plan(df)
    history: list[dict] = []
    for op in plan_ops:
        # special case: standardize_nan_global -> apply standardize_nan across all string cols
        if op.get("strategy") == "standardize_nan_global":
            from . import missing as _m
            df, code, msg = _m.standardize_nan(df, column=None)
        else:
            df, code, msg = apply_op(df, op)
        history.append({"op": op, "code": code, "message": msg})
    return df, history
