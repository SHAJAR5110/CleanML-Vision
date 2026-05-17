"""Datetime parsing and feature extraction."""

from __future__ import annotations

import pandas as pd


def parse(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    fmt = (params or {}).get("format")
    if fmt:
        df[column] = pd.to_datetime(df[column], format=fmt, errors="coerce")
        code = f"df['{column}'] = pd.to_datetime(df['{column}'], format={fmt!r}, errors='coerce')"
    else:
        df[column] = pd.to_datetime(df[column], errors="coerce")
        code = f"df['{column}'] = pd.to_datetime(df['{column}'], errors='coerce')"
    return df, code, f"Parsed '{column}' as datetime."


def extract_parts(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    s = pd.to_datetime(df[column], errors="coerce")
    parts = (params or {}).get("parts") or ["year", "month", "day", "weekday"]
    added = []
    for p in parts:
        col = f"{column}_{p}"
        if p == "year": df[col] = s.dt.year
        elif p == "month": df[col] = s.dt.month
        elif p == "day": df[col] = s.dt.day
        elif p == "hour": df[col] = s.dt.hour
        elif p == "weekday": df[col] = s.dt.weekday
        elif p == "quarter": df[col] = s.dt.quarter
        else: continue
        added.append(col)
    code = (
        f"_s = pd.to_datetime(df['{column}'], errors='coerce')\n"
        + "\n".join(
            {
                "year":    f"df['{column}_year'] = _s.dt.year",
                "month":   f"df['{column}_month'] = _s.dt.month",
                "day":     f"df['{column}_day'] = _s.dt.day",
                "hour":    f"df['{column}_hour'] = _s.dt.hour",
                "weekday": f"df['{column}_weekday'] = _s.dt.weekday",
                "quarter": f"df['{column}_quarter'] = _s.dt.quarter",
            }[p] for p in parts if p in {"year","month","day","hour","weekday","quarter"}
        )
    )
    return df, code, f"Extracted {added} from '{column}'."


STRATEGIES = {
    "parse": parse,
    "extract": extract_parts,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown datetime strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
