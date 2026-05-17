"""Missing-value handling.

Every op returns: (df_new, code_snippet, message)
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer

from .profiler import NAN_STRINGS


def standardize_nan(df: pd.DataFrame, column: str | None = None) -> tuple[pd.DataFrame, str, str]:
    """Replace embedded NaN strings ('?', 'N/A', '-', …) with actual NaN."""
    df = df.copy()
    targets = [column] if column else df.columns
    changed = 0
    for c in targets:
        s = df[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        norm = s.astype(str).str.strip().str.lower()
        mask = norm.isin(NAN_STRINGS) & s.notna()
        changed += int(mask.sum())
        df.loc[mask, c] = None
    scope = f"['{column}']" if column else ".columns"
    code = (
        f"_nan_tokens = {sorted(NAN_STRINGS)!r}\n"
        f"for _c in df{scope}:\n"
        f"    if df[_c].dtype == object or pd.api.types.is_string_dtype(df[_c]):\n"
        f"        _norm = df[_c].astype(str).str.strip().str.lower()\n"
        f"        df.loc[_norm.isin(_nan_tokens), _c] = None"
    )
    return df, code, f"Standardized {changed} embedded NaN strings."


def fill_mean(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    # Cast to float64 — the fill value (mean) is fractional and pandas 3.0
    # nullable Int64 won't accept floats via fillna().
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    val = float(s.mean())
    missing = int(df[column].isna().sum())
    df[column] = s.fillna(val)
    code = f"df['{column}'] = pd.to_numeric(df['{column}'], errors='coerce').astype(float).fillna(df['{column}'].mean())"
    return df, code, f"Filled {missing} missing in '{column}' with mean ({val:.4g})."


def fill_median(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    val = float(s.median())
    missing = int(df[column].isna().sum())
    df[column] = s.fillna(val)
    code = f"df['{column}'] = pd.to_numeric(df['{column}'], errors='coerce').astype(float).fillna(df['{column}'].median())"
    return df, code, f"Filled {missing} missing in '{column}' with median ({val:.4g})."


def fill_mode(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    modes = df[column].mode(dropna=True)
    val = modes.iloc[0] if len(modes) else None
    missing = int(df[column].isna().sum())
    df[column] = df[column].fillna(val)
    code = f"df['{column}'] = df['{column}'].fillna(df['{column}'].mode().iloc[0])"
    return df, code, f"Filled {missing} missing in '{column}' with mode ({val!r})."


def fill_constant(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    val = params.get("value", "Unknown")
    missing = int(df[column].isna().sum())
    df[column] = df[column].fillna(val)
    code = f"df['{column}'] = df['{column}'].fillna({val!r})"
    return df, code, f"Filled {missing} missing in '{column}' with {val!r}."


def fill_ffill(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    missing = int(df[column].isna().sum())
    df[column] = df[column].ffill()
    code = f"df['{column}'] = df['{column}'].ffill()"
    return df, code, f"Forward-filled {missing} missing in '{column}'."


def fill_bfill(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    missing = int(df[column].isna().sum())
    df[column] = df[column].bfill()
    code = f"df['{column}'] = df['{column}'].bfill()"
    return df, code, f"Back-filled {missing} missing in '{column}'."


def fill_knn(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    k = (params or {}).get("k", 5)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if column not in numeric:
        return df, "", f"KNN imputer requires numeric column; '{column}' is not numeric."
    imputer = KNNImputer(n_neighbors=k)
    imputed = imputer.fit_transform(df[numeric])
    df[numeric] = imputed
    code = (
        f"from sklearn.impute import KNNImputer\n"
        f"_num = df.select_dtypes(include='number').columns\n"
        f"df[_num] = KNNImputer(n_neighbors={k}).fit_transform(df[_num])"
    )
    return df, code, f"KNN-imputed missing values across all numeric columns (k={k})."


def drop_rows_missing(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    before = len(df)
    df = df.dropna(subset=[column]).reset_index(drop=True)
    code = f"df = df.dropna(subset=['{column}']).reset_index(drop=True)"
    return df, code, f"Dropped {before - len(df)} rows with missing '{column}'."


def drop_column(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.drop(columns=[column])
    code = f"df = df.drop(columns=['{column}'])"
    return df, code, f"Dropped column '{column}'."


def _standardize_nan_op(df, column, params=None):
    return standardize_nan(df, column)


def fill_all_smart(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Smart fill every NaN across every column: numeric -> median, else -> mode/'Unknown'.

    Also standardizes embedded NaN tokens first.
    """
    df, _, _ = standardize_nan(df, column=None)
    filled = 0
    parts = ["# Standardize NaN tokens then smart-fill every column"]
    parts.append(
        "_nan_tokens = " + repr(sorted(NAN_STRINGS)) + "\n"
        "for _c in df.columns:\n"
        "    if df[_c].dtype == object or pd.api.types.is_string_dtype(df[_c]):\n"
        "        _norm = df[_c].astype(str).str.strip().str.lower()\n"
        "        df.loc[_norm.isin(_nan_tokens), _c] = None"
    )
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # cast to float64 first — pandas 3.0 nullable Int64 rejects float fills
            df[c] = df[c].astype(float)
            val = df[c].median()
            df[c] = df[c].fillna(val)
            parts.append(f"df['{c}'] = df['{c}'].astype(float).fillna(df['{c}'].median())")
        else:
            mode = df[c].mode(dropna=True)
            val = mode.iloc[0] if len(mode) else "Unknown"
            df[c] = df[c].fillna(val)
            parts.append(f"df['{c}'] = df['{c}'].fillna(df['{c}'].mode().iloc[0] if len(df['{c}'].mode()) else 'Unknown')")
        filled += n
    code = "\n".join(parts)
    return df, code, f"Smart-filled {filled} missing cells across all columns."


STRATEGIES = {
    "mean": fill_mean,
    "median": fill_median,
    "mode": fill_mode,
    "constant": fill_constant,
    "ffill": fill_ffill,
    "bfill": fill_bfill,
    "knn": fill_knn,
    "drop_rows": drop_rows_missing,
    "drop_column": drop_column,
    "standardize_nan": _standardize_nan_op,
    "fill_all_smart": fill_all_smart,
}


def apply(df: pd.DataFrame, column: str, strategy: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown missing strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
