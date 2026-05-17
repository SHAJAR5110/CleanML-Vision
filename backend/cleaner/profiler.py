"""Smart auto-profiler.

Detects per column:
  - inferred type (numeric / categorical / datetime / text / boolean / id_like / constant)
  - missing count (including embedded NaN strings like "?", "N/A", "null", "-")
  - uniqueness, cardinality bucket
  - outlier count (IQR) for numeric
  - skewness for numeric
  - basic stats
  - sample values

Computes overall data quality score (0-100).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

NAN_STRINGS = {
    "", " ", "?", "-", "--", "n/a", "na", "nan", "null", "none",
    "missing", "unknown", "#n/a", "#null!", "(null)", ".",
}

DATE_HINT_RE = re.compile(
    r"^\s*\d{1,4}[-/.\s]\d{1,2}[-/.\s]\d{1,4}(\s+\d{1,2}:\d{2}(:\d{2})?)?\s*$"
)


def _is_text_like(series: pd.Series) -> bool:
    """True if series holds strings — covers both `object` and pandas 3 `str` dtypes."""
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    ) and not pd.api.types.is_numeric_dtype(series)


def _count_embedded_nans(series: pd.Series) -> int:
    if not _is_text_like(series):
        return 0
    s = series.dropna().astype(str).str.strip().str.lower()
    return int(s.isin(NAN_STRINGS).sum())


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _is_text_like(series):
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    hits = sample.str.match(DATE_HINT_RE).sum()
    if hits >= max(3, len(sample) * 0.6):
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().mean() >= 0.8
        except Exception:
            return False
    return False


def _is_numeric_string(series: pd.Series) -> bool:
    """True if a string column is actually numeric (e.g. '1,234' or '$5.00')."""
    if not _is_text_like(series):
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    cleaned = sample.str.replace(r"[$,%\s]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    vals = set(str(v).strip().lower() for v in series.dropna().unique()[:20])
    bool_sets = [
        {"true", "false"}, {"yes", "no"}, {"y", "n"}, {"0", "1"}, {"t", "f"},
    ]
    return any(vals.issubset(b) and len(vals) > 0 for b in bool_sets)


def _cardinality_bucket(nunique: int, n: int) -> str:
    if n == 0:
        return "empty"
    if nunique <= 1:
        return "constant"
    if nunique == 2:
        return "binary"
    ratio = nunique / n
    if ratio >= 0.95:
        return "unique"
    if nunique <= 20:
        return "low_card"
    if nunique <= 100:
        return "medium_card"
    return "high_card"


def _infer_type(series: pd.Series, name: str) -> str:
    n = len(series)
    nunique = series.nunique(dropna=True)
    if nunique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series) or _is_boolean_like(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series) or _is_datetime_like(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series) or _is_numeric_string(series):
        if nunique / max(n, 1) >= 0.95 and "id" in name.lower():
            return "id_like"
        return "numeric"
    bucket = _cardinality_bucket(nunique, n)
    if bucket in ("low_card", "medium_card", "binary"):
        return "categorical"
    if bucket == "unique":
        return "id_like" if "id" in name.lower() else "text"
    return "text"


def _outlier_count_iqr(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _column_profile(series: pd.Series, name: str) -> dict:
    n = len(series)
    null_count = int(series.isna().sum())
    embedded = _count_embedded_nans(series)
    total_missing = null_count + embedded
    nunique = int(series.nunique(dropna=True))
    inferred = _infer_type(series, name)
    bucket = _cardinality_bucket(nunique, n)

    out: dict[str, Any] = {
        "name": name,
        "inferred_type": inferred,
        "dtype": str(series.dtype),
        "count": n,
        "missing": total_missing,
        "missing_pct": round(total_missing / n * 100, 2) if n else 0,
        "null_count": null_count,
        "embedded_nan_count": embedded,
        "unique": nunique,
        "unique_pct": round(nunique / n * 100, 2) if n else 0,
        "cardinality": bucket,
    }

    if inferred == "numeric":
        s = pd.to_numeric(series, errors="coerce").dropna()
        out["min"] = _safe_float(s.min()) if len(s) else None
        out["max"] = _safe_float(s.max()) if len(s) else None
        out["mean"] = _safe_float(s.mean()) if len(s) else None
        out["median"] = _safe_float(s.median()) if len(s) else None
        out["std"] = _safe_float(s.std()) if len(s) else None
        out["skew"] = _safe_float(s.skew()) if len(s) > 2 else None
        out["outlier_count"] = _outlier_count_iqr(series)
    elif inferred in ("categorical", "boolean", "text"):
        top = series.dropna().astype(str).value_counts().head(5)
        out["top_values"] = [{"value": k, "count": int(v)} for k, v in top.items()]
    elif inferred == "datetime":
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            out["min"] = str(parsed.min()) if parsed.notna().any() else None
            out["max"] = str(parsed.max()) if parsed.notna().any() else None
        except Exception:
            pass

    # warnings
    warnings = []
    if out["missing_pct"] > 50:
        warnings.append("very_high_missing")
    elif out["missing_pct"] > 10:
        warnings.append("high_missing")
    if embedded > 0:
        warnings.append("embedded_nan_strings")
    if bucket == "constant":
        warnings.append("constant_column")
    if inferred == "id_like":
        warnings.append("id_like_column")
    if inferred == "numeric" and out.get("outlier_count", 0) > 0:
        if n and out["outlier_count"] / n > 0.05:
            warnings.append("many_outliers")
    if inferred == "numeric" and out.get("skew") is not None and abs(out["skew"]) > 2:
        warnings.append("highly_skewed")
    out["warnings"] = warnings

    return out


def _quality_score(profile_cols: list[dict], duplicates: int, n_rows: int) -> int:
    if not profile_cols:
        return 0
    score = 100.0
    avg_missing = sum(c["missing_pct"] for c in profile_cols) / len(profile_cols)
    score -= min(40, avg_missing * 0.8)
    constants = sum(1 for c in profile_cols if c["cardinality"] == "constant")
    score -= min(15, constants * 5)
    high_missing_cols = sum(1 for c in profile_cols if c["missing_pct"] > 50)
    score -= min(15, high_missing_cols * 3)
    dup_pct = (duplicates / n_rows * 100) if n_rows else 0
    score -= min(10, dup_pct * 0.5)
    embedded_cols = sum(1 for c in profile_cols if c["embedded_nan_count"] > 0)
    score -= min(10, embedded_cols * 2)
    skewed = sum(1 for c in profile_cols if "highly_skewed" in c["warnings"])
    score -= min(5, skewed)
    outlier_heavy = sum(1 for c in profile_cols if "many_outliers" in c["warnings"])
    score -= min(5, outlier_heavy * 2)
    return max(0, min(100, int(round(score))))


def profile_dataframe(df: pd.DataFrame) -> dict:
    cols = [_column_profile(df[c], c) for c in df.columns]
    duplicates = int(df.duplicated().sum())
    n_rows, n_cols = df.shape
    score = _quality_score(cols, duplicates, n_rows)
    grade = (
        "A" if score >= 90 else
        "B" if score >= 75 else
        "C" if score >= 60 else
        "D" if score >= 40 else "F"
    )

    total_missing_cells = sum(c["missing"] for c in cols)
    total_cells = n_rows * n_cols if n_cols else 0
    missing_matrix = []
    if n_rows and n_cols:
        sample = df.head(200)
        for c in df.columns:
            missing_matrix.append(sample[c].isna().astype(int).tolist())

    return {
        "rows": n_rows,
        "cols": n_cols,
        "duplicate_rows": duplicates,
        "total_missing": total_missing_cells,
        "missing_pct": round(total_missing_cells / total_cells * 100, 2) if total_cells else 0,
        "quality_score": score,
        "grade": grade,
        "columns": cols,
        "missing_matrix": missing_matrix,
        "missing_matrix_cols": list(df.columns),
    }
