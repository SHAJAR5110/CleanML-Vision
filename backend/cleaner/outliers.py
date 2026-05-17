"""Outlier detection and treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_remove(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    lo, hi = _iqr_bounds(s.dropna())
    mask = s.between(lo, hi) | s.isna()
    n = int((~mask).sum())
    df = df[mask].reset_index(drop=True)
    code = (
        f"_q1, _q3 = df['{column}'].quantile([0.25, 0.75])\n"
        f"_lo, _hi = _q1 - 1.5*(_q3-_q1), _q3 + 1.5*(_q3-_q1)\n"
        f"df = df[df['{column}'].between(_lo, _hi) | df['{column}'].isna()].reset_index(drop=True)"
    )
    return df, code, f"Removed {n} IQR outliers from '{column}' (bounds [{lo:.4g}, {hi:.4g}])."


def iqr_cap(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    # Cast to float64 — IQR bounds are fractional and pandas 3.0 nullable Int64
    # rejects float values in clip()/fillna().
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    lo, hi = _iqr_bounds(s.dropna())
    capped = int(((s < lo) | (s > hi)).sum())
    df[column] = s.clip(lower=lo, upper=hi)
    code = (
        f"_q1, _q3 = df['{column}'].quantile([0.25, 0.75])\n"
        f"_lo, _hi = _q1 - 1.5*(_q3-_q1), _q3 + 1.5*(_q3-_q1)\n"
        f"df['{column}'] = df['{column}'].clip(lower=_lo, upper=_hi)"
    )
    return df, code, f"Capped {capped} IQR outliers in '{column}' at [{lo:.4g}, {hi:.4g}]."


def zscore_remove(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    thresh = float((params or {}).get("threshold", 3.0))
    z = (s - s.mean()) / s.std(ddof=0)
    mask = (z.abs() <= thresh) | s.isna()
    n = int((~mask).sum())
    df = df[mask].reset_index(drop=True)
    code = (
        f"_z = (df['{column}'] - df['{column}'].mean()) / df['{column}'].std(ddof=0)\n"
        f"df = df[(_z.abs() <= {thresh}) | df['{column}'].isna()].reset_index(drop=True)"
    )
    return df, code, f"Removed {n} z-score outliers (|z|>{thresh}) from '{column}'."


def isolation_forest_remove(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    contamination = float((params or {}).get("contamination", 0.05))
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return df, "", "Isolation Forest needs at least one numeric column."
    X = df[numeric].fillna(df[numeric].median())
    iso = IsolationForest(contamination=contamination, random_state=42)
    pred = iso.fit_predict(X)
    n = int((pred == -1).sum())
    df = df[pred == 1].reset_index(drop=True)
    code = (
        f"from sklearn.ensemble import IsolationForest\n"
        f"_num = df.select_dtypes(include='number').columns\n"
        f"_X = df[_num].fillna(df[_num].median())\n"
        f"_pred = IsolationForest(contamination={contamination}, random_state=42).fit_predict(_X)\n"
        f"df = df[_pred == 1].reset_index(drop=True)"
    )
    return df, code, f"Removed {n} multivariate outliers via Isolation Forest."


def dbscan_remove(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Multivariate density-based anomaly detection.

    Points labelled -1 by DBSCAN are noise (anomalies); they are removed.
    """
    df = df.copy()
    eps = float((params or {}).get("eps", 0.5))
    min_samples = int((params or {}).get("min_samples", 5))
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return df, "", "DBSCAN needs at least one numeric column."
    X = df[numeric].fillna(df[numeric].median())
    X_scaled = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    n_anom = int((labels == -1).sum())
    df = df[labels != -1].reset_index(drop=True)
    code = (
        "from sklearn.cluster import DBSCAN\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "_num = df.select_dtypes(include='number').columns\n"
        "_X = df[_num].fillna(df[_num].median())\n"
        "_X_scaled = StandardScaler().fit_transform(_X)\n"
        f"_labels = DBSCAN(eps={eps}, min_samples={min_samples}).fit_predict(_X_scaled)\n"
        "df = df[_labels != -1].reset_index(drop=True)"
    )
    return df, code, f"DBSCAN flagged {n_anom} anomalies (eps={eps}, min_samples={min_samples})."


STRATEGIES = {
    "iqr_remove": iqr_remove,
    "iqr_cap": iqr_cap,
    "zscore_remove": zscore_remove,
    "isolation_forest": isolation_forest_remove,
    "dbscan": dbscan_remove,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown outlier strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
