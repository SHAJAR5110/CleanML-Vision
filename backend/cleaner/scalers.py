"""Numeric scalers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)


def _apply_scaler(df, column, cls, name, code_import):
    df = df.copy()
    # Cast to float64 — scalers output floats; nullable Int64 would reject them.
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    scaler = cls()
    arr = s.to_numpy()  # 1D
    mask = ~np.isnan(arr)
    if not mask.any():
        return df, "", f"'{column}' has no numeric values to scale."
    out = arr.copy()
    out[mask] = scaler.fit_transform(arr[mask].reshape(-1, 1)).ravel()
    df[column] = out
    code = (
        f"{code_import}\n"
        f"_s = pd.to_numeric(df['{column}'], errors='coerce').to_numpy().reshape(-1, 1)\n"
        f"_mask = ~pd.isna(_s.ravel())\n"
        f"_scaled = _s.copy(); _scaled[_mask] = {cls.__name__}().fit_transform(_s[_mask].reshape(-1,1)).ravel()\n"
        f"df['{column}'] = _scaled.ravel()"
    )
    return df, code, f"{name} '{column}'."


def standard(df, column, params=None):
    return _apply_scaler(df, column, StandardScaler, "Standard-scaled",
                         "from sklearn.preprocessing import StandardScaler")


def minmax(df, column, params=None):
    return _apply_scaler(df, column, MinMaxScaler, "Min-max scaled",
                         "from sklearn.preprocessing import MinMaxScaler")


def robust(df, column, params=None):
    return _apply_scaler(df, column, RobustScaler, "Robust-scaled",
                         "from sklearn.preprocessing import RobustScaler")


def normalize(df, column, params=None):
    return _apply_scaler(df, column, Normalizer, "L2-normalized",
                         "from sklearn.preprocessing import Normalizer")


def log_transform(df, column, params=None):
    df = df.copy()
    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    shift = max(0.0, -float(s.min()) + 1.0) if s.min() is not None and s.min() <= 0 else 0.0
    df[column] = np.log1p(s + shift)
    code = (
        f"_shift = max(0.0, -df['{column}'].min() + 1.0) if df['{column}'].min() <= 0 else 0.0\n"
        f"df['{column}'] = np.log1p(df['{column}'] + _shift)"
    )
    return df, code, f"Log-transformed '{column}' (shift={shift:.4g})."


STRATEGIES = {
    "standard": standard,
    "minmax": minmax,
    "robust": robust,
    "normalize": normalize,
    "log": log_transform,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown scaler strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
