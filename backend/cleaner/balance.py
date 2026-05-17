"""Class balancing — oversample / undersample / SMOTE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _class_counts(df: pd.DataFrame, target: str) -> dict:
    return {str(k): int(v) for k, v in df[target].value_counts().items()}


def oversample(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Random oversampling — duplicate minority rows until each class equals the majority."""
    p = params or {}
    target = p.get("target")
    if not target or target not in df.columns:
        return df, "", "balance.oversample requires a target column."
    rs = int(p.get("random_state", 42))
    counts = df[target].value_counts()
    max_n = counts.max()
    parts = []
    for cls, n in counts.items():
        sub = df[df[target] == cls]
        if n < max_n:
            extra = sub.sample(n=max_n - n, replace=True, random_state=rs)
            parts.append(pd.concat([sub, extra]))
        else:
            parts.append(sub)
    df_new = pd.concat(parts).sample(frac=1, random_state=rs).reset_index(drop=True)
    before, after = _class_counts(df, target), _class_counts(df_new, target)
    code = (
        f"_counts = df['{target}'].value_counts()\n"
        "_parts = []\n"
        f"for cls, n in _counts.items():\n"
        f"    sub = df[df['{target}'] == cls]\n"
        f"    if n < _counts.max():\n"
        f"        sub = pd.concat([sub, sub.sample(n=_counts.max()-n, replace=True, random_state={rs})])\n"
        f"    _parts.append(sub)\n"
        f"df = pd.concat(_parts).sample(frac=1, random_state={rs}).reset_index(drop=True)"
    )
    return df_new, code, f"Oversampled '{target}': {before} -> {after}."


def undersample(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Random undersampling — shrink majority to match minority."""
    p = params or {}
    target = p.get("target")
    if not target or target not in df.columns:
        return df, "", "balance.undersample requires a target column."
    rs = int(p.get("random_state", 42))
    counts = df[target].value_counts()
    min_n = counts.min()
    parts = [df[df[target] == cls].sample(n=min_n, random_state=rs) for cls in counts.index]
    df_new = pd.concat(parts).sample(frac=1, random_state=rs).reset_index(drop=True)
    before, after = _class_counts(df, target), _class_counts(df_new, target)
    code = (
        f"_counts = df['{target}'].value_counts()\n"
        f"df = pd.concat([df[df['{target}'] == cls].sample(n=_counts.min(), random_state={rs}) for cls in _counts.index])\n"
        f"df = df.sample(frac=1, random_state={rs}).reset_index(drop=True)"
    )
    return df_new, code, f"Undersampled '{target}': {before} -> {after}."


def smote(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """SMOTE: synthetic minority oversampling. Requires all-numeric features."""
    p = params or {}
    target = p.get("target")
    if not target or target not in df.columns:
        return df, "", "balance.smote requires a target column."
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return df, "", "SMOTE needs 'imbalanced-learn' installed."

    rs = int(p.get("random_state", 42))
    features = [c for c in df.columns if c != target]
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return df, "", f"SMOTE needs all-numeric features. Non-numeric: {non_numeric}"

    X = df[features].fillna(df[features].median())
    y = df[target]
    sm = SMOTE(random_state=rs)
    X_res, y_res = sm.fit_resample(X, y)
    df_new = pd.concat([X_res, y_res.rename(target).reset_index(drop=True)], axis=1)
    before, after = _class_counts(df, target), _class_counts(df_new, target)
    code = (
        "from imblearn.over_sampling import SMOTE\n"
        f"_features = [c for c in df.columns if c != '{target}']\n"
        f"_X, _y = df[_features].fillna(df[_features].median()), df['{target}']\n"
        f"_Xr, _yr = SMOTE(random_state={rs}).fit_resample(_X, _y)\n"
        f"df = pd.concat([_Xr, _yr.rename('{target}').reset_index(drop=True)], axis=1)"
    )
    return df_new, code, f"SMOTE balanced '{target}': {before} -> {after}."


STRATEGIES = {
    "oversample": oversample,
    "undersample": undersample,
    "smote": smote,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown balance strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
