"""Categorical encoders."""

from __future__ import annotations

import pandas as pd


def label_encode(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = df[column].astype("category").cat.codes
    code = f"df['{column}'] = df['{column}'].astype('category').cat.codes"
    return df, code, f"Label-encoded '{column}'."


def one_hot(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    drop_first = bool((params or {}).get("drop_first", False))
    new = pd.get_dummies(df[column], prefix=column, drop_first=drop_first, dummy_na=False)
    df = df.drop(columns=[column]).join(new)
    code = (
        f"df = pd.concat([df.drop(columns=['{column}']), "
        f"pd.get_dummies(df['{column}'], prefix='{column}', drop_first={drop_first})], axis=1)"
    )
    return df, code, f"One-hot encoded '{column}' → {new.shape[1]} columns."


def ordinal_encode(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    order = params.get("order")
    if not order:
        return df, "", "ordinal_encode requires params.order (list of categories)."
    mapping = {v: i for i, v in enumerate(order)}
    df[column] = df[column].map(mapping)
    code = f"df['{column}'] = df['{column}'].map({mapping!r})"
    return df, code, f"Ordinal-encoded '{column}' with order {order}."


def frequency_encode(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    freq = df[column].value_counts(normalize=True)
    df[column] = df[column].map(freq)
    code = (
        f"_freq = df['{column}'].value_counts(normalize=True)\n"
        f"df['{column}'] = df['{column}'].map(_freq)"
    )
    return df, code, f"Frequency-encoded '{column}'."


def target_encode(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    target = params.get("target")
    if not target or target not in df.columns:
        return df, "", "target_encode requires params.target (existing column)."
    means = df.groupby(column)[target].mean()
    df[column] = df[column].map(means)
    code = (
        f"_tgt = df.groupby('{column}')['{target}'].mean()\n"
        f"df['{column}'] = df['{column}'].map(_tgt)"
    )
    return df, code, f"Target-encoded '{column}' by mean of '{target}'."


def binary_encode(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Map any 2-value column to 0/1."""
    df = df.copy()
    vals = df[column].dropna().unique()
    if len(vals) != 2:
        return df, "", f"binary_encode needs exactly 2 unique values in '{column}', got {len(vals)}."
    a, b = sorted(vals, key=lambda x: str(x))
    mapping = {a: 0, b: 1}
    df[column] = df[column].map(mapping)
    code = f"df['{column}'] = df['{column}'].map({mapping!r})"
    return df, code, f"Binary-encoded '{column}': {a!r}→0, {b!r}→1."


STRATEGIES = {
    "label": label_encode,
    "onehot": one_hot,
    "ordinal": ordinal_encode,
    "frequency": frequency_encode,
    "target": target_encode,
    "binary": binary_encode,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown encoder strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
