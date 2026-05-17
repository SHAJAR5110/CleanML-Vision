"""Dimensionality reduction + feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import StandardScaler


def pca(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    p = params or {}
    n_components = int(p.get("n_components", 5))
    target = p.get("target")
    keep_target = bool(target and target in df.columns)
    drop_original = bool(p.get("drop_original", True))

    feature_cols = [c for c in df.columns if c != target]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 2:
        return df, "", "PCA needs ≥2 numeric feature columns."
    n_components = min(n_components, len(numeric))

    X = df[numeric].fillna(df[numeric].median())
    X_scaled = StandardScaler().fit_transform(X)
    p_obj = PCA(n_components=n_components)
    components = p_obj.fit_transform(X_scaled)
    explained = p_obj.explained_variance_ratio_

    pc_df = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df.index,
    )
    other_cols = [c for c in df.columns if c not in (numeric if drop_original else []) and c != target]
    parts = [df[other_cols].reset_index(drop=True), pc_df.reset_index(drop=True)]
    if keep_target:
        parts.append(df[[target]].reset_index(drop=True))
    df_new = pd.concat(parts, axis=1)

    explained_pct = ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(explained))
    code = (
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n"
        f"_num = {numeric!r}\n"
        "_X = df[_num].fillna(df[_num].median())\n"
        "_Xs = StandardScaler().fit_transform(_X)\n"
        f"_pcs = PCA(n_components={n_components}).fit_transform(_Xs)\n"
        f"_pc_df = pd.DataFrame(_pcs, columns=[f'PC{{i+1}}' for i in range({n_components})], index=df.index)\n"
        + (f"df = pd.concat([df.drop(columns=_num).reset_index(drop=True), _pc_df.reset_index(drop=True)], axis=1)"
           if drop_original else
           "df = pd.concat([df.reset_index(drop=True), _pc_df.reset_index(drop=True)], axis=1)")
    )
    return df_new, code, f"PCA → {n_components} components. Explained variance: {explained_pct}."


def variance_threshold(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    p = params or {}
    threshold = float(p.get("threshold", 0.0))
    target = p.get("target")
    feature_cols = [c for c in df.columns if c != target]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return df, "", "VarianceThreshold needs numeric columns."

    X = df[numeric].fillna(0)
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X)
    kept = [c for c, k in zip(numeric, vt.get_support()) if k]
    dropped = [c for c in numeric if c not in kept]
    df = df.drop(columns=dropped)
    code = (
        "from sklearn.feature_selection import VarianceThreshold\n"
        f"_num = {numeric!r}\n"
        f"_keep = [c for c, k in zip(_num, VarianceThreshold(threshold={threshold}).fit(df[_num].fillna(0)).get_support()) if k]\n"
        f"df = df.drop(columns=[c for c in _num if c not in _keep])"
    )
    return df, code, f"Dropped {len(dropped)} low-variance cols: {dropped}."


def select_k_best(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    p = params or {}
    target = p.get("target")
    k = int(p.get("k", 10))
    score_func_name = p.get("score_func", "f_classif")
    if not target or target not in df.columns:
        return df, "", "select_k_best requires a target column."

    feature_cols = [c for c in df.columns if c != target]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return df, "", "select_k_best needs numeric feature columns."
    k = min(k, len(numeric))

    score_funcs = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
    }
    score_func = score_funcs.get(score_func_name, f_classif)

    X = df[numeric].fillna(df[numeric].median())
    y = df[target]
    selector = SelectKBest(score_func=score_func, k=k).fit(X, y)
    kept = [c for c, m in zip(numeric, selector.get_support()) if m]
    dropped = [c for c in numeric if c not in kept]
    df = df.drop(columns=dropped)
    code = (
        f"from sklearn.feature_selection import SelectKBest, {score_func_name}\n"
        f"_num = {numeric!r}\n"
        f"_X, _y = df[_num].fillna(df[_num].median()), df['{target}']\n"
        f"_sel = SelectKBest(score_func={score_func_name}, k={k}).fit(_X, _y)\n"
        f"_keep = [c for c, m in zip(_num, _sel.get_support()) if m]\n"
        f"df = df.drop(columns=[c for c in _num if c not in _keep])"
    )
    return df, code, f"Kept top {k} features by {score_func_name}: {kept[:5]}{'...' if len(kept) > 5 else ''}. Dropped {len(dropped)}."


STRATEGIES = {
    "pca": pca,
    "variance_threshold": variance_threshold,
    "select_k_best": select_k_best,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown reduce strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
