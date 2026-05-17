"""Inconsistent categorical label normalization (no rapidfuzz dependency).

Detects groups of similar string values like {"Male", "M", "male", "MALE"} and
proposes a canonical replacement (most frequent form).
"""

from __future__ import annotations

import difflib
import re

import pandas as pd


def _normalize_token(s: str) -> str:
    """Lowercase + strip + collapse whitespace; used to bucket similar forms."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def detect_groups(series: pd.Series, threshold: float = 0.85) -> list[dict]:
    """Return a list of label clusters with their canonical (most-frequent) form.

    Each cluster: {canonical, members: [{value, count}, ...], total}
    """
    s = series.dropna().astype(str)
    counts = s.value_counts()
    if counts.empty:
        return []

    # Step 1: exact-after-normalization bucketing
    norm_buckets: dict[str, list[tuple[str, int]]] = {}
    for value, count in counts.items():
        key = _normalize_token(value)
        norm_buckets.setdefault(key, []).append((value, int(count)))

    # Step 2: fuzzy-merge buckets whose normalized keys are similar
    bucket_keys = list(norm_buckets.keys())
    merged: list[list[str]] = []
    used: set[int] = set()
    for i, k in enumerate(bucket_keys):
        if i in used:
            continue
        group = [k]
        used.add(i)
        for j in range(i + 1, len(bucket_keys)):
            if j in used:
                continue
            ratio = difflib.SequenceMatcher(None, k, bucket_keys[j]).ratio()
            if ratio >= threshold:
                group.append(bucket_keys[j])
                used.add(j)
        merged.append(group)

    # Step 3: build cluster output
    clusters: list[dict] = []
    for keys in merged:
        members: list[tuple[str, int]] = []
        for k in keys:
            members.extend(norm_buckets[k])
        members.sort(key=lambda m: -m[1])
        canonical = members[0][0]
        total = sum(c for _, c in members)
        # Only report meaningful clusters: more than one form, or capitalization variants
        if len(members) >= 2:
            clusters.append({
                "canonical": canonical,
                "members": [{"value": v, "count": c} for v, c in members],
                "total": total,
            })

    clusters.sort(key=lambda c: -c["total"])
    return clusters


def apply_mapping(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Apply a {old_value: canonical_value} mapping to one column."""
    df = df.copy()
    mapping: dict[str, str] = (params or {}).get("mapping") or {}
    if not mapping:
        return df, "", "label_normalize requires params.mapping (old → canonical)."
    s = df[column]
    df[column] = s.replace(mapping)
    affected = int(s.isin(mapping.keys()).sum())
    code = (
        f"_map = {mapping!r}\n"
        f"df['{column}'] = df['{column}'].replace(_map)"
    )
    return df, code, f"Normalized {affected} inconsistent labels in '{column}'."


STRATEGIES = {
    "normalize": apply_mapping,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown label_norm strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
