"""Smart 'what to do next' recommender.

Reads the current profile and returns a ranked list of suggestions, each
with a one-click op payload that the frontend can apply.

Each suggestion: {
  id, title, reason, impact: 'high'|'medium'|'low',
  op: {family, strategy, column?, params?}, badge: 'cleanup'|'transform'|...
}
"""

from __future__ import annotations

import pandas as pd

from .profiler import profile_dataframe

IMPACT_ORDER = {"high": 3, "medium": 2, "low": 1}


def suggest(df: pd.DataFrame, max_items: int = 8) -> list[dict]:
    p = profile_dataframe(df)
    items: list[dict] = []

    # 1. Drop duplicate rows
    if p["duplicate_rows"] > 0:
        items.append({
            "id": "drop_dupes",
            "title": f"Drop {p['duplicate_rows']} duplicate rows",
            "reason": "Duplicates can bias the model and waste compute.",
            "impact": "high",
            "badge": "cleanup",
            "op": {"family": "duplicates", "strategy": "drop_rows"},
        })

    # 2. Per-column suggestions
    for c in p["columns"]:
        name = c["name"]
        t = c["inferred_type"]
        miss = c["missing_pct"]

        if c["cardinality"] == "constant":
            items.append({
                "id": f"drop_const_{name}",
                "title": f"Drop constant column '{name}'",
                "reason": "All values are identical — no information for ML.",
                "impact": "high",
                "badge": "cleanup",
                "op": {"family": "missing", "strategy": "drop_column", "column": name},
            })
            continue

        if miss > 70:
            items.append({
                "id": f"drop_high_miss_{name}",
                "title": f"Drop '{name}' ({miss}% missing)",
                "reason": "Too much missing data to reliably impute.",
                "impact": "high",
                "badge": "cleanup",
                "op": {"family": "missing", "strategy": "drop_column", "column": name},
            })
            continue

        if t == "id_like":
            items.append({
                "id": f"drop_id_{name}",
                "title": f"Drop ID column '{name}'",
                "reason": "Unique-per-row identifier; not predictive for ML.",
                "impact": "medium",
                "badge": "cleanup",
                "op": {"family": "missing", "strategy": "drop_column", "column": name},
            })
            continue

        if c["embedded_nan_count"] > 0:
            items.append({
                "id": f"std_nan_{name}",
                "title": f"Convert '?', 'N/A' tokens to NaN in '{name}'",
                "reason": f"{c['embedded_nan_count']} cells contain placeholder strings.",
                "impact": "medium",
                "badge": "cleanup",
                "op": {"family": "missing", "strategy": "standardize_nan", "column": name},
            })

        if t == "numeric":
            if miss > 0:
                items.append({
                    "id": f"fill_med_{name}",
                    "title": f"Fill missing in '{name}' with median",
                    "reason": f"{miss}% missing; median is robust to outliers.",
                    "impact": "high" if miss > 10 else "medium",
                    "badge": "fill",
                    "op": {"family": "missing", "strategy": "median", "column": name},
                })
            if c.get("outlier_count", 0):
                ratio = c["outlier_count"] / c["count"] if c["count"] else 0
                strat = "iqr_cap" if ratio > 0.05 else "iqr_remove"
                action = "Cap" if strat == "iqr_cap" else "Remove"
                items.append({
                    "id": f"outliers_{name}",
                    "title": f"{action} {c['outlier_count']} outliers in '{name}'",
                    "reason": "IQR-based outlier treatment for cleaner training data.",
                    "impact": "medium",
                    "badge": "outliers",
                    "op": {"family": "outliers", "strategy": strat, "column": name},
                })
            if c.get("skew") is not None and abs(c["skew"]) > 2:
                items.append({
                    "id": f"log_{name}",
                    "title": f"Log-transform '{name}' (skew = {c['skew']:.2f})",
                    "reason": "Heavy skew hurts linear models; log makes it normal-ish.",
                    "impact": "medium",
                    "badge": "transform",
                    "op": {"family": "scalers", "strategy": "log", "column": name},
                })

        elif t == "categorical":
            if miss > 0:
                items.append({
                    "id": f"fill_mode_{name}",
                    "title": f"Fill missing in '{name}' with mode",
                    "reason": f"{miss}% missing in categorical column.",
                    "impact": "high" if miss > 10 else "medium",
                    "badge": "fill",
                    "op": {"family": "missing", "strategy": "mode", "column": name},
                })
            card = c["cardinality"]
            if card in ("binary", "low_card"):
                items.append({
                    "id": f"onehot_{name}",
                    "title": f"One-hot encode '{name}'",
                    "reason": f"{c['unique']} unique values — perfect for one-hot.",
                    "impact": "high",
                    "badge": "encode",
                    "op": {"family": "encoders", "strategy": "onehot", "column": name},
                })
            elif card == "medium_card":
                items.append({
                    "id": f"freq_{name}",
                    "title": f"Frequency-encode '{name}'",
                    "reason": f"{c['unique']} categories — too many for one-hot.",
                    "impact": "medium",
                    "badge": "encode",
                    "op": {"family": "encoders", "strategy": "frequency", "column": name},
                })

        elif t == "datetime":
            items.append({
                "id": f"datetime_{name}",
                "title": f"Extract year/month/weekday from '{name}'",
                "reason": "Raw datetimes can't be modeled directly; extract features.",
                "impact": "medium",
                "badge": "transform",
                "op": {"family": "datetime", "strategy": "extract", "column": name,
                       "params": {"parts": ["year", "month", "weekday"]}},
            })

        elif t == "text":
            if c["cardinality"] == "unique":
                items.append({
                    "id": f"drop_text_{name}",
                    "title": f"Drop text column '{name}'",
                    "reason": "Unique free-text column — not directly usable.",
                    "impact": "low",
                    "badge": "cleanup",
                    "op": {"family": "missing", "strategy": "drop_column", "column": name},
                })

    # 3. Rank by impact
    items.sort(key=lambda s: -IMPACT_ORDER.get(s["impact"], 0))

    return items[:max_items]
