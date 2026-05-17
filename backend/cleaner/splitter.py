"""Train/test split — stores train + test inside the session and registers as history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split(df: pd.DataFrame, column=None, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Stratified or random split — does NOT modify df; writes train/test pickles."""
    p = params or {}
    target = p.get("target")
    test_size = float(p.get("test_size", 0.2))
    random_state = int(p.get("random_state", 42))
    stratify_col = p.get("stratify_col") or target
    session_dir = p.get("session_dir")

    if not session_dir:
        return df, "", "splitter.split needs params.session_dir (internal)."

    stratify = None
    if stratify_col and stratify_col in df.columns:
        try:
            stratify = df[stratify_col]
            # stratify needs at least 2 of each class
            if stratify.value_counts().min() < 2:
                stratify = None
        except Exception:
            stratify = None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    Path(session_dir).joinpath("train.pkl").write_bytes(b"")  # touch
    train_df.to_pickle(Path(session_dir) / "train.pkl")
    test_df.to_pickle(Path(session_dir) / "test.pkl")

    strat_note = f" stratified on '{stratify_col}'" if stratify is not None else " (random)"
    code = (
        "from sklearn.model_selection import train_test_split\n"
        f"train_df, test_df = train_test_split(\n"
        f"    df, test_size={test_size}, random_state={random_state},\n"
        f"    stratify={'df[' + repr(stratify_col) + ']' if stratify is not None else 'None'},\n"
        f")"
    )
    msg = (
        f"Train/test split: {len(train_df)} train ({(1-test_size)*100:.0f}%) · "
        f"{len(test_df)} test ({test_size*100:.0f}%){strat_note}."
    )
    return df, code, msg


STRATEGIES = {"split": split}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown splitter strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
