"""One-click magic cleaning for image datasets.

Even on a "clean" dataset like Kaggle Cats vs Dogs, Magic Clean is opinionated
about ML preparation and ALWAYS performs:
  - compute perceptual hashes (so future loads can detect duplicates)
  - flag quality issues (no removal, just reporting)
  - resize every image to 224x224 (ImageNet standard input shape)
  - standardize every image to RGB (channels-last training-ready)

Plus conditional steps that only fire when there's something to fix:
  - remove corrupt files (if any)
  - remove near-duplicates by perceptual hash
  - remove severely blurry images (if >5%)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import dedup, profiler, quality, transforms


# Defaults aligned with common CV practice.
TARGET_SIZE = (224, 224)          # ImageNet input shape — works for almost every CNN
HASH_DEDUP_THRESHOLD = 5          # Hamming distance: ≤5 are near-duplicates
SEVERE_BLUR_THRESHOLD = 50.0      # Laplacian variance < 50 is unusable
BLUR_FLAG_THRESHOLD = 100.0       # variance < 100 → flagged (not removed)
BLURRY_REMOVE_RATIO = 0.05        # only remove blurry images if >5% are blurry


def run(df: pd.DataFrame, session_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Apply opinionated ML-preparation cleaning to an image dataset.

    Returns (cleaned_df, operations_history).
    """
    operations: list[dict] = []
    current_df = df.copy()

    # ---------- conditional cleanup ----------

    # Remove corrupt files first (only if any).
    df_after, code, message = quality.remove_corrupt(current_df, session_dir)
    if len(df_after) < len(current_df):
        operations.append({
            "op": {"family": "quality", "strategy": "remove_corrupt"},
            "code": code,
            "message": message,
        })
        current_df = df_after

    # ---------- always-on: compute perceptual hashes ----------
    df_after, code, message = dedup.compute_hashes(current_df, session_dir)
    operations.append({
        "op": {"family": "dedup", "strategy": "compute_hashes"},
        "code": code,
        "message": f"{message} — your dataset is now deduplication-ready.",
    })
    current_df = df_after

    # Remove near-duplicates (only show step if duplicates were actually removed).
    df_after, code, message = dedup.remove_duplicates(
        current_df, session_dir, {"threshold": HASH_DEDUP_THRESHOLD}
    )
    if len(df_after) < len(current_df):
        operations.append({
            "op": {
                "family": "dedup", "strategy": "remove_duplicates",
                "params": {"threshold": HASH_DEDUP_THRESHOLD},
            },
            "code": code,
            "message": message,
        })
        current_df = df_after
    else:
        operations.append({
            "op": {"family": "dedup", "strategy": "remove_duplicates",
                   "params": {"threshold": HASH_DEDUP_THRESHOLD}},
            "code": code,
            "message": (
                f"Scanned for near-duplicates (Hamming ≤ {HASH_DEDUP_THRESHOLD}); "
                "none found — dataset is already unique."
            ),
        })

    # ---------- always-on: flag quality issues ----------
    df_after, code, message = quality.flag_low_quality(
        current_df, session_dir, {"blur_threshold": BLUR_FLAG_THRESHOLD}
    )
    operations.append({
        "op": {
            "family": "quality", "strategy": "flag_low_quality",
            "params": {"blur_threshold": BLUR_FLAG_THRESHOLD},
        },
        "code": code,
        "message": message,
    })
    current_df = df_after

    # Conditional removal of severely blurry images (>5% blurry).
    if "is_blurry" in current_df.columns and len(current_df):
        blurry_ratio = current_df["is_blurry"].sum() / len(current_df)
        if blurry_ratio > BLURRY_REMOVE_RATIO:
            df_after, code, message = quality.remove_blurry(
                current_df, session_dir, {"threshold": SEVERE_BLUR_THRESHOLD}
            )
            if len(df_after) < len(current_df):
                operations.append({
                    "op": {
                        "family": "quality", "strategy": "remove_blurry",
                        "params": {"threshold": SEVERE_BLUR_THRESHOLD},
                    },
                    "code": code,
                    "message": message,
                })
                current_df = df_after

    # ---------- always-on: standardize to RGB ----------
    df_after, code, message = transforms.convert_color(
        current_df, session_dir, {"mode": "RGB"}
    )
    operations.append({
        "op": {"family": "transforms", "strategy": "convert_color",
               "params": {"mode": "RGB"}},
        "code": code,
        "message": f"{message} — channels-last, ML-ready.",
    })
    current_df = df_after

    # ---------- always-on: standardize to 224x224 ----------
    tw, th = TARGET_SIZE
    df_after, code, message = transforms.resize(
        current_df, session_dir,
        {"width": tw, "height": th, "mode": "pad"},
    )
    operations.append({
        "op": {
            "family": "transforms", "strategy": "resize",
            "params": {"width": tw, "height": th, "mode": "pad"},
        },
        "code": code,
        "message": (
            f"{message} — uniform {tw}×{th} input shape (ImageNet standard, "
            "ready for any CNN)."
        ),
    })
    current_df = df_after

    return current_df, operations


def get_summary(operations: list[dict]) -> str:
    """Generate a human-readable summary of magic cleaning operations."""
    if not operations:
        return "No cleaning operations applied."

    lines = [f"Applied {len(operations)} cleaning operations:"]
    for i, op in enumerate(operations, 1):
        lines.append(f"  {i}. {op.get('message', '')}")
    return "\n".join(lines)


def get_code(operations: list[dict]) -> str:
    """Generate combined Python code for all magic cleaning operations."""
    if not operations:
        return "# No cleaning operations applied"

    blocks = ["# CleanML Vision — magic cleaning pipeline\n"]
    for i, op in enumerate(operations, 1):
        code = op.get("code", "")
        if code:
            blocks.append(f"# Step {i}: {op.get('message', '')}")
            blocks.append(code)
            blocks.append("")
    return "\n".join(blocks)
