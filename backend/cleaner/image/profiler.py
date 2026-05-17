"""Image dataset profiler — quality scoring and statistics.

Analyzes image metadata and computes quality scores, warnings, and statistics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def profile_images(df: pd.DataFrame, session_dir: Path) -> dict[str, Any]:
    """Profile image dataset and compute quality metrics.
    
    Args:
        df: Metadata DataFrame from loader.py
        session_dir: Session directory (for future use with actual image analysis)
        
    Returns:
        Profile dict with stats, quality score, grade, warnings, and per-image metadata
    """
    if df.empty:
        return {
            "total_images": 0,
            "formats": {},
            "avg_width": 0,
            "avg_height": 0,
            "avg_channels": 0,
            "total_size_mb": 0,
            "dimension_stats": {},
            "aspect_ratio_stats": {},
            "mode_distribution": {},
            "duplicates_pending": True,
            "quality_score": 0,
            "grade": "F",
            "warnings": ["No images loaded"],
            "images": [],
        }
    
    # Basic stats
    total_images = len(df)
    formats = dict(Counter(df["format"]))
    avg_width = float(df["width"].mean())
    avg_height = float(df["height"].mean())
    avg_channels = float(df["channels"].mean())
    total_size_mb = float(df["file_size_kb"].sum() / 1024)
    
    # Dimension statistics
    dimension_stats = {
        "width": {
            "min": int(df["width"].min()),
            "max": int(df["width"].max()),
            "median": int(df["width"].median()),
            "p95": int(df["width"].quantile(0.95)),
            "std": float(df["width"].std()),
        },
        "height": {
            "min": int(df["height"].min()),
            "max": int(df["height"].max()),
            "median": int(df["height"].median()),
            "p95": int(df["height"].quantile(0.95)),
            "std": float(df["height"].std()),
        },
    }
    
    # Aspect ratio statistics
    aspect_ratios = df["width"] / df["height"]
    aspect_ratio_stats = {
        "min": float(aspect_ratios.min()),
        "max": float(aspect_ratios.max()),
        "median": float(aspect_ratios.median()),
        "p95": float(aspect_ratios.quantile(0.95)),
        "std": float(aspect_ratios.std()),
    }
    
    # Mode distribution
    mode_distribution = dict(Counter(df["mode"]))
    
    # Check if perceptual_hash column exists
    duplicates_pending = "perceptual_hash" not in df.columns
    
    # Compute warnings
    warnings = []
    
    # Check for mixed dimensions (>20% variation)
    width_cv = dimension_stats["width"]["std"] / avg_width if avg_width > 0 else 0
    height_cv = dimension_stats["height"]["std"] / avg_height if avg_height > 0 else 0
    if width_cv > 0.2 or height_cv > 0.2:
        warnings.append("mixed_dimensions")
    
    # Check for mixed color modes
    if len(mode_distribution) > 1:
        warnings.append("mixed_color_modes")
    
    # Check for extreme aspect ratios (< 0.5 or > 2.0)
    extreme_ar = ((aspect_ratios < 0.5) | (aspect_ratios > 2.0)).sum()
    if extreme_ar > 0:
        warnings.append(f"extreme_aspect_ratios ({extreme_ar} images)")
    
    # Check for very small images (< 32x32)
    tiny = ((df["width"] < 32) | (df["height"] < 32)).sum()
    if tiny > 0:
        warnings.append(f"very_small_images ({tiny} images)")
    
    # Check for very large images (> 4096 on any dimension)
    huge = ((df["width"] > 4096) | (df["height"] > 4096)).sum()
    if huge > 0:
        warnings.append(f"very_large_images ({huge} images)")
    
    # Compute quality score (0-100)
    score = 100.0
    
    # Penalize mixed dimensions (>20% variation)
    if width_cv > 0.2 or height_cv > 0.2:
        score -= 10
    
    # Penalize mixed color modes
    if len(mode_distribution) > 1:
        score -= 10
    
    # Penalize per warning beyond the first
    if len(warnings) > 1:
        score -= 5 * (len(warnings) - 1)
    
    # Note: We'll penalize corrupt files in loader (they're already skipped)
    # This is reflected in the message but not in the DataFrame
    
    # Clamp score
    score = max(0, min(100, score))
    quality_score = int(round(score))
    
    # Assign grade
    if quality_score >= 90:
        grade = "A"
    elif quality_score >= 75:
        grade = "B"
    elif quality_score >= 60:
        grade = "C"
    elif quality_score >= 40:
        grade = "D"
    else:
        grade = "F"
    
    # Per-image metadata (simplified for now; quality.py and dedup.py will enrich)
    images = []
    for _, row in df.iterrows():
        img_dict = {
            "image_id": row["image_id"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "channels": int(row["channels"]),
            "format": row["format"],
            "mode": row["mode"],
            "file_size_kb": float(row["file_size_kb"]),
            "integrity": "ok",  # Assume ok since loader skipped corrupt files
            "warnings": [],
        }
        
        # Add per-image warnings
        ar = row["width"] / row["height"]
        if ar < 0.5 or ar > 2.0:
            img_dict["warnings"].append("extreme_aspect_ratio")
        if row["width"] < 32 or row["height"] < 32:
            img_dict["warnings"].append("very_small")
        if row["width"] > 4096 or row["height"] > 4096:
            img_dict["warnings"].append("very_large")
        
        images.append(img_dict)
    
    return {
        "total_images": total_images,
        "formats": formats,
        "avg_width": round(avg_width, 1),
        "avg_height": round(avg_height, 1),
        "avg_channels": round(avg_channels, 2),
        "total_size_mb": round(total_size_mb, 2),
        "dimension_stats": dimension_stats,
        "aspect_ratio_stats": aspect_ratio_stats,
        "mode_distribution": mode_distribution,
        "duplicates_pending": duplicates_pending,
        "quality_score": quality_score,
        "grade": grade,
        "warnings": warnings,
        "images": images,
    }
