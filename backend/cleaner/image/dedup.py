"""Perceptual hash-based image deduplication.

Uses imagehash library to compute perceptual hashes and detect near-duplicate images.
"""

from __future__ import annotations

from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image


def compute_hashes(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Compute perceptual hashes for all images and add to metadata.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Optional dict with 'hash_size' (default 8)
        
    Returns:
        (new_df, code, message) with 'perceptual_hash' column added
    """
    df = df.copy()
    hash_size = int((params or {}).get("hash_size", 8))
    
    hashes = []
    failed = 0
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        try:
            with Image.open(img_path) as img:
                # Compute perceptual hash (pHash)
                phash = imagehash.phash(img, hash_size=hash_size)
                hashes.append(str(phash))
        except Exception:
            hashes.append(None)
            failed += 1
    
    df["perceptual_hash"] = hashes
    
    code = f"""import imagehash
from PIL import Image

def compute_phash(img_path, hash_size={hash_size}):
    try:
        with Image.open(img_path) as img:
            return str(imagehash.phash(img, hash_size={hash_size}))
    except:
        return None

df['perceptual_hash'] = df['path'].apply(lambda p: compute_phash(p, {hash_size}))
"""
    
    computed = len(df) - failed
    message = f"Computed perceptual hashes for {computed} images"
    if failed > 0:
        message += f" ({failed} failed)"
    message += "."
    
    return df, code, message


def remove_duplicates(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Remove duplicate images based on perceptual hash similarity.
    
    Keeps the first occurrence of each duplicate group.
    
    Args:
        df: Metadata DataFrame (must have 'perceptual_hash' column)
        session_dir: Session directory containing images/
        params: Optional dict with 'threshold' (Hamming distance, default 5)
        
    Returns:
        (new_df, code, message)
    """
    if "perceptual_hash" not in df.columns:
        return df, "", "Error: perceptual_hash column not found. Run compute_hashes first."
    
    threshold = int((params or {}).get("threshold", 5))
    
    # Filter out rows with None hash
    df_valid = df[df["perceptual_hash"].notna()].copy()
    df_invalid = df[df["perceptual_hash"].isna()].copy()
    
    if df_valid.empty:
        return df, "", "No valid hashes to compare."
    
    # Convert hash strings to imagehash objects
    hash_objects = []
    for h in df_valid["perceptual_hash"]:
        try:
            hash_objects.append(imagehash.hex_to_hash(h))
        except Exception:
            hash_objects.append(None)
    
    # Find duplicates using Hamming distance
    keep_indices = []
    seen_hashes = []
    
    for idx, (i, row) in enumerate(df_valid.iterrows()):
        current_hash = hash_objects[idx]
        if current_hash is None:
            keep_indices.append(i)
            continue
        
        # Check if similar to any seen hash
        is_duplicate = False
        for seen_hash in seen_hashes:
            if seen_hash is not None:
                hamming_dist = current_hash - seen_hash
                if hamming_dist <= threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            keep_indices.append(i)
            seen_hashes.append(current_hash)
    
    # Keep only non-duplicates
    df_clean = df_valid.loc[keep_indices]
    
    # Re-add invalid hash rows (keep them)
    if not df_invalid.empty:
        df_clean = pd.concat([df_clean, df_invalid], ignore_index=True)
    
    before = len(df)
    after = len(df_clean)
    removed = before - after
    
    code = f"""import imagehash

def find_duplicates(df, threshold={threshold}):
    hashes = [imagehash.hex_to_hash(h) if pd.notna(h) else None 
              for h in df['perceptual_hash']]
    keep = []
    seen = []
    for i, h in enumerate(hashes):
        if h is None or not any((h - s) <= {threshold} for s in seen if s):
            keep.append(i)
            if h: seen.append(h)
    return df.iloc[keep].reset_index(drop=True)

df = find_duplicates(df, {threshold})
"""
    
    message = f"Removed {removed} duplicate images (Hamming distance ≤ {threshold})."
    return df_clean.reset_index(drop=True), code, message


def find_duplicate_groups(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Add 'duplicate_group' column to identify similar images without removing them.
    
    Args:
        df: Metadata DataFrame (must have 'perceptual_hash' column)
        session_dir: Session directory containing images/
        params: Optional dict with 'threshold' (Hamming distance, default 5)
        
    Returns:
        (new_df, code, message) with 'duplicate_group' column added
    """
    if "perceptual_hash" not in df.columns:
        return df, "", "Error: perceptual_hash column not found. Run compute_hashes first."
    
    df = df.copy()
    threshold = int((params or {}).get("threshold", 5))
    
    # Filter out rows with None hash
    df_valid = df[df["perceptual_hash"].notna()].copy()
    
    if df_valid.empty:
        df["duplicate_group"] = -1
        return df, "", "No valid hashes to compare."
    
    # Convert hash strings to imagehash objects
    hash_objects = []
    for h in df_valid["perceptual_hash"]:
        try:
            hash_objects.append(imagehash.hex_to_hash(h))
        except Exception:
            hash_objects.append(None)
    
    # Assign group IDs
    group_ids = [-1] * len(df_valid)
    current_group = 0
    
    for idx in range(len(df_valid)):
        if group_ids[idx] != -1:
            continue  # Already assigned
        
        current_hash = hash_objects[idx]
        if current_hash is None:
            continue
        
        # Start new group
        group_ids[idx] = current_group
        
        # Find all similar images
        for jdx in range(idx + 1, len(df_valid)):
            if group_ids[jdx] != -1:
                continue
            
            other_hash = hash_objects[jdx]
            if other_hash is not None:
                hamming_dist = current_hash - other_hash
                if hamming_dist <= threshold:
                    group_ids[jdx] = current_group
        
        current_group += 1
    
    # Add group IDs to valid rows
    df_valid["duplicate_group"] = group_ids
    
    # Merge back with invalid rows
    df_invalid = df[df["perceptual_hash"].isna()].copy()
    if not df_invalid.empty:
        df_invalid["duplicate_group"] = -1
        df = pd.concat([df_valid, df_invalid], ignore_index=True)
    else:
        df = df_valid
    
    num_groups = len([g for g in group_ids if g >= 0])
    num_duplicates = sum(1 for g in group_ids if g >= 0)
    
    code = f"""import imagehash

def assign_duplicate_groups(df, threshold={threshold}):
    hashes = [imagehash.hex_to_hash(h) if pd.notna(h) else None 
              for h in df['perceptual_hash']]
    groups = [-1] * len(df)
    group_id = 0
    for i, h1 in enumerate(hashes):
        if groups[i] != -1 or h1 is None: continue
        groups[i] = group_id
        for j in range(i+1, len(hashes)):
            if groups[j] == -1 and hashes[j] and (h1 - hashes[j]) <= {threshold}:
                groups[j] = group_id
        group_id += 1
    df['duplicate_group'] = groups
    return df

df = assign_duplicate_groups(df, {threshold})
"""
    
    message = f"Identified {num_groups} duplicate groups containing {num_duplicates} images (threshold={threshold})."
    return df, code, message


STRATEGIES = {
    "compute_hashes": compute_hashes,
    "remove_duplicates": remove_duplicates,
    "find_groups": find_duplicate_groups,
}


def apply(df: pd.DataFrame, column: str | None, strategy: str, params: dict | None = None, session_dir: Path | None = None) -> tuple[pd.DataFrame, str, str]:
    """Apply a deduplication strategy.

    Args:
        df: Metadata DataFrame
        column: Not used (kept for consistency with tabular API)
        strategy: Strategy name from STRATEGIES
        params: Strategy-specific parameters
        session_dir: Session directory containing images/ — MUST be supplied
                     by the caller. Required to locate image files for hashing.

    Returns:
        (new_df, code, message)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown dedup strategy: {strategy}")

    if df.empty:
        return df, "", "No images to process."

    if session_dir is None:
        raise ValueError("dedup.apply needs session_dir to locate image files")

    return STRATEGIES[strategy](df, Path(session_dir), params or {})
