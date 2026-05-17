"""Image quality assessment — blur detection and exposure checks.

Uses OpenCV for Laplacian variance (blur) and histogram analysis (exposure).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def detect_blur(image_path: Path, threshold: float = 100.0) -> tuple[bool, float]:
    """Detect if an image is blurry using Laplacian variance.
    
    Args:
        image_path: Path to the image file
        threshold: Variance threshold below which image is considered blurry
        
    Returns:
        (is_blurry, variance) where variance is the Laplacian variance score
    """
    try:
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0
        
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = float(laplacian.var())
        
        is_blurry = variance < threshold
        return is_blurry, variance
    except Exception:
        return False, 0.0


def check_exposure(image_path: Path) -> str:
    """Check if image is properly exposed using histogram analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        'ok' | 'underexposed' | 'overexposed'
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return "ok"
        
        # Convert to grayscale for histogram
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Calculate mean brightness
        total_pixels = gray.shape[0] * gray.shape[1]
        mean_brightness = np.sum(hist * np.arange(256)) / total_pixels
        
        # Check distribution in dark/bright regions
        dark_pixels = np.sum(hist[:85]) / total_pixels  # 0-84 range
        bright_pixels = np.sum(hist[170:]) / total_pixels  # 170-255 range
        
        # Underexposed: mean < 85 OR >60% pixels in dark region
        if mean_brightness < 85 or dark_pixels > 0.6:
            return "underexposed"
        
        # Overexposed: mean > 170 OR >60% pixels in bright region
        if mean_brightness > 170 or bright_pixels > 0.6:
            return "overexposed"
        
        return "ok"
    except Exception:
        return "ok"


def remove_blurry(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Remove blurry images from the dataset.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Optional dict with 'threshold' (default 100.0)
        
    Returns:
        (new_df, code, message)
    """
    threshold = float((params or {}).get("threshold", 100.0))
    
    # Detect blur for each image
    blur_flags = []
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        is_blurry, variance = detect_blur(img_path, threshold)
        blur_flags.append(is_blurry)
    
    # Filter out blurry images
    before = len(df)
    df_clean = df[~pd.Series(blur_flags)].reset_index(drop=True)
    removed = before - len(df_clean)
    
    code = f"""import cv2
import numpy as np

def is_blurry(img_path, threshold={threshold}):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian.var() < threshold

# Remove blurry images
df = df[~df['path'].apply(lambda p: is_blurry(p, {threshold}))].reset_index(drop=True)
"""
    
    message = f"Removed {removed} blurry images (Laplacian variance < {threshold})."
    return df_clean, code, message


def remove_corrupt(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Remove corrupt/unreadable images from the dataset.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Not used
        
    Returns:
        (new_df, code, message)
    """
    # Check integrity for each image
    valid_flags = []
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify integrity
            valid_flags.append(True)
        except Exception:
            valid_flags.append(False)
    
    # Filter out corrupt images
    before = len(df)
    df_clean = df[pd.Series(valid_flags)].reset_index(drop=True)
    removed = before - len(df_clean)
    
    code = """from PIL import Image

def is_valid(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except:
        return False

# Remove corrupt images
df = df[df['path'].apply(is_valid)].reset_index(drop=True)
"""
    
    message = f"Removed {removed} corrupt/unreadable images."
    return df_clean, code, message


def flag_low_quality(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Add quality flags to metadata without removing images.
    
    Adds columns: 'blur_score', 'is_blurry', 'exposure'
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Optional dict with 'blur_threshold' (default 100.0)
        
    Returns:
        (new_df, code, message)
    """
    df = df.copy()
    threshold = float((params or {}).get("blur_threshold", 100.0))
    
    blur_scores = []
    blur_flags = []
    exposures = []
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        
        # Blur detection
        is_blurry, variance = detect_blur(img_path, threshold)
        blur_scores.append(round(variance, 2))
        blur_flags.append(is_blurry)
        
        # Exposure check
        exposure = check_exposure(img_path)
        exposures.append(exposure)
    
    df["blur_score"] = blur_scores
    df["is_blurry"] = blur_flags
    df["exposure"] = exposures
    
    blurry_count = sum(blur_flags)
    under_count = exposures.count("underexposed")
    over_count = exposures.count("overexposed")
    
    code = f"""import cv2
import numpy as np

def compute_blur_score(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian.var()

def check_exposure(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    if mean < 85: return 'underexposed'
    if mean > 170: return 'overexposed'
    return 'ok'

df['blur_score'] = df['path'].apply(compute_blur_score)
df['is_blurry'] = df['blur_score'] < {threshold}
df['exposure'] = df['path'].apply(check_exposure)
"""
    
    message = f"Flagged quality: {blurry_count} blurry, {under_count} underexposed, {over_count} overexposed."
    return df, code, message


STRATEGIES = {
    "remove_blurry": remove_blurry,
    "remove_corrupt": remove_corrupt,
    "flag_low_quality": flag_low_quality,
}


def apply(df: pd.DataFrame, column: str | None, strategy: str, params: dict | None = None, session_dir: Path | None = None) -> tuple[pd.DataFrame, str, str]:
    """Apply a quality strategy.

    Args:
        df: Metadata DataFrame
        column: Not used (kept for consistency with tabular API)
        strategy: Strategy name from STRATEGIES
        params: Strategy-specific parameters
        session_dir: Session directory containing images/ — MUST be supplied
                     by the caller (the Flask endpoint). Without it, image
                     files cannot be located on disk.

    Returns:
        (new_df, code, message)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown quality strategy: {strategy}")

    if df.empty:
        return df, "", "No images to process."

    if session_dir is None:
        # Last-resort fallback — almost certainly wrong; surface a clear error.
        raise ValueError("quality.apply needs session_dir to locate image files")

    return STRATEGIES[strategy](df, Path(session_dir), params or {})
