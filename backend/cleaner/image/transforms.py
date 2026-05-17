"""Core image transformations — resize, crop, color conversion, normalization.

Uses Pillow for basic transform operations with reproducible code generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def resize(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Resize all images to specified dimensions.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'width', 'height', 'mode' ('stretch'|'pad'|'crop')
        
    Returns:
        (new_df, code, message)
    """
    width = int(params["width"])
    height = int(params["height"])
    mode = params.get("mode", "stretch")
    
    if mode not in ("stretch", "pad", "crop"):
        return df, "", f"Invalid resize mode: {mode}. Use 'stretch', 'pad', or 'crop'."
    
    df = df.copy()
    processed = 0
    failed = 0
    
    for idx, row in df.iterrows():
        img_path = session_dir / row["path"]
        try:
            with Image.open(img_path) as img:
                if mode == "stretch":
                    # Simple resize (may distort aspect ratio)
                    resized = img.resize((width, height), Image.Resampling.LANCZOS)
                elif mode == "pad":
                    # Pad to maintain aspect ratio
                    resized = ImageOps.pad(img, (width, height), Image.Resampling.LANCZOS)
                elif mode == "crop":
                    # Center crop to exact dimensions
                    resized = ImageOps.fit(img, (width, height), Image.Resampling.LANCZOS)
                
                # Save back to same path
                resized.save(img_path, format=img.format or "PNG")
                
                # Update metadata
                df.at[idx, "width"] = width
                df.at[idx, "height"] = height
                df.at[idx, "file_size_kb"] = round(img_path.stat().st_size / 1024, 2)
                
                processed += 1
        except Exception:
            failed += 1
    
    code = f"""from PIL import Image, ImageOps

def resize_image(img_path, width={width}, height={height}, mode='{mode}'):
    with Image.open(img_path) as img:
        if mode == 'stretch':
            resized = img.resize(({width}, {height}), Image.Resampling.LANCZOS)
        elif mode == 'pad':
            resized = ImageOps.pad(img, ({width}, {height}), Image.Resampling.LANCZOS)
        elif mode == 'crop':
            resized = ImageOps.fit(img, ({width}, {height}), Image.Resampling.LANCZOS)
        resized.save(img_path, format=img.format or 'PNG')

# Resize all images
for img_path in df['path']:
    resize_image(img_path, {width}, {height}, '{mode}')
"""
    
    message = f"Resized {processed} images to {width}×{height} ({mode} mode)"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df, code, message


def convert_color(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Convert all images to specified color mode.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'mode' ('RGB'|'L'|'RGBA')
        
    Returns:
        (new_df, code, message)
    """
    target_mode = params["mode"]
    
    if target_mode not in ("RGB", "L", "RGBA"):
        return df, "", f"Invalid color mode: {target_mode}. Use 'RGB', 'L', or 'RGBA'."
    
    df = df.copy()
    processed = 0
    failed = 0
    
    for idx, row in df.iterrows():
        if row["mode"] == target_mode:
            continue  # Already in target mode
        
        img_path = session_dir / row["path"]
        try:
            with Image.open(img_path) as img:
                converted = img.convert(target_mode)
                converted.save(img_path, format=img.format or "PNG")
                
                # Update metadata
                df.at[idx, "mode"] = target_mode
                if target_mode == "L":
                    df.at[idx, "channels"] = 1
                elif target_mode == "RGB":
                    df.at[idx, "channels"] = 3
                elif target_mode == "RGBA":
                    df.at[idx, "channels"] = 4
                
                df.at[idx, "file_size_kb"] = round(img_path.stat().st_size / 1024, 2)
                processed += 1
        except Exception:
            failed += 1
    
    code = f"""from PIL import Image

def convert_color_mode(img_path, target_mode='{target_mode}'):
    with Image.open(img_path) as img:
        if img.mode != target_mode:
            converted = img.convert(target_mode)
            converted.save(img_path, format=img.format or 'PNG')

# Convert all images to {target_mode}
for img_path in df['path']:
    convert_color_mode(img_path, '{target_mode}')
"""
    
    message = f"Converted {processed} images to {target_mode} mode"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df, code, message


def normalize(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Normalize pixel values using specified method.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'method' ('imagenet'|'0-1'|'z-score')
        
    Returns:
        (new_df, code, message)
    """
    method = params["method"]
    
    if method not in ("imagenet", "0-1", "z-score"):
        return df, "", f"Invalid normalization method: {method}. Use 'imagenet', '0-1', or 'z-score'."
    
    processed = 0
    failed = 0
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        try:
            with Image.open(img_path) as img:
                # Convert to numpy array
                arr = np.array(img, dtype=np.float32)
                
                if method == "imagenet":
                    # ImageNet normalization: (x/255 - mean) / std
                    arr = arr / 255.0
                    if len(arr.shape) == 3 and arr.shape[2] == 3:  # RGB
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        arr = (arr - mean) / std
                elif method == "0-1":
                    # Simple 0-1 normalization
                    arr = arr / 255.0
                elif method == "z-score":
                    # Z-score normalization
                    arr = arr / 255.0
                    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
                
                # Convert back to uint8 range for saving
                if method == "0-1":
                    arr_uint8 = (arr * 255).astype(np.uint8)
                else:
                    # For imagenet/z-score, clamp and rescale to 0-255
                    arr = np.clip(arr, -3, 3)  # Clamp extreme values
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                    arr_uint8 = (arr * 255).astype(np.uint8)
                
                # Save normalized image
                normalized_img = Image.fromarray(arr_uint8, mode=img.mode)
                normalized_img.save(img_path, format=img.format or "PNG")
                
                processed += 1
        except Exception:
            failed += 1
    
    code = f"""import numpy as np
from PIL import Image

def normalize_image(img_path, method='{method}'):
    with Image.open(img_path) as img:
        arr = np.array(img, dtype=np.float32)
        
        if method == 'imagenet':
            arr = arr / 255.0
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                arr = (arr - mean) / std
        elif method == '0-1':
            arr = arr / 255.0
        elif method == 'z-score':
            arr = arr / 255.0
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        
        # Convert back to uint8 for saving
        if method == '0-1':
            arr_uint8 = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, -3, 3)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr_uint8 = (arr * 255).astype(np.uint8)
        
        normalized = Image.fromarray(arr_uint8, mode=img.mode)
        normalized.save(img_path, format=img.format or 'PNG')

# Normalize all images
for img_path in df['path']:
    normalize_image(img_path, '{method}')
"""
    
    message = f"Normalized {processed} images using {method} method"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df, code, message


def center_crop(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Center crop all images to specified dimensions.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'width', 'height'
        
    Returns:
        (new_df, code, message)
    """
    width = int(params["width"])
    height = int(params["height"])
    
    df = df.copy()
    processed = 0
    failed = 0
    
    for idx, row in df.iterrows():
        img_path = session_dir / row["path"]
        try:
            with Image.open(img_path) as img:
                # Calculate crop box for center crop
                img_width, img_height = img.size
                left = (img_width - width) // 2
                top = (img_height - height) // 2
                right = left + width
                bottom = top + height
                
                # Ensure crop box is within image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(img_width, right)
                bottom = min(img_height, bottom)
                
                cropped = img.crop((left, top, right, bottom))
                cropped.save(img_path, format=img.format or "PNG")
                
                # Update metadata
                df.at[idx, "width"] = cropped.width
                df.at[idx, "height"] = cropped.height
                df.at[idx, "file_size_kb"] = round(img_path.stat().st_size / 1024, 2)
                
                processed += 1
        except Exception:
            failed += 1
    
    code = f"""from PIL import Image

def center_crop(img_path, width={width}, height={height}):
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        left = (img_width - {width}) // 2
        top = (img_height - {height}) // 2
        right = left + {width}
        bottom = top + {height}
        
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)
        
        cropped = img.crop((left, top, right, bottom))
        cropped.save(img_path, format=img.format or 'PNG')

# Center crop all images
for img_path in df['path']:
    center_crop(img_path, {width}, {height})
"""
    
    message = f"Center cropped {processed} images to {width}×{height}"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df, code, message


STRATEGIES = {
    "resize": resize,
    "convert_color": convert_color,
    "normalize": normalize,
    "center_crop": center_crop,
}


def apply(df: pd.DataFrame, column: str | None, strategy: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Apply a transformation strategy.
    
    Args:
        df: Metadata DataFrame
        column: Not used (kept for consistency with tabular API)
        strategy: Strategy name from STRATEGIES
        params: Strategy-specific parameters
        
    Returns:
        (new_df, code, message)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown transform strategy: {strategy}")
    
    if not params:
        return df, "", f"Error: {strategy} requires parameters."
    
    # Extract session_dir from first image path
    if df.empty:
        return df, "", "No images to process."
    
    # Image paths are relative like "images/image1.png"
    # We need to find the actual session directory by going up from the path
    # This is a workaround - in production, session_dir should be passed explicitly
    first_path = Path(df.iloc[0]["path"])
    
    # Try to find session_dir by checking if images/ exists in current working directory
    import os
    cwd = Path(os.getcwd())
    
    # Check if we're already in the session directory (has images/ subdirectory)
    if (cwd / "images").exists():
        session_dir = cwd
    else:
        # Assume the path is absolute or we need to use a temp location
        # For tests, the fixture should handle this properly
        session_dir = cwd
    
    return STRATEGIES[strategy](df, session_dir, params)
