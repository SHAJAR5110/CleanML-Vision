"""Data augmentation operations — rotation, flip, brightness, contrast, random crop.

Creates new augmented images with _aug_N suffix and appends rows to metadata DataFrame.
Uses Pillow for augmentation operations.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


def rotate(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Rotate images and create augmented copies.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'angle' (90|180|270|'random') and optional 'count' (for random)
        
    Returns:
        (new_df, code, message) with augmented images appended
    """
    angle = params["angle"]
    count = int(params.get("count", 1)) if angle == "random" else 1
    
    if angle not in (90, 180, 270, "random"):
        return df, "", f"Invalid angle: {angle}. Use 90, 180, 270, or 'random'."
    
    new_rows = []
    processed = 0
    failed = 0
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        
        for i in range(count):
            try:
                with Image.open(img_path) as img:
                    # Determine rotation angle
                    if angle == "random":
                        rot_angle = random.choice([90, 180, 270])
                    else:
                        rot_angle = angle
                    
                    # Rotate image
                    rotated = img.rotate(-rot_angle, expand=True)  # PIL rotates counter-clockwise
                    
                    # Generate new filename
                    orig_path = Path(row["path"])
                    stem = orig_path.stem
                    suffix = orig_path.suffix
                    new_name = f"{stem}_aug_rot{rot_angle}_{i}{suffix}"
                    new_path = orig_path.parent / new_name
                    new_full_path = session_dir / new_path
                    
                    # Save augmented image
                    rotated.save(new_full_path, format=img.format or "PNG")
                    
                    # Create new metadata row
                    new_row = row.copy()
                    new_row["image_id"] = f"{row['image_id']}_aug_rot{rot_angle}_{i}"
                    new_row["path"] = str(new_path)
                    new_row["width"] = rotated.width
                    new_row["height"] = rotated.height
                    new_row["file_size_kb"] = round(new_full_path.stat().st_size / 1024, 2)
                    
                    new_rows.append(new_row)
                    processed += 1
            except Exception:
                failed += 1
    
    # Append new rows to DataFrame
    if new_rows:
        df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_aug = df
    
    code = f"""from PIL import Image
import random

def rotate_augment(img_path, angle={angle}, count={count}):
    augmented = []
    with Image.open(img_path) as img:
        for i in range({count}):
            rot_angle = random.choice([90, 180, 270]) if angle == 'random' else {angle}
            rotated = img.rotate(-rot_angle, expand=True)
            new_path = img_path.parent / f"{{img_path.stem}}_aug_rot{{rot_angle}}_{{i}}{{img_path.suffix}}"
            rotated.save(new_path, format=img.format or 'PNG')
            augmented.append(new_path)
    return augmented

# Augment all images
for img_path in df['path']:
    rotate_augment(img_path, {angle}, {count})
"""
    
    message = f"Created {processed} rotated augmentations"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df_aug, code, message


def flip(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Flip images and create augmented copies.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'direction' ('horizontal'|'vertical'|'both')
        
    Returns:
        (new_df, code, message) with augmented images appended
    """
    direction = params["direction"]
    
    if direction not in ("horizontal", "vertical", "both"):
        return df, "", f"Invalid direction: {direction}. Use 'horizontal', 'vertical', or 'both'."
    
    new_rows = []
    processed = 0
    failed = 0
    
    directions = []
    if direction == "both":
        directions = ["horizontal", "vertical"]
    else:
        directions = [direction]
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        
        for flip_dir in directions:
            try:
                with Image.open(img_path) as img:
                    # Flip image
                    if flip_dir == "horizontal":
                        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                        suffix = "hflip"
                    else:  # vertical
                        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
                        suffix = "vflip"
                    
                    # Generate new filename
                    orig_path = Path(row["path"])
                    stem = orig_path.stem
                    ext = orig_path.suffix
                    new_name = f"{stem}_aug_{suffix}{ext}"
                    new_path = orig_path.parent / new_name
                    new_full_path = session_dir / new_path
                    
                    # Save augmented image
                    flipped.save(new_full_path, format=img.format or "PNG")
                    
                    # Create new metadata row
                    new_row = row.copy()
                    new_row["image_id"] = f"{row['image_id']}_aug_{suffix}"
                    new_row["path"] = str(new_path)
                    new_row["file_size_kb"] = round(new_full_path.stat().st_size / 1024, 2)
                    
                    new_rows.append(new_row)
                    processed += 1
            except Exception:
                failed += 1
    
    # Append new rows to DataFrame
    if new_rows:
        df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_aug = df
    
    code = f"""from PIL import Image

def flip_augment(img_path, direction='{direction}'):
    augmented = []
    directions = ['horizontal', 'vertical'] if direction == 'both' else [direction]
    
    with Image.open(img_path) as img:
        for flip_dir in directions:
            if flip_dir == 'horizontal':
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                suffix = 'hflip'
            else:
                flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
                suffix = 'vflip'
            
            new_path = img_path.parent / f"{{img_path.stem}}_aug_{{suffix}}{{img_path.suffix}}"
            flipped.save(new_path, format=img.format or 'PNG')
            augmented.append(new_path)
    return augmented

# Augment all images
for img_path in df['path']:
    flip_augment(img_path, '{direction}')
"""
    
    message = f"Created {processed} flipped augmentations ({direction})"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df_aug, code, message


def adjust_brightness(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Adjust brightness and create augmented copies.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'factor' (0.5-2.0, where 1.0 is original)
        
    Returns:
        (new_df, code, message) with augmented images appended
    """
    factor = float(params["factor"])
    
    if not (0.1 <= factor <= 3.0):
        return df, "", f"Invalid brightness factor: {factor}. Use range 0.1-3.0."
    
    new_rows = []
    processed = 0
    failed = 0
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        
        try:
            with Image.open(img_path) as img:
                # Adjust brightness
                enhancer = ImageEnhance.Brightness(img)
                brightened = enhancer.enhance(factor)
                
                # Generate new filename
                orig_path = Path(row["path"])
                stem = orig_path.stem
                ext = orig_path.suffix
                new_name = f"{stem}_aug_bright{factor:.1f}{ext}"
                new_path = orig_path.parent / new_name
                new_full_path = session_dir / new_path
                
                # Save augmented image
                brightened.save(new_full_path, format=img.format or "PNG")
                
                # Create new metadata row
                new_row = row.copy()
                new_row["image_id"] = f"{row['image_id']}_aug_bright{factor:.1f}"
                new_row["path"] = str(new_path)
                new_row["file_size_kb"] = round(new_full_path.stat().st_size / 1024, 2)
                
                new_rows.append(new_row)
                processed += 1
        except Exception:
            failed += 1
    
    # Append new rows to DataFrame
    if new_rows:
        df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_aug = df
    
    code = f"""from PIL import Image, ImageEnhance

def brightness_augment(img_path, factor={factor}):
    with Image.open(img_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        brightened = enhancer.enhance({factor})
        new_path = img_path.parent / f"{{img_path.stem}}_aug_bright{factor:.1f}{{img_path.suffix}}"
        brightened.save(new_path, format=img.format or 'PNG')
        return new_path

# Augment all images
for img_path in df['path']:
    brightness_augment(img_path, {factor})
"""
    
    message = f"Created {processed} brightness-adjusted augmentations (factor={factor})"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df_aug, code, message


def adjust_contrast(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Adjust contrast and create augmented copies.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'factor' (0.5-2.0, where 1.0 is original)
        
    Returns:
        (new_df, code, message) with augmented images appended
    """
    factor = float(params["factor"])
    
    if not (0.1 <= factor <= 3.0):
        return df, "", f"Invalid contrast factor: {factor}. Use range 0.1-3.0."
    
    new_rows = []
    processed = 0
    failed = 0
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        
        try:
            with Image.open(img_path) as img:
                # Adjust contrast
                enhancer = ImageEnhance.Contrast(img)
                contrasted = enhancer.enhance(factor)
                
                # Generate new filename
                orig_path = Path(row["path"])
                stem = orig_path.stem
                ext = orig_path.suffix
                new_name = f"{stem}_aug_contrast{factor:.1f}{ext}"
                new_path = orig_path.parent / new_name
                new_full_path = session_dir / new_path
                
                # Save augmented image
                contrasted.save(new_full_path, format=img.format or "PNG")
                
                # Create new metadata row
                new_row = row.copy()
                new_row["image_id"] = f"{row['image_id']}_aug_contrast{factor:.1f}"
                new_row["path"] = str(new_path)
                new_row["file_size_kb"] = round(new_full_path.stat().st_size / 1024, 2)
                
                new_rows.append(new_row)
                processed += 1
        except Exception:
            failed += 1
    
    # Append new rows to DataFrame
    if new_rows:
        df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_aug = df
    
    code = f"""from PIL import Image, ImageEnhance

def contrast_augment(img_path, factor={factor}):
    with Image.open(img_path) as img:
        enhancer = ImageEnhance.Contrast(img)
        contrasted = enhancer.enhance({factor})
        new_path = img_path.parent / f"{{img_path.stem}}_aug_contrast{factor:.1f}{{img_path.suffix}}"
        contrasted.save(new_path, format=img.format or 'PNG')
        return new_path

# Augment all images
for img_path in df['path']:
    contrast_augment(img_path, {factor})
"""
    
    message = f"Created {processed} contrast-adjusted augmentations (factor={factor})"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df_aug, code, message


def random_crop(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Create random crops of images.
    
    Args:
        df: Metadata DataFrame
        session_dir: Session directory containing images/
        params: Dict with 'width', 'height', 'count' (number of crops per image)
        
    Returns:
        (new_df, code, message) with augmented images appended
    """
    width = int(params["width"])
    height = int(params["height"])
    count = int(params.get("count", 1))
    
    new_rows = []
    processed = 0
    failed = 0
    
    for _, row in df.iterrows():
        img_path = session_dir / row["path"]
        
        for i in range(count):
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    
                    # Skip if image is smaller than crop size
                    if img_width < width or img_height < height:
                        continue
                    
                    # Random crop position
                    left = random.randint(0, img_width - width)
                    top = random.randint(0, img_height - height)
                    right = left + width
                    bottom = top + height
                    
                    cropped = img.crop((left, top, right, bottom))
                    
                    # Generate new filename
                    orig_path = Path(row["path"])
                    stem = orig_path.stem
                    ext = orig_path.suffix
                    new_name = f"{stem}_aug_rcrop_{i}{ext}"
                    new_path = orig_path.parent / new_name
                    new_full_path = session_dir / new_path
                    
                    # Save augmented image
                    cropped.save(new_full_path, format=img.format or "PNG")
                    
                    # Create new metadata row
                    new_row = row.copy()
                    new_row["image_id"] = f"{row['image_id']}_aug_rcrop_{i}"
                    new_row["path"] = str(new_path)
                    new_row["width"] = width
                    new_row["height"] = height
                    new_row["file_size_kb"] = round(new_full_path.stat().st_size / 1024, 2)
                    
                    new_rows.append(new_row)
                    processed += 1
            except Exception:
                failed += 1
    
    # Append new rows to DataFrame
    if new_rows:
        df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_aug = df
    
    code = f"""from PIL import Image
import random

def random_crop_augment(img_path, width={width}, height={height}, count={count}):
    augmented = []
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        if img_width < {width} or img_height < {height}:
            return augmented
        
        for i in range({count}):
            left = random.randint(0, img_width - {width})
            top = random.randint(0, img_height - {height})
            cropped = img.crop((left, top, left + {width}, top + {height}))
            new_path = img_path.parent / f"{{img_path.stem}}_aug_rcrop_{{i}}{{img_path.suffix}}"
            cropped.save(new_path, format=img.format or 'PNG')
            augmented.append(new_path)
    return augmented

# Augment all images
for img_path in df['path']:
    random_crop_augment(img_path, {width}, {height}, {count})
"""
    
    message = f"Created {processed} random crop augmentations ({width}×{height}, {count} per image)"
    if failed > 0:
        message += f" · {failed} failed"
    message += "."
    
    return df_aug, code, message


STRATEGIES = {
    "rotate": rotate,
    "flip": flip,
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
    "random_crop": random_crop,
}


def apply(df: pd.DataFrame, column: str | None, strategy: str, params: dict | None = None, session_dir: Path | None = None) -> tuple[pd.DataFrame, str, str]:
    """Apply an augmentation strategy.
    
    Args:
        df: Metadata DataFrame
        column: Not used (kept for consistency with tabular API)
        strategy: Strategy name from STRATEGIES
        params: Strategy-specific parameters
        session_dir: Optional session directory path (for testing)
        
    Returns:
        (new_df, code, message)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown augmentation strategy: {strategy}")
    
    if not params:
        return df, "", f"Error: {strategy} requires parameters."
    
    # Extract session_dir from first image path
    if df.empty:
        return df, "", "No images to process."
    
    if session_dir is None:
        session_dir = Path.cwd()  # Will be overridden by caller in production
    
    return STRATEGIES[strategy](df, session_dir, params)
