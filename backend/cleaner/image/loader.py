"""Image dataset loader — ZIP extraction and metadata extraction.

Extracts uploaded ZIP files, validates image formats, and builds a metadata DataFrame.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def load_from_zip(zip_path: Path, session_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """Extract ZIP to session_dir/images/, walk files, detect formats.
    
    Args:
        zip_path: Path to the uploaded ZIP file
        session_dir: Session directory where images/ subdirectory will be created
        
    Returns:
        (metadata_df, code, message) where:
        - metadata_df has columns: image_id, path, format, width, height, channels, mode, file_size_kb
        - code is reproducible Python snippet
        - message is user-friendly summary
    """
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(images_dir)
    
    # Walk extracted files
    records: list[dict[str, Any]] = []
    skipped = 0
    total = 0
    labels_csv_found = False
    
    for file_path in images_dir.rglob("*"):
        if file_path.is_dir():
            continue
            
        # Check for labels.csv
        if file_path.name.lower() == "labels.csv":
            # Copy to session root for pair module
            (session_dir / "labels.csv").write_bytes(file_path.read_bytes())
            labels_csv_found = True
            continue
        
        # Check if supported image format
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
            
        total += 1
        
        try:
            with Image.open(file_path) as img:
                # Extract metadata
                width, height = img.size
                mode = img.mode  # RGB, L, RGBA, etc.
                format_name = img.format or file_path.suffix[1:].upper()
                
                # Determine channels from mode
                if mode == "L":
                    channels = 1
                elif mode == "RGB":
                    channels = 3
                elif mode == "RGBA":
                    channels = 4
                elif mode == "LA":
                    channels = 2
                else:
                    # Fallback: count bands
                    channels = len(img.getbands())
                
                # File size in KB
                file_size_kb = file_path.stat().st_size / 1024
                
                # Use filename stem as image_id (ensure uniqueness by appending parent if needed)
                image_id = file_path.stem
                # If duplicate, append parent directory name
                if any(r["image_id"] == image_id for r in records):
                    image_id = f"{file_path.parent.name}_{image_id}"
                
                # Path relative to session_dir
                rel_path = file_path.relative_to(session_dir)
                
                records.append({
                    "image_id": image_id,
                    "path": str(rel_path),
                    "format": format_name,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "mode": mode,
                    "file_size_kb": round(file_size_kb, 2),
                })
        except Exception as e:
            # Skip corrupt/unreadable files
            skipped += 1
            continue
    
    if not records:
        df = pd.DataFrame(columns=[
            "image_id", "path", "format", "width", "height", 
            "channels", "mode", "file_size_kb"
        ])
    else:
        df = pd.DataFrame(records)
    
    # Generate code snippet
    code = f"""import zipfile
from pathlib import Path
from PIL import Image
import pandas as pd

# Extract ZIP
images_dir = Path('images')
images_dir.mkdir(exist_ok=True)
with zipfile.ZipFile('{zip_path.name}', 'r') as zf:
    zf.extractall(images_dir)

# Load image metadata
records = []
for img_path in images_dir.rglob('*'):
    if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}:
        try:
            with Image.open(img_path) as img:
                records.append({{
                    'image_id': img_path.stem,
                    'path': str(img_path),
                    'format': img.format,
                    'width': img.size[0],
                    'height': img.size[1],
                    'channels': len(img.getbands()),
                    'mode': img.mode,
                    'file_size_kb': img_path.stat().st_size / 1024,
                }})
        except:
            pass
df = pd.DataFrame(records)
"""
    
    # Generate message
    loaded = len(records)
    msg_parts = [f"Loaded {loaded} images"]
    if skipped > 0:
        msg_parts.append(f"skipped {skipped} corrupt/invalid files")
    if labels_csv_found:
        msg_parts.append("found labels.csv")
    msg_parts.append(f"total {total} files scanned")
    
    message = " · ".join(msg_parts) + "."
    
    return df, code, message
