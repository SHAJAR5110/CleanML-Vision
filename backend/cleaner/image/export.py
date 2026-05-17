"""Export cleaned image datasets to various formats.

Supports ZIP archives, NumPy arrays (.npy), and PyTorch tensors (.pt).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def export_to_zip(df: pd.DataFrame, session_dir: Path, output_path: Path) -> tuple[bool, str]:
    """Export images and metadata to a ZIP archive.
    
    Args:
        df: Image metadata DataFrame
        session_dir: Session directory containing images/
        output_path: Path where ZIP file should be saved
        
    Returns:
        (success, message)
    """
    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add all images
            for _, row in df.iterrows():
                img_path = session_dir / row["path"]
                if img_path.exists():
                    # Use relative path in ZIP (e.g., "images/image1.png")
                    zf.write(img_path, row["path"])
            
            # Add metadata CSV
            metadata_csv = df.to_csv(index=False)
            zf.writestr("metadata.csv", metadata_csv)
            
            # Add labels.csv if it exists
            labels_path = session_dir / "labels.csv"
            if labels_path.exists():
                zf.write(labels_path, "labels.csv")
        
        return True, f"Exported {len(df)} images to {output_path.name}"
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def export_to_numpy(df: pd.DataFrame, session_dir: Path, output_path: Path, params: dict | None = None) -> tuple[bool, str]:
    """Export images as NumPy array (.npy file).
    
    Args:
        df: Image metadata DataFrame
        session_dir: Session directory containing images/
        output_path: Path where .npy file should be saved
        params: Optional dict with 'include_labels' (bool, default False)
        
    Returns:
        (success, message)
    """
    include_labels = (params or {}).get("include_labels", False)
    
    try:
        # Load all images into memory
        images = []
        labels = []
        failed = 0
        
        for _, row in df.iterrows():
            img_path = session_dir / row["path"]
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Convert to numpy array
                    img_array = np.array(img)
                    images.append(img_array)
                    
                    # Extract label if requested
                    if include_labels and "label" in row:
                        labels.append(row["label"])
            except Exception:
                failed += 1
        
        if not images:
            return False, "No images could be loaded."
        
        # Stack into single array
        images_array = np.stack(images, axis=0)
        
        # Save to file
        if include_labels and labels:
            # Save as dictionary with images and labels
            np.save(output_path, {"images": images_array, "labels": np.array(labels)})
            message = f"Exported {len(images)} images with labels to {output_path.name}"
        else:
            # Save just images
            np.save(output_path, images_array)
            message = f"Exported {len(images)} images to {output_path.name}"
        
        if failed > 0:
            message += f" · {failed} images failed to load"
        
        return True, message
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def export_to_pytorch(df: pd.DataFrame, session_dir: Path, output_path: Path, params: dict | None = None) -> tuple[bool, str]:
    """Export images as PyTorch tensors (.pt file).
    
    Args:
        df: Image metadata DataFrame
        session_dir: Session directory containing images/
        output_path: Path where .pt file should be saved
        params: Optional dict with 'include_labels' (bool, default False), 
                'normalize' (bool, default True)
        
    Returns:
        (success, message)
    """
    include_labels = (params or {}).get("include_labels", False)
    normalize = (params or {}).get("normalize", True)
    
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed. Install with: pip install torch"
    
    try:
        # Load all images into memory
        images = []
        labels = []
        failed = 0
        
        for _, row in df.iterrows():
            img_path = session_dir / row["path"]
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Convert to numpy array
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Normalize to [0, 1] if requested
                    if normalize:
                        img_array = img_array / 255.0
                    
                    # Convert to CHW format (PyTorch convention)
                    img_array = np.transpose(img_array, (2, 0, 1))
                    
                    images.append(img_array)
                    
                    # Extract label if requested
                    if include_labels and "label" in row:
                        labels.append(row["label"])
            except Exception:
                failed += 1
        
        if not images:
            return False, "No images could be loaded."
        
        # Stack into single array and convert to tensor
        images_array = np.stack(images, axis=0)
        images_tensor = torch.from_numpy(images_array)
        
        # Save to file
        if include_labels and labels:
            # Save as dictionary with images and labels
            labels_tensor = torch.tensor(labels)
            torch.save({"images": images_tensor, "labels": labels_tensor}, output_path)
            message = f"Exported {len(images)} images with labels to {output_path.name} (PyTorch tensors)"
        else:
            # Save just images
            torch.save(images_tensor, output_path)
            message = f"Exported {len(images)} images to {output_path.name} (PyTorch tensors)"
        
        if normalize:
            message += " · normalized to [0, 1]"
        
        if failed > 0:
            message += f" · {failed} images failed to load"
        
        return True, message
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def export_split(df: pd.DataFrame, session_dir: Path, output_dir: Path, params: dict | None = None) -> tuple[bool, str]:
    """Export train/test split as separate ZIP files.
    
    Args:
        df: Image metadata DataFrame (must have 'split' column)
        session_dir: Session directory containing images/
        output_dir: Directory where train.zip and test.zip should be saved
        params: Optional dict with 'format' ('zip'|'numpy'|'pytorch')
        
    Returns:
        (success, message)
    """
    if "split" not in df.columns:
        return False, "Error: DataFrame must have 'split' column. Use pair.split_by_label first."
    
    export_format = (params or {}).get("format", "zip")
    
    if export_format not in ("zip", "numpy", "pytorch"):
        return False, f"Invalid format: {export_format}. Use 'zip', 'numpy', or 'pytorch'."
    
    # Split dataframe
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    
    if df_train.empty or df_test.empty:
        return False, "Error: Both train and test splits must be non-empty."
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if export_format == "zip":
            # Export as ZIP files
            train_path = output_dir / "train.zip"
            test_path = output_dir / "test.zip"
            
            success_train, msg_train = export_to_zip(df_train, session_dir, train_path)
            success_test, msg_test = export_to_zip(df_test, session_dir, test_path)
            
            if success_train and success_test:
                return True, f"Exported train ({len(df_train)} images) and test ({len(df_test)} images) to {output_dir}"
            else:
                return False, f"Export failed: {msg_train} | {msg_test}"
        
        elif export_format == "numpy":
            # Export as NumPy arrays
            train_path = output_dir / "train.npy"
            test_path = output_dir / "test.npy"
            
            success_train, msg_train = export_to_numpy(df_train, session_dir, train_path, {"include_labels": True})
            success_test, msg_test = export_to_numpy(df_test, session_dir, test_path, {"include_labels": True})
            
            if success_train and success_test:
                return True, f"Exported train ({len(df_train)} images) and test ({len(df_test)} images) to {output_dir}"
            else:
                return False, f"Export failed: {msg_train} | {msg_test}"
        
        else:  # pytorch
            # Export as PyTorch tensors
            train_path = output_dir / "train.pt"
            test_path = output_dir / "test.pt"
            
            success_train, msg_train = export_to_pytorch(df_train, session_dir, train_path, {"include_labels": True})
            success_test, msg_test = export_to_pytorch(df_test, session_dir, test_path, {"include_labels": True})
            
            if success_train and success_test:
                return True, f"Exported train ({len(df_train)} images) and test ({len(df_test)} images) to {output_dir}"
            else:
                return False, f"Export failed: {msg_train} | {msg_test}"
    
    except Exception as e:
        return False, f"Export failed: {str(e)}"


STRATEGIES = {
    "zip": export_to_zip,
    "numpy": export_to_numpy,
    "pytorch": export_to_pytorch,
    "split": export_split,
}


def apply(df: pd.DataFrame, output_path: Path, strategy: str, params: dict | None = None) -> tuple[bool, str]:
    """Apply an export strategy.
    
    Args:
        df: Metadata DataFrame
        output_path: Path where output should be saved
        strategy: Strategy name from STRATEGIES
        params: Strategy-specific parameters
        
    Returns:
        (success, message)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown export strategy: {strategy}")
    
    # Extract session_dir from first image path
    if df.empty:
        return False, "No images to export."
    
    first_path = Path(df.iloc[0]["path"])
    session_dir = Path.cwd()  # Will be overridden by caller
    
    if strategy == "split":
        # For split export, output_path is a directory
        return STRATEGIES[strategy](df, session_dir, output_path, params or {})
    else:
        return STRATEGIES[strategy](df, session_dir, output_path, params or {})
