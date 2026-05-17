"""Tests for image module Phase 1: loader and profiler."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cleaner.image import loader, profiler


@pytest.fixture
def synthetic_image_zip(tmp_path):
    """Generate a ZIP with 5 synthetic test images + 1 corrupt file."""
    zip_path = tmp_path / "test_images.zip"
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Create 5 valid images with varying properties
        # Image 1: RGB 64x64
        img1 = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
        )
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        zf.writestr("image1.png", buf1.getvalue())
        
        # Image 2: RGB 128x128
        img2 = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8), mode="RGB"
        )
        buf2 = io.BytesIO()
        img2.save(buf2, format="JPEG")
        zf.writestr("image2.jpg", buf2.getvalue())
        
        # Image 3: Grayscale 64x64
        img3 = Image.fromarray(
            np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L"
        )
        buf3 = io.BytesIO()
        img3.save(buf3, format="PNG")
        zf.writestr("image3.png", buf3.getvalue())
        
        # Image 4: RGB 256x128 (different aspect ratio)
        img4 = Image.fromarray(
            np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8), mode="RGB"
        )
        buf4 = io.BytesIO()
        img4.save(buf4, format="PNG")
        zf.writestr("subdir/image4.png", buf4.getvalue())
        
        # Image 5: RGBA 64x64
        img5 = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8), mode="RGBA"
        )
        buf5 = io.BytesIO()
        img5.save(buf5, format="PNG")
        zf.writestr("image5.png", buf5.getvalue())
        
        # Corrupt file (not an image)
        zf.writestr("corrupt.png", b"This is not a valid image file")
        
        # Add a labels.csv
        zf.writestr("labels.csv", "filename,label\nimage1.png,cat\nimage2.jpg,dog\n")
    
    return zip_path


@pytest.fixture
def session_dir(tmp_path):
    """Create a temporary session directory."""
    sdir = tmp_path / "session_test"
    sdir.mkdir()
    return sdir


def test_load_from_zip_valid(synthetic_image_zip, session_dir):
    """Test loading a valid ZIP with multiple images."""
    df, code, message = loader.load_from_zip(synthetic_image_zip, session_dir)
    
    # Should load 5 valid images (skip 1 corrupt)
    assert len(df) == 5
    assert "image_id" in df.columns
    assert "path" in df.columns
    assert "format" in df.columns
    assert "width" in df.columns
    assert "height" in df.columns
    assert "channels" in df.columns
    assert "mode" in df.columns
    assert "file_size_kb" in df.columns
    
    # Check that images directory was created
    assert (session_dir / "images").exists()
    
    # Check that labels.csv was extracted
    assert (session_dir / "labels.csv").exists()
    
    # Verify message mentions loaded count and skipped count
    assert "5 images" in message.lower() or "loaded 5" in message.lower()
    assert "skipped 1" in message.lower() or "1 corrupt" in message.lower()
    assert "labels.csv" in message.lower()
    
    # Verify code is a string
    assert isinstance(code, str)
    assert len(code) > 0


def test_load_from_zip_dimensions(synthetic_image_zip, session_dir):
    """Test that dimensions are correctly extracted."""
    df, _, _ = loader.load_from_zip(synthetic_image_zip, session_dir)
    
    # Check specific dimensions
    img1 = df[df["image_id"] == "image1"]
    assert len(img1) == 1
    assert img1.iloc[0]["width"] == 64
    assert img1.iloc[0]["height"] == 64
    assert img1.iloc[0]["channels"] == 3
    assert img1.iloc[0]["mode"] == "RGB"
    
    img3 = df[df["image_id"] == "image3"]
    assert len(img3) == 1
    assert img3.iloc[0]["channels"] == 1
    assert img3.iloc[0]["mode"] == "L"
    
    img5 = df[df["image_id"] == "image5"]
    assert len(img5) == 1
    assert img5.iloc[0]["channels"] == 4
    assert img5.iloc[0]["mode"] == "RGBA"


def test_load_from_zip_skips_corrupt(synthetic_image_zip, session_dir):
    """Test that corrupt files are gracefully skipped."""
    df, code, message = loader.load_from_zip(synthetic_image_zip, session_dir)
    
    # Should not include corrupt.png
    assert "corrupt" not in df["image_id"].values
    
    # Message should mention skipped files
    assert "skipped" in message.lower() or "corrupt" in message.lower()


def test_profile_images_basic(synthetic_image_zip, session_dir):
    """Test basic profiling of image dataset."""
    df, _, _ = loader.load_from_zip(synthetic_image_zip, session_dir)
    profile = profiler.profile_images(df, session_dir)
    
    # Check required keys
    assert "total_images" in profile
    assert "formats" in profile
    assert "avg_width" in profile
    assert "avg_height" in profile
    assert "avg_channels" in profile
    assert "total_size_mb" in profile
    assert "dimension_stats" in profile
    assert "aspect_ratio_stats" in profile
    assert "mode_distribution" in profile
    assert "duplicates_pending" in profile
    assert "quality_score" in profile
    assert "grade" in profile
    assert "warnings" in profile
    assert "images" in profile
    
    # Check values
    assert profile["total_images"] == 5
    assert profile["quality_score"] >= 0
    assert profile["quality_score"] <= 100
    assert profile["grade"] in ["A", "B", "C", "D", "F"]
    assert isinstance(profile["warnings"], list)
    assert isinstance(profile["images"], list)
    assert len(profile["images"]) == 5


def test_profile_images_quality_score(synthetic_image_zip, session_dir):
    """Test that quality score is computed correctly."""
    df, _, _ = loader.load_from_zip(synthetic_image_zip, session_dir)
    profile = profiler.profile_images(df, session_dir)
    
    # Should have warnings due to mixed dimensions and mixed color modes
    assert len(profile["warnings"]) > 0
    assert "mixed_dimensions" in profile["warnings"] or "mixed_color_modes" in profile["warnings"]
    
    # Quality score should be penalized (not 100)
    assert profile["quality_score"] < 100
    
    # But should still be reasonable (not F grade for this dataset)
    assert profile["quality_score"] >= 40


def test_profile_images_dimension_stats(synthetic_image_zip, session_dir):
    """Test dimension statistics computation."""
    df, _, _ = loader.load_from_zip(synthetic_image_zip, session_dir)
    profile = profiler.profile_images(df, session_dir)
    
    dim_stats = profile["dimension_stats"]
    assert "width" in dim_stats
    assert "height" in dim_stats
    
    # Check width stats
    assert dim_stats["width"]["min"] == 64
    assert dim_stats["width"]["max"] == 256
    assert "median" in dim_stats["width"]
    assert "p95" in dim_stats["width"]
    assert "std" in dim_stats["width"]
    
    # Check height stats
    assert dim_stats["height"]["min"] == 64
    assert dim_stats["height"]["max"] == 128
    assert "median" in dim_stats["height"]


def test_profile_images_mode_distribution(synthetic_image_zip, session_dir):
    """Test color mode distribution."""
    df, _, _ = loader.load_from_zip(synthetic_image_zip, session_dir)
    profile = profiler.profile_images(df, session_dir)
    
    mode_dist = profile["mode_distribution"]
    assert "RGB" in mode_dist
    assert "L" in mode_dist
    assert "RGBA" in mode_dist
    
    # Should have 3 RGB, 1 L, 1 RGBA
    assert mode_dist["RGB"] == 3
    assert mode_dist["L"] == 1
    assert mode_dist["RGBA"] == 1


def test_profile_images_empty_dataframe(session_dir):
    """Test profiling an empty DataFrame."""
    import pandas as pd
    df = pd.DataFrame(columns=[
        "image_id", "path", "format", "width", "height", 
        "channels", "mode", "file_size_kb"
    ])
    
    profile = profiler.profile_images(df, session_dir)
    
    assert profile["total_images"] == 0
    assert profile["quality_score"] == 0
    assert profile["grade"] == "F"
    assert "No images loaded" in profile["warnings"]


def test_profile_images_per_image_metadata(synthetic_image_zip, session_dir):
    """Test per-image metadata in profile."""
    df, _, _ = loader.load_from_zip(synthetic_image_zip, session_dir)
    profile = profiler.profile_images(df, session_dir)
    
    images = profile["images"]
    assert len(images) == 5
    
    # Check first image structure
    img = images[0]
    assert "image_id" in img
    assert "width" in img
    assert "height" in img
    assert "channels" in img
    assert "format" in img
    assert "mode" in img
    assert "file_size_kb" in img
    assert "integrity" in img
    assert "warnings" in img
    
    # All should have integrity "ok" (corrupt files were skipped)
    assert all(img["integrity"] == "ok" for img in images)
