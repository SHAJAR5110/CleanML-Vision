"""Tests for image module Phase 2: quality and deduplication."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cleaner.image import dedup, loader, quality


@pytest.fixture
def test_images_with_duplicates(tmp_path):
    """Generate a ZIP with images including near-duplicates."""
    zip_path = tmp_path / "test_images_dup.zip"
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Image 1: RGB 64x64
        img1 = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
        )
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        zf.writestr("image1.png", buf1.getvalue())
        
        # Image 2: Exact duplicate of image1
        zf.writestr("image2.png", buf1.getvalue())
        
        # Image 3: Slightly modified version of image1 (near-duplicate)
        img3_arr = np.array(img1)
        img3_arr[0:10, 0:10] = 255  # Modify small region
        img3 = Image.fromarray(img3_arr, mode="RGB")
        buf3 = io.BytesIO()
        img3.save(buf3, format="PNG")
        zf.writestr("image3.png", buf3.getvalue())
        
        # Image 4: Completely different
        img4 = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
        )
        buf4 = io.BytesIO()
        img4.save(buf4, format="PNG")
        zf.writestr("image4.png", buf4.getvalue())
        
        # Image 5: Blurry image (low variance)
        img5_arr = np.ones((64, 64, 3), dtype=np.uint8) * 128  # Uniform gray
        img5 = Image.fromarray(img5_arr, mode="RGB")
        buf5 = io.BytesIO()
        img5.save(buf5, format="PNG")
        zf.writestr("image5_blurry.png", buf5.getvalue())
    
    return zip_path


@pytest.fixture
def session_with_images(test_images_with_duplicates, tmp_path):
    """Load test images into a session."""
    sdir = tmp_path / "session"
    sdir.mkdir()
    df, _, _ = loader.load_from_zip(test_images_with_duplicates, sdir)
    return df, sdir


def test_compute_hashes(session_with_images):
    """Test perceptual hash computation."""
    df, sdir = session_with_images
    
    df_hashed, code, message = dedup.compute_hashes(df, sdir)
    
    # Should have perceptual_hash column
    assert "perceptual_hash" in df_hashed.columns
    
    # All hashes should be computed (no failures for valid images)
    assert df_hashed["perceptual_hash"].notna().sum() == len(df_hashed)
    
    # Hashes should be strings
    assert all(isinstance(h, str) for h in df_hashed["perceptual_hash"])
    
    # Message should mention computed count
    assert "computed" in message.lower()
    assert str(len(df_hashed)) in message or "5" in message


def test_compute_hashes_with_hash_size(session_with_images):
    """Test perceptual hash computation with custom hash size."""
    df, sdir = session_with_images
    
    df_hashed, code, message = dedup.compute_hashes(df, sdir, {"hash_size": 16})
    
    # Should have perceptual_hash column
    assert "perceptual_hash" in df_hashed.columns
    
    # Hashes with larger hash_size should be longer
    assert all(len(h) > 10 for h in df_hashed["perceptual_hash"] if h)


def test_remove_duplicates(session_with_images):
    """Test duplicate removal based on perceptual hash."""
    df, sdir = session_with_images
    
    # First compute hashes
    df_hashed, _, _ = dedup.compute_hashes(df, sdir)
    
    # Then remove duplicates
    df_clean, code, message = dedup.remove_duplicates(df_hashed, sdir, {"threshold": 5})
    
    # Should have fewer images (duplicates removed)
    assert len(df_clean) < len(df_hashed)
    
    # Should have at least 2 unique images (image1/2/3 group + image4)
    assert len(df_clean) >= 2
    
    # Message should mention removed count
    assert "removed" in message.lower()


def test_remove_duplicates_without_hashes(session_with_images):
    """Test that remove_duplicates fails gracefully without hashes."""
    df, sdir = session_with_images
    
    # Try to remove duplicates without computing hashes first
    df_result, code, message = dedup.remove_duplicates(df, sdir)
    
    # Should return error message
    assert "error" in message.lower()
    assert "perceptual_hash" in message.lower()


def test_find_duplicate_groups(session_with_images):
    """Test duplicate group identification."""
    df, sdir = session_with_images
    
    # First compute hashes
    df_hashed, _, _ = dedup.compute_hashes(df, sdir)
    
    # Find duplicate groups
    df_grouped, code, message = dedup.find_duplicate_groups(df_hashed, sdir, {"threshold": 5})
    
    # Should have duplicate_group column
    assert "duplicate_group" in df_grouped.columns
    
    # Should have at least one group (image1, image2, image3 are similar)
    assert df_grouped["duplicate_group"].max() >= 0
    
    # Message should mention groups
    assert "group" in message.lower()


def test_detect_blur(session_with_images):
    """Test blur detection."""
    df, sdir = session_with_images
    
    # Test on the blurry image (image5_blurry.png)
    blurry_row = df[df["image_id"].str.contains("blurry")].iloc[0]
    blurry_path = sdir / blurry_row["path"]
    
    is_blurry, variance = quality.detect_blur(blurry_path, threshold=100.0)
    
    # Uniform gray image should have very low variance
    assert variance < 100.0
    assert is_blurry is True


def test_check_exposure(session_with_images):
    """Test exposure checking."""
    df, sdir = session_with_images
    
    # Test on a normal image
    normal_row = df[df["image_id"] == "image1"].iloc[0]
    normal_path = sdir / normal_row["path"]
    
    exposure = quality.check_exposure(normal_path)
    
    # Should return one of the valid values
    assert exposure in ("ok", "underexposed", "overexposed")


def test_remove_blurry(session_with_images):
    """Test blurry image removal."""
    df, sdir = session_with_images
    
    df_clean, code, message = quality.remove_blurry(df, sdir, {"threshold": 100.0})
    
    # Should remove at least the uniform gray image
    assert len(df_clean) < len(df)
    
    # Message should mention removed count
    assert "removed" in message.lower()


def test_remove_corrupt(session_with_images):
    """Test corrupt image removal."""
    df, sdir = session_with_images
    
    # All test images are valid, so none should be removed
    df_clean, code, message = quality.remove_corrupt(df, sdir)
    
    # Should have same number of images
    assert len(df_clean) == len(df)
    
    # Message should mention 0 removed
    assert "0" in message or "removed 0" in message.lower()


def test_flag_low_quality(session_with_images):
    """Test quality flagging without removal."""
    df, sdir = session_with_images
    
    df_flagged, code, message = quality.flag_low_quality(df, sdir, {"blur_threshold": 100.0})
    
    # Should have new quality columns
    assert "blur_score" in df_flagged.columns
    assert "is_blurry" in df_flagged.columns
    assert "exposure" in df_flagged.columns
    
    # Should have same number of images (no removal)
    assert len(df_flagged) == len(df)
    
    # At least one image should be flagged as blurry
    assert df_flagged["is_blurry"].sum() > 0
    
    # Message should mention flagged counts
    assert "flagged" in message.lower()


def test_dedup_apply_function(session_with_images):
    """Test the apply() dispatcher for dedup module."""
    df, sdir = session_with_images
    
    # Test compute_hashes through apply
    df_result, code, message = dedup.apply(df, None, "compute_hashes")
    
    assert "perceptual_hash" in df_result.columns


def test_quality_apply_function(session_with_images):
    """Test the apply() dispatcher for quality module."""
    df, sdir = session_with_images
    
    # Test flag_low_quality through apply
    df_result, code, message = quality.apply(df, None, "flag_low_quality")
    
    assert "blur_score" in df_result.columns


def test_dedup_invalid_strategy(session_with_images):
    """Test that invalid strategy raises error."""
    df, sdir = session_with_images
    
    with pytest.raises(ValueError, match="unknown dedup strategy"):
        dedup.apply(df, None, "invalid_strategy")


def test_quality_invalid_strategy(session_with_images):
    """Test that invalid strategy raises error."""
    df, sdir = session_with_images
    
    with pytest.raises(ValueError, match="unknown quality strategy"):
        quality.apply(df, None, "invalid_strategy")
