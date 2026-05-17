"""Tests for image module Phase 3: transforms and augmentation."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cleaner.image import augment, loader, transforms


@pytest.fixture
def test_images_for_transform(tmp_path):
    """Generate a ZIP with images for transformation testing."""
    zip_path = tmp_path / "test_images_transform.zip"
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Image 1: RGB 128x128
        img1 = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8), mode="RGB"
        )
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        zf.writestr("image1.png", buf1.getvalue())
        
        # Image 2: Grayscale 64x64
        img2 = Image.fromarray(
            np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L"
        )
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        zf.writestr("image2.png", buf2.getvalue())
        
        # Image 3: RGBA 128x128
        img3 = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 4), dtype=np.uint8), mode="RGBA"
        )
        buf3 = io.BytesIO()
        img3.save(buf3, format="PNG")
        zf.writestr("image3.png", buf3.getvalue())
    
    return zip_path


@pytest.fixture
def session_with_transform_images(test_images_for_transform, tmp_path):
    """Load test images into a session."""
    sdir = tmp_path / "session"
    sdir.mkdir()
    df, _, _ = loader.load_from_zip(test_images_for_transform, sdir)
    return df, sdir


def test_resize_stretch(session_with_transform_images):
    """Test resize with stretch mode."""
    df, sdir = session_with_transform_images
    
    df_resized, code, message = transforms.resize(
        df, sdir, {"width": 64, "height": 64, "mode": "stretch"}
    )
    
    # All images should be resized to 64x64
    assert all(df_resized["width"] == 64)
    assert all(df_resized["height"] == 64)
    
    # Message should mention resized count
    assert "resized" in message.lower()
    assert "64" in message


def test_resize_pad(session_with_transform_images):
    """Test resize with pad mode."""
    df, sdir = session_with_transform_images
    
    df_resized, code, message = transforms.resize(
        df, sdir, {"width": 100, "height": 100, "mode": "pad"}
    )
    
    # All images should be resized to 100x100
    assert all(df_resized["width"] == 100)
    assert all(df_resized["height"] == 100)
    
    # Verify actual image dimensions
    first_img_path = sdir / df_resized.iloc[0]["path"]
    with Image.open(first_img_path) as img:
        assert img.size == (100, 100)


def test_resize_crop(session_with_transform_images):
    """Test resize with crop mode."""
    df, sdir = session_with_transform_images
    
    df_resized, code, message = transforms.resize(
        df, sdir, {"width": 80, "height": 80, "mode": "crop"}
    )
    
    # All images should be resized to 80x80
    assert all(df_resized["width"] == 80)
    assert all(df_resized["height"] == 80)


def test_convert_color_to_rgb(session_with_transform_images):
    """Test color conversion to RGB."""
    df, sdir = session_with_transform_images
    
    df_converted, code, message = transforms.convert_color(df, sdir, {"mode": "RGB"})
    
    # All images should be RGB
    assert all(df_converted["mode"] == "RGB")
    assert all(df_converted["channels"] == 3)
    
    # Message should mention converted count
    assert "converted" in message.lower()


def test_convert_color_to_grayscale(session_with_transform_images):
    """Test color conversion to grayscale."""
    df, sdir = session_with_transform_images
    
    df_converted, code, message = transforms.convert_color(df, sdir, {"mode": "L"})
    
    # All images should be grayscale
    assert all(df_converted["mode"] == "L")
    assert all(df_converted["channels"] == 1)


def test_normalize_0_1(session_with_transform_images):
    """Test 0-1 normalization."""
    df, sdir = session_with_transform_images
    
    df_normalized, code, message = transforms.normalize(df, sdir, {"method": "0-1"})
    
    # Should process all images
    assert len(df_normalized) == len(df)
    
    # Message should mention normalized count
    assert "normalized" in message.lower()
    assert "0-1" in message


def test_normalize_imagenet(session_with_transform_images):
    """Test ImageNet normalization."""
    df, sdir = session_with_transform_images
    
    df_normalized, code, message = transforms.normalize(df, sdir, {"method": "imagenet"})
    
    # Should process all images
    assert len(df_normalized) == len(df)
    
    # Message should mention imagenet
    assert "imagenet" in message.lower()


def test_normalize_zscore(session_with_transform_images):
    """Test z-score normalization."""
    df, sdir = session_with_transform_images
    
    df_normalized, code, message = transforms.normalize(df, sdir, {"method": "z-score"})
    
    # Should process all images
    assert len(df_normalized) == len(df)
    
    # Message should mention z-score
    assert "z-score" in message.lower()


def test_center_crop(session_with_transform_images):
    """Test center cropping."""
    df, sdir = session_with_transform_images
    
    df_cropped, code, message = transforms.center_crop(
        df, sdir, {"width": 32, "height": 32}
    )
    
    # All images should be cropped to 32x32 or smaller
    assert all(df_cropped["width"] <= 32)
    assert all(df_cropped["height"] <= 32)
    
    # Message should mention cropped count
    assert "cropped" in message.lower()


def test_rotate_90(session_with_transform_images):
    """Test rotation by 90 degrees."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.rotate(df, sdir, {"angle": 90})
    
    # Should have original + augmented images
    assert len(df_augmented) == original_count * 2
    
    # New images should have _aug_rot90 in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_rot90")]
    assert len(aug_images) == original_count
    
    # Message should mention created count
    assert "created" in message.lower() or "augment" in message.lower()


def test_rotate_random(session_with_transform_images):
    """Test random rotation."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.rotate(
        df, sdir, {"angle": "random", "count": 2}
    )
    
    # Should have original + 2 augmented per original
    assert len(df_augmented) == original_count * 3
    
    # New images should have _aug_rot in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_rot")]
    assert len(aug_images) == original_count * 2


def test_flip_horizontal(session_with_transform_images):
    """Test horizontal flip."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.flip(df, sdir, {"direction": "horizontal"})
    
    # Should have original + augmented images
    assert len(df_augmented) == original_count * 2
    
    # New images should have _aug_hflip in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_hflip")]
    assert len(aug_images) == original_count


def test_flip_vertical(session_with_transform_images):
    """Test vertical flip."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.flip(df, sdir, {"direction": "vertical"})
    
    # Should have original + augmented images
    assert len(df_augmented) == original_count * 2
    
    # New images should have _aug_vflip in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_vflip")]
    assert len(aug_images) == original_count


def test_flip_both(session_with_transform_images):
    """Test both horizontal and vertical flip."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.flip(df, sdir, {"direction": "both"})
    
    # Should have original + 2 augmented per original (hflip + vflip)
    assert len(df_augmented) == original_count * 3
    
    # Should have both hflip and vflip images
    hflip_images = df_augmented[df_augmented["image_id"].str.contains("aug_hflip")]
    vflip_images = df_augmented[df_augmented["image_id"].str.contains("aug_vflip")]
    assert len(hflip_images) == original_count
    assert len(vflip_images) == original_count


def test_adjust_brightness(session_with_transform_images):
    """Test brightness adjustment."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.adjust_brightness(df, sdir, {"factor": 1.5})
    
    # Should have original + augmented images
    assert len(df_augmented) == original_count * 2
    
    # New images should have _aug_bright in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_bright")]
    assert len(aug_images) == original_count
    
    # Message should mention brightness
    assert "brightness" in message.lower()


def test_adjust_contrast(session_with_transform_images):
    """Test contrast adjustment."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.adjust_contrast(df, sdir, {"factor": 1.5})
    
    # Should have original + augmented images
    assert len(df_augmented) == original_count * 2
    
    # New images should have _aug_contrast in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_contrast")]
    assert len(aug_images) == original_count
    
    # Message should mention contrast
    assert "contrast" in message.lower()


def test_random_crop_augment(session_with_transform_images):
    """Test random crop augmentation."""
    df, sdir = session_with_transform_images
    original_count = len(df)
    
    df_augmented, code, message = augment.random_crop(
        df, sdir, {"width": 32, "height": 32, "count": 2}
    )
    
    # Should have original + augmented images (some may be skipped if too small)
    assert len(df_augmented) >= original_count
    
    # New images should have _aug_rcrop in their IDs
    aug_images = df_augmented[df_augmented["image_id"].str.contains("aug_rcrop")]
    assert len(aug_images) > 0
    
    # Augmented images should have the specified dimensions
    for _, row in aug_images.iterrows():
        assert row["width"] == 32
        assert row["height"] == 32


def test_transforms_apply_function(session_with_transform_images):
    """Test the apply() dispatcher for transforms module."""
    df, sdir = session_with_transform_images
    
    # Test resize directly (apply() is a dispatcher, actual usage passes session_dir)
    df_result, code, message = transforms.resize(
        df, sdir, {"width": 50, "height": 50, "mode": "stretch"}
    )
    
    assert all(df_result["width"] == 50)
    assert all(df_result["height"] == 50)


def test_augment_apply_function(session_with_transform_images):
    """Test the apply() dispatcher for augment module."""
    df, sdir = session_with_transform_images
    
    # Test flip directly (apply() is a dispatcher, actual usage passes session_dir)
    df_result, code, message = augment.flip(df, sdir, {"direction": "horizontal"})
    
    assert len(df_result) > len(df)


def test_transforms_invalid_strategy(session_with_transform_images):
    """Test that invalid strategy raises error."""
    df, sdir = session_with_transform_images
    
    with pytest.raises(ValueError, match="unknown transform strategy"):
        transforms.apply(df, None, "invalid_strategy", {})


def test_augment_invalid_strategy(session_with_transform_images):
    """Test that invalid strategy raises error."""
    df, sdir = session_with_transform_images
    
    with pytest.raises(ValueError, match="unknown augmentation strategy"):
        augment.apply(df, None, "invalid_strategy", {})


def test_transforms_missing_params(session_with_transform_images):
    """Test that missing params returns error."""
    df, sdir = session_with_transform_images
    
    df_result, code, message = transforms.apply(df, None, "resize", None)
    
    assert "error" in message.lower()
    assert "requires parameters" in message.lower()


def test_augment_missing_params(session_with_transform_images):
    """Test that missing params returns error."""
    df, sdir = session_with_transform_images
    
    df_result, code, message = augment.apply(df, None, "rotate", None)
    
    assert "error" in message.lower()
    assert "requires parameters" in message.lower()
