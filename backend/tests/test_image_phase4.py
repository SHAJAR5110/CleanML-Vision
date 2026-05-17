"""Tests for image module Phase 4: pairing, magic cleaning, and export."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cleaner.image import export, loader, magic, pair


@pytest.fixture
def test_images_with_labels(tmp_path):
    """Generate a ZIP with images and labels.csv."""
    zip_path = tmp_path / "test_images_labeled.zip"
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Create 6 images (3 cats, 3 dogs)
        for i in range(3):
            # Cat images
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(f"cat_{i}.png", buf.getvalue())
            
            # Dog images
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(f"dog_{i}.png", buf.getvalue())
        
        # Create labels.csv
        labels_data = {
            "image_id": [f"cat_{i}" for i in range(3)] + [f"dog_{i}" for i in range(3)],
            "label": ["cat"] * 3 + ["dog"] * 3,
            "category": ["animal"] * 6
        }
        labels_df = pd.DataFrame(labels_data)
        labels_csv = labels_df.to_csv(index=False)
        zf.writestr("labels.csv", labels_csv)
    
    return zip_path


@pytest.fixture
def session_with_labels(test_images_with_labels, tmp_path):
    """Load test images with labels into a session."""
    sdir = tmp_path / "session"
    sdir.mkdir()
    df, _, _ = loader.load_from_zip(test_images_with_labels, sdir)
    return df, sdir


def test_load_labels_csv(session_with_labels):
    """Test loading labels.csv from session directory."""
    df, sdir = session_with_labels
    
    df_labels = pair.load_labels_csv(sdir)
    
    assert df_labels is not None
    assert "image_id" in df_labels.columns
    assert "label" in df_labels.columns
    assert len(df_labels) == 6


def test_join_with_labels(session_with_labels):
    """Test joining images with labels."""
    df, sdir = session_with_labels
    
    df_joined, code, message = pair.join_with_labels(df, sdir)
    
    # Should have label columns
    assert "label" in df_joined.columns
    assert "category" in df_joined.columns
    
    # All images should be matched
    assert df_joined["label"].notna().sum() == len(df_joined)
    
    # Message should mention matched count
    assert "joined" in message.lower()


def test_join_with_labels_custom_column(session_with_labels):
    """Test joining with custom column name."""
    df, sdir = session_with_labels
    
    df_joined, code, message = pair.join_with_labels(df, sdir, {"on": "image_id"})
    
    assert "label" in df_joined.columns


def test_filter_by_label(session_with_labels):
    """Test filtering images by label."""
    df, sdir = session_with_labels
    
    # First join with labels
    df_joined, _, _ = pair.join_with_labels(df, sdir)
    
    # Filter to only cats
    df_filtered, code, message = pair.filter_by_label(
        df_joined, sdir, {"column": "label", "values": ["cat"]}
    )
    
    # Should have only 3 images (cats)
    assert len(df_filtered) == 3
    assert all(df_filtered["label"] == "cat")
    
    # Message should mention filtered count
    assert "filtered" in message.lower()


def test_balance_by_label_undersample(session_with_labels):
    """Test undersampling to balance classes."""
    df, sdir = session_with_labels
    
    # First join with labels
    df_joined, _, _ = pair.join_with_labels(df, sdir)
    
    # Create imbalanced dataset (remove 1 dog)
    df_imbalanced = df_joined[~((df_joined["label"] == "dog") & (df_joined.index == 5))]
    
    # Balance by undersampling
    df_balanced, code, message = pair.balance_by_label(
        df_imbalanced, sdir, {"column": "label", "strategy": "undersample"}
    )
    
    # Should have equal counts (2 cats, 2 dogs)
    assert df_balanced["label"].value_counts()["cat"] == 2
    assert df_balanced["label"].value_counts()["dog"] == 2
    
    # Message should mention undersampling
    assert "undersample" in message.lower()


def test_balance_by_label_oversample(session_with_labels):
    """Test oversampling to balance classes."""
    df, sdir = session_with_labels
    
    # First join with labels
    df_joined, _, _ = pair.join_with_labels(df, sdir)
    
    # Create imbalanced dataset (remove 1 dog)
    df_imbalanced = df_joined[~((df_joined["label"] == "dog") & (df_joined.index == 5))]
    
    # Balance by oversampling
    df_balanced, code, message = pair.balance_by_label(
        df_imbalanced, sdir, {"column": "label", "strategy": "oversample"}
    )
    
    # Should have equal counts (3 cats, 3 dogs)
    assert df_balanced["label"].value_counts()["cat"] == 3
    assert df_balanced["label"].value_counts()["dog"] == 3
    
    # Message should mention oversampling
    assert "oversample" in message.lower()


def test_split_by_label(session_with_labels):
    """Test stratified train/test split."""
    df, sdir = session_with_labels
    
    # First join with labels
    df_joined, _, _ = pair.join_with_labels(df, sdir)
    
    # Split 80/20
    df_split, code, message = pair.split_by_label(
        df_joined, sdir, {"column": "label", "test_size": 0.33, "random_state": 42}
    )
    
    # Should have split column
    assert "split" in df_split.columns
    
    # Should have both train and test
    assert "train" in df_split["split"].values
    assert "test" in df_split["split"].values
    
    # Check stratification (roughly equal proportions)
    train_df = df_split[df_split["split"] == "train"]
    test_df = df_split[df_split["split"] == "test"]
    
    # Both splits should have both classes
    assert len(train_df["label"].unique()) == 2
    assert len(test_df["label"].unique()) == 2
    
    # Message should mention split counts
    assert "split" in message.lower()


def test_magic_cleaning(session_with_labels):
    """Test magic cleaning pipeline."""
    df, sdir = session_with_labels
    
    df_cleaned, operations = magic.run(df, sdir)
    
    # Should have applied multiple operations
    assert len(operations) > 0
    
    # Should have perceptual_hash column (from dedup)
    assert "perceptual_hash" in df_cleaned.columns
    
    # Should have quality flags
    assert "blur_score" in df_cleaned.columns
    assert "is_blurry" in df_cleaned.columns
    assert "exposure" in df_cleaned.columns
    
    # All operations should have required fields
    for op in operations:
        assert "op" in op
        assert "code" in op
        assert "message" in op


def test_magic_get_summary(session_with_labels):
    """Test magic cleaning summary generation."""
    df, sdir = session_with_labels
    
    df_cleaned, operations = magic.run(df, sdir)
    summary = magic.get_summary(operations)
    
    # Summary should mention operation count
    assert str(len(operations)) in summary
    assert "cleaning operations" in summary.lower()


def test_magic_get_code(session_with_labels):
    """Test magic cleaning code generation."""
    df, sdir = session_with_labels
    
    df_cleaned, operations = magic.run(df, sdir)
    code = magic.get_code(operations)
    
    # Code should contain step markers
    assert "Step" in code
    assert "# Magic cleaning pipeline" in code


def test_export_to_zip(session_with_labels, tmp_path):
    """Test exporting to ZIP archive."""
    df, sdir = session_with_labels
    
    output_path = tmp_path / "export.zip"
    success, message = export.export_to_zip(df, sdir, output_path)
    
    assert success is True
    assert output_path.exists()
    
    # Verify ZIP contents
    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        assert "metadata.csv" in names
        assert "labels.csv" in names
        # Should have image files
        assert any("cat_" in name for name in names)
        assert any("dog_" in name for name in names)


def test_export_to_numpy(session_with_labels, tmp_path):
    """Test exporting to NumPy array."""
    df, sdir = session_with_labels
    
    output_path = tmp_path / "export.npy"
    success, message = export.export_to_numpy(df, sdir, output_path)
    
    assert success is True
    assert output_path.exists()
    
    # Load and verify
    data = np.load(output_path)
    assert data.shape[0] == len(df)  # Number of images
    assert len(data.shape) == 4  # (N, H, W, C)


def test_export_to_numpy_with_labels(session_with_labels, tmp_path):
    """Test exporting to NumPy with labels."""
    df, sdir = session_with_labels
    
    # First join with labels
    df_joined, _, _ = pair.join_with_labels(df, sdir)
    
    output_path = tmp_path / "export.npy"
    success, message = export.export_to_numpy(
        df_joined, sdir, output_path, {"include_labels": True}
    )
    
    assert success is True
    assert output_path.exists()
    
    # Load and verify
    data = np.load(output_path, allow_pickle=True).item()
    assert "images" in data
    assert "labels" in data
    assert data["images"].shape[0] == len(df_joined)


def test_export_split(session_with_labels, tmp_path):
    """Test exporting train/test split."""
    df, sdir = session_with_labels
    
    # First join with labels and split
    df_joined, _, _ = pair.join_with_labels(df, sdir)
    df_split, _, _ = pair.split_by_label(
        df_joined, sdir, {"column": "label", "test_size": 0.33, "random_state": 42}
    )
    
    output_dir = tmp_path / "split_export"
    success, message = export.export_split(
        df_split, sdir, output_dir, {"format": "zip"}
    )
    
    assert success is True
    assert (output_dir / "train.zip").exists()
    assert (output_dir / "test.zip").exists()
    
    # Message should mention both splits
    assert "train" in message.lower()
    assert "test" in message.lower()


def test_export_split_without_split_column(session_with_labels, tmp_path):
    """Test that export_split fails without split column."""
    df, sdir = session_with_labels
    
    output_dir = tmp_path / "split_export"
    success, message = export.export_split(df, sdir, output_dir)
    
    assert success is False
    assert "split" in message.lower()


def test_pair_apply_function(session_with_labels):
    """Test the apply() dispatcher for pair module."""
    df, sdir = session_with_labels
    
    # Test join_labels through apply (pass session_dir explicitly for testing)
    df_result, code, message = pair.apply(df, None, "join_labels", session_dir=sdir)
    
    assert "label" in df_result.columns


def test_pair_invalid_strategy(session_with_labels):
    """Test that invalid strategy raises error."""
    df, sdir = session_with_labels
    
    with pytest.raises(ValueError, match="unknown pair strategy"):
        pair.apply(df, None, "invalid_strategy")


def test_export_invalid_strategy(session_with_labels, tmp_path):
    """Test that invalid strategy raises error."""
    df, sdir = session_with_labels
    
    output_path = tmp_path / "export.zip"
    
    with pytest.raises(ValueError, match="unknown export strategy"):
        export.apply(df, output_path, "invalid_strategy")
