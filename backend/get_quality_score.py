"""Standalone script to get quality score from synthetic test images."""
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from cleaner.image import loader, profiler

# Create temp directory
tmp_path = Path(tempfile.mkdtemp())
zip_path = tmp_path / "test_images.zip"

# Generate synthetic images
with zipfile.ZipFile(zip_path, "w") as zf:
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
    
    # Corrupt file
    zf.writestr("corrupt.png", b"This is not a valid image file")
    
    # Labels CSV
    zf.writestr("labels.csv", "filename,label\nimage1.png,cat\nimage2.jpg,dog\n")

# Create session directory
sdir = tmp_path / "session"
sdir.mkdir()

# Load and profile
df, code, message = loader.load_from_zip(zip_path, sdir)
profile = profiler.profile_images(df, sdir)

print("=" * 60)
print("PHASE 1 TEST RESULTS")
print("=" * 60)
print(f"Quality Score: {profile['quality_score']}")
print(f"Grade: {profile['grade']}")
print(f"Total Images: {profile['total_images']}")
print(f"Formats: {profile['formats']}")
print(f"Mode Distribution: {profile['mode_distribution']}")
print(f"Warnings: {profile['warnings']}")
print(f"\nLoader Message: {message}")
print("=" * 60)
