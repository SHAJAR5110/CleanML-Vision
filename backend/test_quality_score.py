"""Quick script to get quality score from synthetic test images."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tests.test_image_phase1 import synthetic_image_zip, session_dir
from cleaner.image import loader, profiler

# Create temp directories
tmp = Path(tempfile.mkdtemp())
zip_path = synthetic_image_zip(tmp)
sdir = tmp / "session"
sdir.mkdir()

# Load and profile
df, code, message = loader.load_from_zip(zip_path, sdir)
profile = profiler.profile_images(df, sdir)

print(f"Quality Score: {profile['quality_score']}")
print(f"Grade: {profile['grade']}")
print(f"Warnings: {profile['warnings']}")
print(f"Total Images: {profile['total_images']}")
print(f"Formats: {profile['formats']}")
print(f"Mode Distribution: {profile['mode_distribution']}")
