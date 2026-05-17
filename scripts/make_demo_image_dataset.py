"""Generate a curated demo image dataset for CleanML Vision.

The output ZIP contains intentional quality issues so the demo can showcase
every Magic Clean feature:

  - 60+ synthetic images across 3 classes (red / green / blue)
  - 6 near-duplicates (same hash, different filenames)
  - 4 deliberately blurry images
  - 3 underexposed + 3 overexposed images
  - 1 corrupt file (not a real image)
  - mixed dimensions (some 256x256, some 320x240, some 512x384)
  - mixed color modes (mostly RGB, a few grayscale)
  - mixed formats (.jpg, .png)
  - a labels.csv pairing every filename with its class

Run from project root:

    python scripts/make_demo_image_dataset.py

Output: samples/demo_images.zip  (~3-5 MB)
"""

from __future__ import annotations

import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "samples"
OUT_DIR.mkdir(exist_ok=True, parents=True)
ZIP_PATH = OUT_DIR / "demo_images.zip"

random.seed(42)
np.random.seed(42)

# Class palette — base color + small variations so images aren't identical
CLASSES = {
    "red":   (220, 60, 60),
    "green": (60, 200, 90),
    "blue":  (70, 100, 230),
}

PER_CLASS = 20            # 60 base images total
NEAR_DUP_COUNT = 6        # 6 extra near-duplicate files
BLUR_COUNT = 4
DARK_COUNT = 3
BRIGHT_COUNT = 3
GRAY_COUNT = 3
SIZES = [(256, 256), (320, 240), (512, 384), (400, 400)]


def _make_image(cls: str, idx: int, size: tuple[int, int]) -> Image.Image:
    """Build a synthetic image with a colored gradient + some shapes + a label."""
    base = CLASSES[cls]
    w, h = size

    # gradient + noise
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        gradient = np.linspace(
            max(0, base[c] - 40), min(255, base[c] + 40), w, dtype=np.uint8
        )
        arr[:, :, c] = np.tile(gradient, (h, 1))
    arr = np.clip(arr + np.random.randint(-15, 15, arr.shape), 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")

    # add a few shapes so each image differs
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(2, 5)):
        x0 = random.randint(0, w - 20)
        y0 = random.randint(0, h - 20)
        x1 = x0 + random.randint(20, max(40, w // 3))
        y1 = y0 + random.randint(20, max(40, h // 3))
        col = tuple(random.randint(50, 255) for _ in range(3))
        if random.random() < 0.5:
            d.ellipse([x0, y0, x1, y1], outline=col, width=3)
        else:
            d.rectangle([x0, y0, x1, y1], outline=col, width=3)

    # label
    text = f"{cls}-{idx:02d}"
    try:
        d.text((10, 10), text, fill=(255, 255, 255))
    except Exception:
        pass

    return img


def main() -> None:
    files_to_zip: list[tuple[str, bytes]] = []
    labels: list[tuple[str, str]] = []

    # 1. Base images — 20 per class
    base_records: list[tuple[str, str, Image.Image, str]] = []
    for cls in CLASSES:
        for i in range(PER_CLASS):
            size = random.choice(SIZES)
            img = _make_image(cls, i, size)
            fmt = "JPEG" if random.random() < 0.7 else "PNG"
            ext = "jpg" if fmt == "JPEG" else "png"
            name = f"{cls}_{i:03d}.{ext}"
            base_records.append((name, cls, img, fmt))

    # 2. Mutations: blur + exposure + grayscale (drawn from base records)
    random.shuffle(base_records)
    used_for_mutation: set[int] = set()

    def pick_unused() -> int:
        for _ in range(len(base_records)):
            i = random.randrange(len(base_records))
            if i not in used_for_mutation:
                used_for_mutation.add(i)
                return i
        return random.randrange(len(base_records))

    # Blurry
    for k in range(BLUR_COUNT):
        i = pick_unused()
        name, cls, img, fmt = base_records[i]
        blurred = img.filter(ImageFilter.GaussianBlur(radius=8))
        new_name = f"{cls}_blur_{k:02d}.{name.rsplit('.', 1)[1]}"
        base_records.append((new_name, cls, blurred, fmt))

    # Dark / overexposed
    for k in range(DARK_COUNT):
        i = pick_unused()
        name, cls, img, fmt = base_records[i]
        dark = ImageEnhance.Brightness(img).enhance(0.2)
        new_name = f"{cls}_dark_{k:02d}.{name.rsplit('.', 1)[1]}"
        base_records.append((new_name, cls, dark, fmt))
    for k in range(BRIGHT_COUNT):
        i = pick_unused()
        name, cls, img, fmt = base_records[i]
        bright = ImageEnhance.Brightness(img).enhance(2.5)
        new_name = f"{cls}_bright_{k:02d}.{name.rsplit('.', 1)[1]}"
        base_records.append((new_name, cls, bright, fmt))

    # Grayscale
    for k in range(GRAY_COUNT):
        i = pick_unused()
        name, cls, img, fmt = base_records[i]
        gray = img.convert("L")
        new_name = f"{cls}_gray_{k:02d}.png"   # save grayscales as PNG
        base_records.append((new_name, cls, gray, "PNG"))

    # 3. Near-duplicates — pick existing images and save under different names
    for k in range(NEAR_DUP_COUNT):
        i = random.randrange(len(base_records))
        name, cls, img, fmt = base_records[i]
        ext = name.rsplit(".", 1)[1]
        new_name = f"{cls}_dup_{k:02d}.{ext}"
        # Tiny modification so hash is *near* but not identical
        slightly = img.copy()
        if slightly.mode == "RGB":
            pix = np.array(slightly)
            pix[0:3, 0:3] = (pix[0:3, 0:3].astype(int) + 5).clip(0, 255).astype(np.uint8)
            slightly = Image.fromarray(pix, mode="RGB")
        base_records.append((new_name, cls, slightly, fmt))

    # 4. Encode everything to bytes
    for name, cls, img, fmt in base_records:
        buf = io.BytesIO()
        save_kwargs = {"quality": 85} if fmt == "JPEG" else {}
        img.save(buf, format=fmt, **save_kwargs)
        files_to_zip.append((f"images/{name}", buf.getvalue()))
        labels.append((name, cls))

    # 5. labels.csv
    labels_csv = io.StringIO()
    writer = csv.writer(labels_csv)
    writer.writerow(["filename", "class"])
    writer.writerows(sorted(labels))
    files_to_zip.append(("labels.csv", labels_csv.getvalue().encode("utf-8")))

    # 6. One corrupt file (not a real image)
    files_to_zip.append(("images/broken.jpg", b"this is not a real image file"))

    # 7. Write the zip
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files_to_zip:
            zf.writestr(name, data)

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH}  ({size_mb:.2f} MB)")
    print(f"  total images:    {len(base_records) + 1} (incl. 1 corrupt)")
    print(f"  classes:         {list(CLASSES)}")
    print(f"  duplicates:      {NEAR_DUP_COUNT}")
    print(f"  blurry:          {BLUR_COUNT}")
    print(f"  dark:            {DARK_COUNT}")
    print(f"  overexposed:     {BRIGHT_COUNT}")
    print(f"  grayscale:       {GRAY_COUNT}")
    print(f"  corrupt:         1")
    print(f"  labels.csv:      {len(labels)} rows")


if __name__ == "__main__":
    main()
