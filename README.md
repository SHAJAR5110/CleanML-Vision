# CleanML Vision — local-first dataset cleaner for ML

**AI-built. AI-free. One click from messy data to ML-ready.**

A web tool that takes a messy CSV **or** an image dataset ZIP and turns it into ML-ready inputs — with smart suggestions, one-click "Magic Clean", interactive visualizations, and a reproducible Python notebook of every transformation.

**Runs 100% locally. Your data never leaves your machine.**

> Built for the **IBM Bob Hackathon (2026)**. IBM Bob accelerated the 48-hour build — the shipped product itself uses **zero external AI APIs at runtime**.

---

## Why this tool exists

ML teams spend 60–80% of their time on data cleaning, not modeling. Existing AI cleaning tools demand API keys, leak your data to third-party servers, cost money per row, and produce non-deterministic output that breaks reproducibility.

For regulated industries — healthcare, finance, defense — that's a non-starter.

**CleanML Vision is the deterministic alternative.** Same one-click magic, but the cleaning engine is pure pandas/scikit-learn/PIL. No API keys, no cloud, no rate limits, no per-row costs.

---

## Stack

- **Backend:** Python 3.14 · Flask 3 · pandas 3 · scikit-learn 1.7 · imbalanced-learn · Pillow · OpenCV (opencv-python-headless) · imagehash · scipy · numpy 2
- **Frontend:** vanilla HTML/CSS/JS (no framework) · Plotly.js 2.35 (CDN)
- **Tests:** pytest · 170+ tests across tabular + image modules

## Run it

```powershell
cd backend
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

To run the test suite:

```powershell
cd backend
python -m pytest tests/ -v
```

A pre-built demo image dataset (`samples/demo_images.zip` — 80 images with intentional duplicates, blurry shots, mixed dimensions, corrupt files, and a `labels.csv`) is included so you can exercise every Vision feature in one upload.

---

## Features — Tabular (CSV)

### Data ingestion
- Drag-drop CSV up to **1 GB** (~5–10M typical rows)
- Paste any CSV URL (Kaggle raw, GitHub raw, anywhere)
- 3 sample datasets one-click: **Titanic / Iris / Tips**
- Auto-recognises **14+ NaN tokens at load time** (`?`, `N/A`, `null`, `nan`, `missing`, `unknown`, `-`, `--`, `none`, `#N/A`, `#NULL!`, `(null)`, `.`, empty)

### Auto-Profiler (runs on every upload)
- 7-way column type inference: numeric / categorical / datetime / boolean / text / id-like / constant
- Detects numeric strings (`$1,200`, `45%`) and datetime strings (`2020-01-15`)
- Counts missing including embedded NaN tokens hidden in string columns
- Per-column: outlier count (IQR), skewness, cardinality bucket, top values
- **Quality Score 0–100 with letter grade A–F**

### Smart Suggestions panel
- Ranked next-step recommendations with reasons and impact level
- **👁 See** button on each suggestion opens a preview modal: stats, top values, sample of affected rows, outlier list, generated code, plus an op-picker to swap to a different operation before applying

### ✨ Magic Clean — one click
- Auto-orchestrates 10–20 operations based on the profile
- Per-column heuristics:
  - Constant → drop · >70% missing → drop · id-like/text → drop
  - Numeric: median impute → IQR cap/remove → log if skewed
  - Categorical: mode fill → one-hot (low-card) or frequency (high-card)
  - Datetime: parse + extract year/month/weekday
- **Tested: Titanic 84/B → 92/A in one click** (15 ops applied)

### Per-column cleaning drawer
Pick category → strategy → optional parameter → Apply.

| Category | Strategies |
|---|---|
| Missing values | mean · median · mode · KNN · ffill · bfill · custom constant · standardize NaN tokens · drop rows · drop col · fill all smart |
| Outliers | IQR remove · IQR cap · Z-score · Isolation Forest · DBSCAN |
| Encoding | One-hot · Label · Ordinal · Frequency · Target · Binary |
| Scaling | Standard · MinMax · Robust · Normalizer · Log |
| Text cleanup | strip · lowercase · collapse · remove special · remove punctuation · remove stopwords · alphabetic only · word-count feature · char-count feature |
| Datetime | parse · extract Y/M/D/weekday/hour/quarter |
| Dtype repair | to numeric (strips `$`,`,`,`%`) · to boolean |

### Visualization workshop
8 Plotly chart types, drag column thumbnails to swap views:
Missing-value heatmap · Column quality issues · Histogram · Box plot · Bar chart · Scatter · Correlation heatmap · Quality score gauge

### ML Preparation tools
- **🧮 Feature engineering** — formula builder (`BMI = weight / (height/100)**2`)
- **✂ Train/Test split** — stratified, zip download (`train.csv` + `test.csv`)
- **⚖ Class balancing** — SMOTE / random oversample / random undersample
- **📉 Feature reduction** — PCA · VarianceThreshold · SelectKBest (4 score functions)
- **🔗 Multi-CSV merge** — inner / left / right / outer join

### Advanced cleaning
- **🏷 Inconsistent label normalizer** — fuzzy clusters `"Male"`/`"M"`/`"male"`/`"MALE"`, editable canonical per group
- **✓ Cross-field validation** — 5 rules: `A<B`, `A≤B`, `A=B`, `sum(cols)=total_col`, age-vs-DOB consistency
- DBSCAN multivariate anomaly detection
- Stopword + punctuation removal for free-text columns

---

## Features — Image datasets (Vision module)

### Image ingestion
- Drag-drop ZIP of images (up to 1 GB)
- Supports `.png .jpg .jpeg .bmp .gif .webp .tiff`
- Auto-detects `labels.csv` inside the ZIP and pairs it with the images
- Skips corrupt files gracefully on load

### Image auto-profiler
- Per image: format, width, height, channels, color mode, file size
- Aggregate stats: format distribution, dimension stats (min/max/median/p95), aspect ratio stats, color mode distribution
- Quality score 0–100 with A–F grade — same scale as the tabular profiler
- Warning tags: `mixed_dimensions`, `mixed_color_modes`, `extreme_aspect_ratios`, `corrupt_files`

### Image cleaning operations

| Category | Strategies |
|---|---|
| Quality | remove_corrupt · remove_blurry (Laplacian variance threshold) · flag_low_quality (blur + exposure) |
| Dedup | compute_hashes (perceptual hash) · remove_duplicates (Hamming distance) |
| Transforms | resize (stretch/pad/crop) · convert_color (RGB/L/RGBA) · normalize (ImageNet/0-1/z-score) · center_crop |
| Augment | rotate (90/180/270/random) · flip · adjust_brightness · adjust_contrast · random_crop |
| Pair | join_with_labels · filter_by_label · balance_by_label · split_by_label |

### ✨ Magic Clean for images — one click, opinionated ML prep
Always runs:
- Remove corrupt files (if any)
- Compute perceptual hashes (dataset becomes dedup-ready)
- Remove near-duplicates (Hamming distance ≤ 5)
- Flag every quality issue (blur, under/overexposure)
- Convert all images to RGB (channels-last)
- Resize all to **224×224** (ImageNet standard — ready for any CNN)

Conditionally:
- Remove severely blurry images (if >5% blurry)

### Image exports
- Cleaned **ZIP** (images + updated metadata.csv)
- **NumPy `.npy`** array of shape `(N, H, W, C)`
- **PyTorch tensor** pickle (`{"images": Tensor, "labels": Tensor, "metadata": df}`)
- Optional train/test split before export

---

## Reproducibility — the killer feature

Every operation (tabular or image) generates the equivalent pandas/sklearn/PIL/cv2 code. Users download a complete Jupyter notebook (`.ipynb`) that reproduces the entire pipeline on any future dataset — no CleanML Vision needed afterwards.

---

## By the numbers

- **59+** tabular cleaning operations
- **22+** image cleaning operations
- **8** Plotly chart types
- **26+** HTTP endpoints (tabular + image)
- **170+** passing tests across both modules
- **81%** code coverage on the core cleaner package
- **1 GB** upload limit (CSV or image ZIP)
- **Magic Clean: Titanic 84/B → 92/A in 1 click** (tabular)
- **Magic Clean: 80 messy images → 73 clean 224×224 RGB tensors in 10s** (image)

---

## Built with IBM Bob

This project was developed during the IBM Bob Hackathon over 48 hours. Bob's repo-context capability accelerated:
- **Codebase onboarding** — Bob explored 30+ existing tabular modules, documented patterns, proposed an image module architecture that mirrors them
- **Backend implementation** — 9 image modules (loader, profiler, quality, dedup, transforms, augment, pair, magic, export)
- **Test authoring** — 64 image-module tests written alongside the code
- **Frontend integration** — image upload routing, profile screen, thumbnail grid, cleaning drawer, export modal

Bob session logs and an `AGENTS.md` describing the project conventions are included in the repo for transparency.

**Importantly:** Bob was used at build time only. The shipped product has zero runtime AI dependency.

---

## Folder layout

```
backend/
  app.py                       Flask routes + safe JSON provider
  requirements.txt
  cleaner/
    profiler.py                auto-profiler + quality score
    missing.py                 11 missing-value strategies
    outliers.py                IQR, Z-score, Isolation Forest, DBSCAN
    encoders.py                One-hot, Label, Ordinal, Frequency, Target, Binary
    scalers.py                 Standard, MinMax, Robust, Normalizer, Log
    duplicates.py              row dedup, constant/high-missing/high-corr drops
    text_clean.py              strip, lower, stopwords, punctuation, word/char count
    datetime_fix.py            parse + extract parts
    dtype_fix.py               to_numeric, to_boolean
    label_norm.py              fuzzy label clustering
    validate.py                cross-field validation rules
    feature_eng.py             formula builder (pandas eval)
    splitter.py                stratified train/test split
    balance.py                 SMOTE, oversample, undersample
    reduce.py                  PCA, VarianceThreshold, SelectKBest
    merge.py                   multi-CSV join
    suggest.py                 ranked next-step recommender
    magic.py                   tabular one-click orchestrator
    pipeline.py                op router + history/undo/reset
    notebook_export.py         .ipynb generator
    image/
      loader.py                ZIP extraction, format detection
      profiler.py              per-image profiling + quality scoring
      quality.py               blur (Laplacian variance) + exposure + corrupt
      dedup.py                 perceptual hashing + near-duplicate removal
      transforms.py            resize, color conversion, normalization
      augment.py               rotation, flip, brightness/contrast, crop
      pair.py                  join with labels.csv, filter, balance, split
      export.py                ZIP, NumPy, PyTorch tensor exports
      magic.py                 image one-click orchestrator
  tests/
    conftest.py                shared fixtures (dirty_df, int64_df, …)
    test_profiler.py           profiler tests
    test_cleaners.py           per-column cleaning tests
    test_advanced.py           label_norm, validate, feature_eng, balance, reduce
    test_pipeline.py           pipeline orchestrator + Magic Clean
    test_api.py                Flask test_client integration
    test_image_phase1.py       image loader + profiler tests
    test_image_phase2.py       quality + dedup tests
    test_image_phase3.py       transforms + augment tests
    test_image_phase4.py       pair + magic + export tests

frontend/
  index.html                   all screens + modals
  css/style.css                dark modern UI
  js/
    api.js                     fetch + XHR upload helpers
    app.js                     main controller, screen routing
    upload.js                  drag-drop, URL load, two-phase progress
    profile.js                 tabular KPI/column/preview rendering
    plots.js                   8 Plotly chart types
    viz.js                     visualization workshop + drag-drop thumbs
    suggest.js                 suggestion panel
    op-modal.js                operation preview modal
    drawer.js                  per-column cleaning drawer
    label-modal.js             label normalizer dialog
    validate-modal.js          cross-field validation dialog
    mlprep.js                  4-in-1 ML prep modal
    merge-modal.js             multi-CSV merge dialog
    compare.js                 before/after compare screen
    image-profile.js           image KPI, grid, quality summary
    image-clean.js             image cleaning panel
    image-export.js            image export modal

samples/
  demo_images.zip              80-image curated demo dataset

scripts/
  make_demo_image_dataset.py   regenerates samples/demo_images.zip

AGENTS.md                      project conventions for AI dev partners
README.md
```

---

## Roadmap

- 📈 **Time-series mode** — seasonality detection, gap-fill, lag features
- 📊 **Auto-EDA reports** — single PDF/HTML with every chart + insight
- 🎯 **Baseline models built-in** — random forest + XGBoost on the cleaned data, with feature importances
- 🎥 **Video / audio support** — extend the same profile → clean → export pattern
- 🌐 **Cloud sync** — share cleaning sessions, collaborate on pipelines
- 🐳 **Docker image** — one-line deploy on any server
- Optional **plain-English suggestion explanations** (kept opt-in to preserve the AI-free runtime guarantee)

---

## License

MIT — use freely.
