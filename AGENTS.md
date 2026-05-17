# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

**CleanML Vision** is a full-stack web application for intelligent dataset cleaning and ML preprocessing — supporting both **tabular** (CSV) and **image** datasets. It transforms messy data into ML-ready inputs through automated profiling, smart suggestions, and one-click "Magic Clean" functionality. The application runs 100% locally with no external data transmission and no AI runtime dependency — the cleaning engine is pure pandas/scikit-learn, deterministic and reproducible.

### Core Purpose
- **Auto-profile** datasets: detect types, missing values, outliers, skewness, cardinality
- **Smart cleaning**: 59+ distinct operations across missing values, outliers, encoding, scaling, text, datetime
- **Magic Clean**: one-click orchestration of 10-20 operations based on heuristics
- **Reproducibility**: export Python notebooks (.ipynb) that replay the entire cleaning pipeline
- **Visualization**: 8 interactive Plotly chart types for data exploration
- **Image datasets** (current sprint focus): ZIP upload, per-image profiling, perceptual-hash deduplication, blur/exposure quality flags, resize / color conversion / normalization, augmentation pipeline, joint cleaning of images with their `labels.csv`, export to ZIP / NumPy `.npy` / PyTorch tensors

### Technology Stack

**Backend:**
- Python 3.14
- Flask 3 with CORS support
- pandas 3 (data manipulation)
- scikit-learn 1.7 (ML preprocessing)
- imbalanced-learn (class balancing)
- scipy, numpy 2
- Pillow, opencv-python-headless, imagehash (image dataset support — current sprint)
- pytest (testing framework)

**Frontend:**
- Vanilla HTML/CSS/JavaScript (no framework)
- Plotly.js 2.35 (CDN) for visualizations
- Modern dark UI with drag-drop support

**Testing:**
- 111 tests with 81% code coverage
- pytest with shared fixtures in `conftest.py`

### Architecture

**Session-Based State Management:**
- Each upload creates a unique session ID (12-char hex)
- Session directory stores: `original.pkl`, `current.pkl`, `source.txt`, `history.json`
- Operations are applied to `current.pkl` and logged in history
- Supports undo (replay history minus last op) and reset (restore original)

**Backend Modules (`backend/cleaner/`):**
- `profiler.py` - Auto-profiler with 7-way type inference and quality scoring (0-100)
- `missing.py` - 11 missing-value strategies (mean, median, KNN, ffill, etc.)
- `outliers.py` - IQR, Z-score, Isolation Forest, DBSCAN
- `encoders.py` - One-hot, Label, Ordinal, Frequency, Target, Binary
- `scalers.py` - Standard, MinMax, Robust, Normalizer, Log
- `text_clean.py` - Strip, lowercase, stopwords, punctuation, word/char count
- `datetime_fix.py` - Parse and extract year/month/day/weekday/hour/quarter
- `dtype_fix.py` - Convert to numeric (strips $, %, commas) and boolean
- `label_norm.py` - Fuzzy clustering for inconsistent labels (e.g., "Male"/"M"/"male")
- `validate.py` - Cross-field validation rules (A<B, A≤B, A=B, sum checks, age-DOB)
- `feature_eng.py` - Formula builder using pandas eval
- `splitter.py` - Stratified train/test split
- `balance.py` - SMOTE, random oversample/undersample
- `reduce.py` - PCA, VarianceThreshold, SelectKBest
- `merge.py` - Multi-CSV join (inner/left/right/outer)
- `suggest.py` - Ranked next-step recommender
- `magic.py` - One-click heuristic orchestrator
- `pipeline.py` - Operation router, history management, undo/reset
- `notebook_export.py` - Jupyter notebook generator

An `image/` sub-package is being added to mirror the tabular pipeline for image datasets: `image/loader.py` (ZIP walk + format detection), `image/profiler.py` (dimensions, channels, integrity, perceptual-hash duplicates), `image/quality.py` (blur via Laplacian variance, brightness/exposure flags), `image/transforms.py` (resize, color conversion, normalization), `image/augment.py` (rotate/flip/crop/brightness/contrast), `image/pair.py` (join images with `labels.csv`), and `image/export.py` (ZIP / `.npy` / PyTorch tensor pickle).

**Frontend Structure (`frontend/`):**
- `index.html` - Single-page app with 5 screens (upload, profile, clean, compare, export)
- `css/style.css` - Dark modern UI
- `js/api.js` - Fetch helpers for backend communication
- `js/app.js` - Main controller and screen routing
- `js/profile.js` - KPI and column rendering
- `js/plots.js` - 8 Plotly chart types
- `js/suggest.js` - Suggestion panel
- `js/drawer.js` - Per-column cleaning drawer
- `js/op-modal.js` - Operation preview modal
- `js/label-modal.js` - Label normalizer dialog
- `js/mlprep.js` - ML prep modal (feature eng, split, balance, reduce)
- `js/merge-modal.js` - Multi-CSV merge dialog
- `js/compare.js` - Before/after comparison screen

**API Endpoints (21 total + image endpoints being added):**
- Upload/Load: `/api/upload`, `/api/load-url`
- Profile: `/api/profile/<sid>`, `/api/suggest/<sid>`, `/api/column/<sid>/<col>`
- Cleaning: `/api/clean/<sid>`, `/api/magic/<sid>`
- History: `/api/history/<sid>`, `/api/undo/<sid>`, `/api/reset/<sid>`
- Visualization: `/api/correlation/<sid>`, `/api/scatter/<sid>`
- Advanced: `/api/label-groups/<sid>/<col>`, `/api/merge/<sid>`, `/api/split/<sid>`
- Export: `/api/download/<sid>`, `/api/notebook/<sid>`, `/api/download-split/<sid>`
- Preview: `/api/preview/<sid>`, `/api/preview-op/<sid>`
- Compare: `/api/compare/<sid>`
- Image endpoints (current sprint): `/api/image/upload`, `/api/image/profile/<sid>`, `/api/image/clean/<sid>`, `/api/image/preview/<sid>`, `/api/image/export/<sid>`

## Building and Running

### Start the Backend

```powershell
cd backend
pip install -r requirements.txt
python app.py
```

The server runs on `http://127.0.0.1:5000` by default (configurable via `PORT` environment variable).

### Run Tests

```powershell
cd backend
python -m pytest tests/ -v
```

For coverage report:
```powershell
python -m pytest tests/ --cov=cleaner --cov-report=html
```

### Access the Application

Open `http://127.0.0.1:5000` in a browser. The Flask app serves the frontend static files.

## Development Conventions

### Code Style

**Python:**
- Type hints using `from __future__ import annotations` for forward references
- Docstrings for modules and complex functions
- Private functions prefixed with `_` (e.g., `_sdir()`, `_json_safe()`)
- Exception handling with specific error messages returned as JSON

**JavaScript:**
- Vanilla ES6+ (no transpilation)
- Async/await for API calls
- DOM manipulation without frameworks
- Event delegation for dynamic content

### Testing Approach

**Fixtures (`tests/conftest.py`):**
- `dirty_df` - Small dataset with every cleaning case (outliers, missing, constants, etc.)
- `clean_numeric_df` - Pure numeric data for scaling/reduction tests
- `int64_df` - Tests pandas 3.0 nullable Int64 dtype handling
- `imbalanced_df` - For class-balancing tests (100 minority, 10 majority)
- `flask_client` - Flask test client for HTTP endpoint testing

**Test Organization:**
- `test_profiler.py` - 14 tests for auto-profiler and quality scoring
- `test_cleaners.py` - 40 tests for missing, outliers, encoders, scalers, text, datetime
- `test_advanced.py` - 24 tests for label normalization, validation, feature eng, balance, reduce
- `test_pipeline.py` - 9 tests for operation routing, history, undo/reset
- `test_api.py` - 26 tests for Flask endpoints using test_client

### Key Patterns

**Safe JSON Handling:**
- Custom `SafeJSONProvider` converts NaN/Infinity to `null` to prevent invalid JSON
- `_json_safe()` helper recursively sanitizes nested structures
- All API responses use this provider

**NaN Token Recognition:**
- On CSV load, treats 14+ tokens as NaN: `?`, `N/A`, `null`, `nan`, `missing`, `unknown`, `-`, `--`, `none`, `#N/A`, `#NULL!`, `(null)`, `.`, empty string
- `profiler.py` detects embedded NaN strings in text columns

**Type Inference:**
- 7-way classification: numeric, categorical, datetime, boolean, text, id_like, constant
- Handles numeric strings (`$1,200`, `45%`) and datetime strings (`2020-01-15`)
- Detects id-like columns (high cardinality + "id" in name)

**Quality Scoring:**
- 0-100 scale with letter grades (A-F)
- Penalizes: high missing %, constant columns, duplicates, embedded NaN, skewness, outliers
- Example: Titanic dataset goes from 84/B to 92/A after Magic Clean

**Operation Pipeline:**
- Each operation returns: `(new_df, code_string, message_string)`
- Code string is valid pandas/sklearn that can be exported to notebook
- History stores: `{"op": {...}, "code": "...", "message": "..."}`

**Session Cleanup:**
- Sessions stored in `backend/_sessions/<session_id>/`
- No automatic cleanup (consider implementing TTL-based cleanup for production)

## Common Tasks

### Adding a New Cleaning Operation

1. **Create the function** in the appropriate `cleaner/*.py` module:
   ```python
   def my_new_strategy(df: pd.DataFrame, column: str, **params) -> tuple[pd.DataFrame, str, str]:
       # Apply transformation
       df_new = df.copy()
       # ... transformation logic ...
       
       # Generate code string
       code = f"df['{column}'] = df['{column}'].transform(...)"
       
       # Generate message
       message = f"Applied my_new_strategy to {column}"
       
       return df_new, code, message
   ```

2. **Register in `pipeline.py`** under the appropriate family in `apply_op()`:
   ```python
   elif family == "my_family":
       if strategy == "my_strategy":
           return my_module.my_new_strategy(df, column, **params)
   ```

3. **Add tests** in `tests/test_cleaners.py` or appropriate test file:
   ```python
   def test_my_new_strategy(dirty_df):
       result, code, msg = my_module.my_new_strategy(dirty_df, "column_name")
       assert "expected_result" in result["column_name"].values
   ```

4. **Update frontend** in `js/drawer.js` to expose the new operation in the UI.

### Adding a New Visualization

1. **Add chart function** in `js/plots.js`:
   ```javascript
   function renderMyChart(data, elementId) {
       const trace = { /* Plotly trace config */ };
       const layout = { /* Plotly layout config */ };
       Plotly.newPlot(elementId, [trace], layout);
   }
   ```

2. **Add UI trigger** in `index.html` and wire up in `js/app.js`.

### Modifying Magic Clean Heuristics

Edit `backend/cleaner/magic.py` in the `run()` function. The function profiles the dataset and applies operations based on column types and quality metrics:

```python
def run(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    profile = profile_dataframe(df)
    operations = []
    
    for col_info in profile["columns"]:
        # Add heuristics based on col_info["inferred_type"], 
        # col_info["missing_pct"], col_info["warnings"], etc.
        
    # Apply operations and collect history
    return final_df, operations
```

## Important Notes

- **Max upload size**: 500 MB (configurable via `MAX_UPLOAD_MB` in `app.py`)
- **Preview limit**: 50 rows (configurable via `PREVIEW_ROWS`)
- **Session storage**: Uses pickle files (not suitable for untrusted data in production)
- **CORS**: Enabled for all origins (restrict in production)
- **Debug mode**: Enabled by default in `app.run()` (disable in production)

## Future Roadmap (v2)

Image dataset support is the **current sprint** (see Core Purpose and `cleaner/image/` notes above). Beyond that, planned future work includes:

- Time-series mode (seasonality detection, gap-fill, lag features)
- Auto-EDA reports (single-page PDF/HTML with every chart + insight)
- Baseline models (random forest, XGBoost with feature importances on the cleaned data)
- Video / audio dataset support (extending the same profile → clean → export pattern)
- Cloud sync and collaborative cleaning sessions
- Docker image for one-line deployment
- Optional plain-English suggestion explanations (kept opt-in to preserve the AI-free runtime guarantee)

## Troubleshooting

**Tests failing with import errors:**
- Ensure `backend/` is in Python path (handled by `conftest.py`)
- Run tests from `backend/` directory: `cd backend && pytest tests/`

**JSON serialization errors:**
- Check that `SafeJSONProvider` is active in Flask app
- Use `_json_safe()` helper for manual serialization

**Session not found errors:**
- Sessions are ephemeral and stored in `_sessions/` directory
- Implement session cleanup or persistence layer for production use

**Frontend not loading:**
- Verify Flask is serving static files from `frontend/` directory
- Check `FRONTEND_DIR` path in `app.py`
- Ensure Plotly CDN is accessible (or download locally)
