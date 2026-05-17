# CleanML test suite

## Run

```powershell
cd backend
python -m pytest tests/ -v
```

With coverage:
```powershell
python -m pytest tests/ --cov=cleaner --cov=app --cov-report=term-missing
```

## Result (last run)

```
111 passed in 5.96s
TOTAL coverage: 81%
```

## Layout

| File | Tests | What it covers |
|---|---|---|
| `conftest.py` | — | Shared fixtures: `dirty_df`, `clean_numeric_df`, `int64_df`, `imbalanced_df`, `flask_client` |
| `test_profiler.py` | 14 | Type inference, missing detection, outlier flags, quality score |
| `test_cleaners.py` | 40 | Every per-column cleaning strategy (missing, outliers, encoders, scalers, text, datetime, dtype, duplicates) |
| `test_advanced.py` | 24 | label_norm, validate, feature_eng, splitter, balance, reduce, merge |
| `test_pipeline.py` | 9  | Pipeline orchestrator, history/undo/reset, Magic Clean |
| `test_api.py` | 24 | All HTTP endpoints via Flask `test_client()` |

## Key regression tests (bugs caught + fixed during dev)

| Test | Bug it pins |
|---|---|
| `test_iqr_cap_handles_int64` | pandas 3.0 nullable `Int64` refusing float clip values |
| `test_fill_mean_handles_int64` | same — float fill into Int64 column |
| `test_standard_scaler_handles_int64` | scaler output is float, Int64 reject |
| `test_responses_contain_no_nan_literal` | `NaN`/`Infinity` in JSON output → invalid JSON in browser |
| `test_preview_op_does_not_commit` | preview endpoint must be dry-run only |
| `test_magic_clean_no_missing_values` | Magic Clean end-to-end correctness |
| `test_upload_recognizes_nan_tokens` | `?`, `N/A`, `null` recognised at CSV load |
