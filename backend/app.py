"""CleanML Flask backend."""

import io
import math
import os
import uuid
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS


class SafeJSONProvider(DefaultJSONProvider):
    """JSON provider that emits null for NaN/Infinity instead of producing invalid JSON."""

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        try:
            return super().dumps(obj, **kwargs)
        except ValueError:
            # fallback: re-sanitize nested NaN/Infinity to None
            import json as _json
            return _json.dumps(_strip_nans(obj), allow_nan=False, default=str)


def _strip_nans(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _strip_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_nans(v) for v in obj]
    return obj

from cleaner import label_norm, magic, pipeline
from cleaner.notebook_export import build_notebook
from cleaner.profiler import profile_dataframe
from cleaner.suggest import suggest as suggest_next
from cleaner.image import loader, profiler as image_profiler, magic as image_magic, export as image_export
from cleaner.image import quality, dedup, transforms, augment, pair

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DATA_DIR = Path(__file__).resolve().parent / "_sessions"
DATA_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 1000
PREVIEW_ROWS = 50

# Extra tokens treated as NaN on CSV load (pandas already handles "", "NA", "NaN", "null", etc.).
EXTRA_NA = [
    "?", "-", "--", "n/a", "N/A", "none", "None", "NONE",
    "missing", "Missing", "MISSING", "unknown", "Unknown", "UNKNOWN",
    "#N/A", "#NULL!", "(null)", ".",
]

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.json = SafeJSONProvider(app)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------- session helpers ----------

def _sdir(sid: str) -> Path:
    return DATA_DIR / sid


def _new_session(df: pd.DataFrame, source: str) -> str:
    sid = uuid.uuid4().hex[:12]
    d = _sdir(sid)
    d.mkdir(parents=True, exist_ok=True)
    df.to_pickle(d / "original.pkl")
    df.to_pickle(d / "current.pkl")
    (d / "source.txt").write_text(source, encoding="utf-8")
    pipeline.save_history(d, [])
    return sid


def _load_current(sid: str) -> pd.DataFrame:
    p = _sdir(sid) / "current.pkl"
    if not p.exists():
        raise FileNotFoundError(sid)
    return pd.read_pickle(p)


def _save_current(sid: str, df: pd.DataFrame) -> None:
    df.to_pickle(_sdir(sid) / "current.pkl")


def _json_safe(v):
    """Convert NaN/Infinity/pd.NA to None so the JSON output is valid."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    # pandas NA, NaT, NumPy NaN-likes
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    # numpy scalar types -> python types
    if hasattr(v, "item"):
        try:
            return _json_safe(v.item())
        except (ValueError, TypeError):
            pass
    # pandas Timestamp etc.
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _preview_payload(df: pd.DataFrame) -> dict:
    head = df.head(PREVIEW_ROWS)
    records = [
        {col: _json_safe(val) for col, val in row.items()}
        for row in head.to_dict(orient="records")
    ]
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "preview": records,
    }


# ---------- static frontend ----------

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@app.get("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.get("/samples/<path:filename>")
def samples_files(filename):
    """Serve bundled sample datasets (e.g. demo_images.zip)."""
    return send_from_directory(str(SAMPLES_DIR), filename)


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


# ---------- upload + load ----------

@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify(error="no file"), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify(error="only .csv files supported"), 400
    try:
        df = pd.read_csv(f.stream, na_values=EXTRA_NA, keep_default_na=True)
    except Exception as e:
        return jsonify(error=f"failed to parse CSV: {e}"), 400
    sid = _new_session(df, f.filename)
    return jsonify(sid=sid, **_preview_payload(df))


@app.post("/api/load-url")
def load_url():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify(error="missing url"), 400
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), na_values=EXTRA_NA, keep_default_na=True)
    except Exception as e:
        return jsonify(error=f"failed to fetch CSV: {e}"), 400
    sid = _new_session(df, url.split("/")[-1] or "remote.csv")
    return jsonify(sid=sid, **_preview_payload(df))


@app.get("/api/preview/<sid>")
def preview(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    return jsonify(**_preview_payload(df))


# ---------- profile ----------

@app.get("/api/profile/<sid>")
def profile(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    return jsonify(profile_dataframe(df))


@app.get("/api/label-groups/<sid>/<column>")
def label_groups(sid, column):
    """Detect inconsistent-label clusters in a categorical column."""
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    if column not in df.columns:
        return jsonify(error="column not found"), 404
    threshold = float(request.args.get("threshold", 0.85))
    groups = label_norm.detect_groups(df[column], threshold=threshold)
    return jsonify(column=column, threshold=threshold, groups=groups)


@app.get("/api/suggest/<sid>")
def suggest(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    return jsonify(suggestions=suggest_next(df))


@app.get("/api/column/<sid>/<column>")
def column_values(sid, column):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    if column not in df.columns:
        return jsonify(error="column not found"), 404
    s = df[column].dropna()
    n = int((df.shape[0]) - s.shape[0])
    if pd.api.types.is_numeric_dtype(s):
        sample = s.sample(n=min(5000, len(s)), random_state=0) if len(s) > 5000 else s
        return jsonify(
            column=column, kind="numeric", missing=n,
            values=[_json_safe(v) for v in sample.tolist()],
        )
    if pd.api.types.is_datetime64_any_dtype(s):
        sample = s.head(5000).astype(str).tolist()
        return jsonify(column=column, kind="datetime", missing=n, values=sample)
    vc = s.astype(str).value_counts().head(30)
    return jsonify(
        column=column, kind="categorical", missing=n,
        categories=vc.index.tolist(), counts=[int(v) for v in vc.values],
    )


@app.get("/api/correlation/<sid>")
def correlation(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return jsonify(columns=[], matrix=[])
    corr = num.corr().fillna(0)
    return jsonify(
        columns=list(corr.columns),
        matrix=[[_json_safe(v) for v in row] for row in corr.values.tolist()],
    )


@app.get("/api/scatter/<sid>")
def scatter(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    x = request.args.get("x")
    y = request.args.get("y")
    if not (x and y) or x not in df.columns or y not in df.columns:
        return jsonify(error="missing x or y columns"), 400
    # Pull each column as its own Series so x == y is safe (df[[x, y]] would
    # create a DataFrame with duplicate column names, breaking sub[x].tolist()).
    sx = df[x].dropna()
    sy = df[y].dropna()
    common = sx.index.intersection(sy.index)
    xs = sx.loc[common]
    ys = sy.loc[common]
    if len(xs) > 3000:
        sample_idx = xs.sample(n=3000, random_state=0).index
        xs = xs.loc[sample_idx]
        ys = ys.loc[sample_idx]
    return jsonify(
        x=x, y=y,
        x_values=[_json_safe(v) for v in xs.tolist()],
        y_values=[_json_safe(v) for v in ys.tolist()],
    )


# ---------- preview op (dry-run, no commit) ----------

@app.post("/api/preview-op/<sid>")
def preview_op(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404

    op = request.get_json(silent=True) or {}
    column = op.get("column")
    family = op.get("family")
    strategy = op.get("strategy")

    info = {"op": op}

    # current column snapshot
    if column and column in df.columns:
        s = df[column]
        info["column_info"] = {
            "name": column,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "rows": int(len(s)),
        }
        # show top values + a sample of affected rows
        top = s.dropna().astype(str).value_counts().head(10)
        info["top_values"] = [{"value": k, "count": int(v)} for k, v in top.items()]

        missing_idx = s[s.isna()].head(8).index.tolist()
        info["sample_missing_rows"] = [
            {"row": int(i), "values": {c: _json_safe(df.loc[i, c]) for c in df.columns}}
            for i in missing_idx
        ]

        # outlier sample (numeric only)
        if pd.api.types.is_numeric_dtype(s) and family == "outliers":
            sn = pd.to_numeric(s, errors="coerce").dropna()
            if len(sn) >= 4:
                q1, q3 = sn.quantile(0.25), sn.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = sn[(sn < lo) | (sn > hi)]
                info["outlier_bounds"] = {"lo": _json_safe(lo), "hi": _json_safe(hi)}
                info["outlier_sample"] = [
                    {"row": int(i), "value": _json_safe(v)}
                    for i, v in outliers.head(10).items()
                ]
                info["outlier_count"] = int(len(outliers))

    # dry-run the op
    try:
        new_df, code, message = pipeline.apply_op(df.copy(), op)
        info["message"] = message
        info["code"] = code

        # diff columns (added/removed)
        before_cols = list(df.columns)
        after_cols = list(new_df.columns)
        info["added_cols"] = [c for c in after_cols if c not in before_cols]
        info["removed_cols"] = [c for c in before_cols if c not in after_cols]

        # row count change
        info["rows_before"] = int(len(df))
        info["rows_after"] = int(len(new_df))

        # result preview — first 8 rows of either the touched column(s) or added cols
        focus_cols = info["added_cols"] or ([column] if column and column in new_df.columns else list(new_df.columns)[:6])
        info["result_preview"] = [
            {c: _json_safe(new_df[c].iloc[i]) for c in focus_cols if c in new_df.columns}
            for i in range(min(8, len(new_df)))
        ]
    except Exception as e:
        info["error"] = str(e)

    return jsonify(info)


# ---------- single op ----------

@app.post("/api/clean/<sid>")
def clean(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    op = request.get_json(silent=True) or {}
    try:
        df_new, code, message = pipeline.apply_op(df, op)
    except Exception as e:
        return jsonify(error=str(e)), 400
    _save_current(sid, df_new)
    pipeline.append_history(_sdir(sid), {"op": op, "code": code, "message": message})
    return jsonify(message=message, code=code, **_preview_payload(df_new))


# ---------- magic clean ----------

@app.post("/api/magic/<sid>")
def magic_clean(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    df_new, entries = magic.run(df)
    _save_current(sid, df_new)
    h = pipeline.load_history(_sdir(sid))
    for e in entries:
        h.append(e)
    pipeline.save_history(_sdir(sid), h)
    return jsonify(
        applied=entries,
        profile=profile_dataframe(df_new),
        **_preview_payload(df_new),
    )


# ---------- history / undo / reset ----------

@app.get("/api/history/<sid>")
def history(sid):
    return jsonify(history=pipeline.load_history(_sdir(sid)))


@app.post("/api/undo/<sid>")
def undo(sid):
    try:
        df, h = pipeline.undo_last(_sdir(sid))
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    return jsonify(history=h, **_preview_payload(df))


@app.post("/api/reset/<sid>")
def reset(sid):
    try:
        df = pipeline.reset(_sdir(sid))
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    return jsonify(**_preview_payload(df))


# ---------- before/after compare ----------

@app.get("/api/compare/<sid>")
def compare(sid):
    d = _sdir(sid)
    if not (d / "original.pkl").exists():
        return jsonify(error="session not found"), 404
    before = pd.read_pickle(d / "original.pkl")
    after = pd.read_pickle(d / "current.pkl")
    return jsonify(
        before=profile_dataframe(before),
        after=profile_dataframe(after),
        history=pipeline.load_history(d),
    )


# ---------- merge with a second CSV ----------

def _read_other(req) -> tuple[pd.DataFrame, str]:
    """Read the 'other' CSV either from form-data file or from a posted URL."""
    if "file" in req.files:
        f = req.files["file"]
        df = pd.read_csv(f.stream, na_values=EXTRA_NA, keep_default_na=True)
        return df, f.filename
    url = (req.form.get("url") or "").strip()
    if not url:
        raise ValueError("missing file or url")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=EXTRA_NA, keep_default_na=True)
    return df, url.split("/")[-1] or "other.csv"


@app.post("/api/merge/inspect")
def merge_inspect():
    """Return the second CSV's columns + a small preview so the UI can pick join keys."""
    try:
        other, source = _read_other(request)
    except Exception as e:
        return jsonify(error=str(e)), 400
    head = other.head(20)
    return jsonify(
        source=source,
        rows=int(len(other)),
        cols=int(other.shape[1]),
        columns=list(other.columns),
        dtypes={c: str(t) for c, t in other.dtypes.items()},
        preview=[
            {c: _json_safe(v) for c, v in row.items()}
            for row in head.to_dict(orient="records")
        ],
    )


@app.post("/api/merge/<sid>")
def merge_apply(sid):
    """Perform the join and replace the current dataset."""
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    try:
        other, source = _read_other(request)
    except Exception as e:
        return jsonify(error=str(e)), 400

    left_on = request.form.get("left_on")
    right_on = request.form.get("right_on")
    how = request.form.get("how", "inner")

    op = {
        "family": "merge", "strategy": "join",
        "params": {
            "other_df": other, "left_on": left_on, "right_on": right_on,
            "how": how, "source_label": source,
        },
    }
    try:
        new_df, code, message = pipeline.apply_op(df, op)
    except Exception as e:
        return jsonify(error=str(e)), 400

    _save_current(sid, new_df)
    # Store a serializable history entry (drop other_df from params before saving).
    history_op = {**op, "params": {
        "left_on": left_on, "right_on": right_on,
        "how": how, "source_label": source,
    }}
    pipeline.append_history(_sdir(sid), {"op": history_op, "code": code, "message": message})

    return jsonify(message=message, code=code, **_preview_payload(new_df))


# ---------- split downloads ----------

@app.post("/api/split/<sid>")
def split_create(sid):
    """Run train/test split — stores train.pkl + test.pkl in session dir."""
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    payload = request.get_json(silent=True) or {}
    op = {
        "family": "splitter", "strategy": "split",
        "params": {**payload, "session_dir": str(_sdir(sid))},
    }
    try:
        _, code, message = pipeline.apply_op(df, op)
    except Exception as e:
        return jsonify(error=str(e)), 400
    pipeline.append_history(_sdir(sid), {"op": op, "code": code, "message": message})

    train_df = pd.read_pickle(_sdir(sid) / "train.pkl")
    test_df = pd.read_pickle(_sdir(sid) / "test.pkl")
    return jsonify(
        message=message, code=code,
        train_rows=int(len(train_df)), test_rows=int(len(test_df)),
    )


@app.get("/api/download-split/<sid>")
def download_split(sid):
    """Zip train.csv + test.csv for download."""
    import zipfile
    d = _sdir(sid)
    if not (d / "train.pkl").exists() or not (d / "test.pkl").exists():
        return jsonify(error="run /api/split first"), 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        train_df = pd.read_pickle(d / "train.pkl")
        test_df = pd.read_pickle(d / "test.pkl")
        zf.writestr("train.csv", train_df.to_csv(index=False))
        zf.writestr("test.csv", test_df.to_csv(index=False))
    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip", as_attachment=True,
        download_name=f"train_test_{sid}.zip",
    )


# ---------- exports ----------

@app.get("/api/download/<sid>")
def download(sid):
    try:
        df = _load_current(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf, mimetype="text/csv", as_attachment=True,
        download_name=f"cleaned_{sid}.csv",
    )


@app.get("/api/notebook/<sid>")
def notebook(sid):
    d = _sdir(sid)
    if not (d / "original.pkl").exists():
        return jsonify(error="session not found"), 404
    source = (d / "source.txt").read_text(encoding="utf-8") if (d / "source.txt").exists() else "data.csv"
    h = pipeline.load_history(d)
    nb_bytes = build_notebook(source, h)
    return send_file(
        io.BytesIO(nb_bytes), mimetype="application/x-ipynb+json",
        as_attachment=True, download_name=f"cleanml_{sid}.ipynb",
    )


# ---------- IMAGE DATASET ENDPOINTS ----------

@app.post("/api/image/upload")
def image_upload():
    """Upload a ZIP file containing images (and optionally labels.csv)."""
    if "file" not in request.files:
        return jsonify(error="no file"), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".zip"):
        return jsonify(error="only .zip files supported"), 400
    
    try:
        # Save ZIP temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            f.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        # Create session
        sid = uuid.uuid4().hex[:12]
        sdir = _sdir(sid)
        sdir.mkdir(parents=True, exist_ok=True)
        
        # Load images from ZIP
        df, code, message = loader.load_from_zip(tmp_path, sdir)
        
        # Clean up temp file
        tmp_path.unlink()
        
        # Save as original and current
        df.to_pickle(sdir / "original.pkl")
        df.to_pickle(sdir / "current.pkl")
        (sdir / "source.txt").write_text(f.filename, encoding="utf-8")
        pipeline.save_history(sdir, [])
        
        # Profile the dataset
        profile = image_profiler.profile_images(df, sdir)
        
        return jsonify(
            sid=sid,
            message=message,
            rows=int(len(df)),
            profile=profile
        )
    except Exception as e:
        return jsonify(error=f"failed to load ZIP: {e}"), 400


@app.get("/api/image/profile/<sid>")
def image_profile(sid):
    """Get profile information for an image dataset."""
    try:
        df = _load_current(sid)
        sdir = _sdir(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    
    profile = image_profiler.profile_images(df, sdir)
    return jsonify(profile)


@app.post("/api/image/clean/<sid>")
def image_clean(sid):
    """Apply a cleaning operation to an image dataset."""
    try:
        df = _load_current(sid)
        sdir = _sdir(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    
    op = request.get_json(silent=True) or {}
    family = op.get("family", "")
    # Strip "image_" prefix if present
    if family.startswith("image_"):
        family = family[6:]
    strategy = op.get("strategy")
    params = op.get("params", {})
    
    try:
        # Route to appropriate image module
        if family == "quality":
            df_new, code, message = quality.apply(df, None, strategy, params, session_dir=sdir)
        elif family == "dedup":
            df_new, code, message = dedup.apply(df, None, strategy, params, session_dir=sdir)
        elif family == "transforms":
            df_new, code, message = transforms.resize(df, sdir, params) if strategy == "resize" else \
                                    transforms.convert_color(df, sdir, params) if strategy == "convert_color" else \
                                    transforms.normalize(df, sdir, params) if strategy == "normalize" else \
                                    transforms.center_crop(df, sdir, params)
        elif family == "augment":
            df_new, code, message = augment.rotate(df, sdir, params) if strategy == "rotate" else \
                                    augment.flip(df, sdir, params) if strategy == "flip" else \
                                    augment.adjust_brightness(df, sdir, params) if strategy == "brightness" else \
                                    augment.adjust_contrast(df, sdir, params) if strategy == "contrast" else \
                                    augment.random_crop(df, sdir, params)
        elif family == "pair":
            df_new, code, message = pair.join_with_labels(df, sdir, params) if strategy == "join_labels" else \
                                    pair.filter_by_label(df, sdir, params) if strategy == "filter_by_label" else \
                                    pair.balance_by_label(df, sdir, params) if strategy == "balance_classes" else \
                                    pair.split_by_label(df, sdir, params)
        else:
            return jsonify(error=f"unknown family: {family}"), 400
        
        _save_current(sid, df_new)
        pipeline.append_history(sdir, {"op": op, "code": code, "message": message})
        
        return jsonify(
            success=True,
            message=message,
            code=code,
            rows=int(len(df_new)),
            cols=int(df_new.shape[1])
        )
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400


@app.post("/api/image/magic/<sid>")
def image_magic_endpoint(sid):
    """Apply magic cleaning to an image dataset."""
    try:
        df = _load_current(sid)
        sdir = _sdir(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404

    try:
        df_new, operations = image_magic.run(df, sdir)
        _save_current(sid, df_new)

        # Append all operations to history
        h = pipeline.load_history(sdir)
        for op in operations:
            h.append(op)
        pipeline.save_history(sdir, h)

        # Get updated profile
        profile = image_profiler.profile_images(df_new, sdir)

        return jsonify(
            applied=operations,
            summary=image_magic.get_summary(operations),
            profile=profile,
            rows=int(len(df_new))
        )
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/image/export/<sid>")
def image_export_endpoint(sid):
    """Export an image dataset to ZIP, NumPy, or PyTorch format."""
    try:
        df = _load_current(sid)
        sdir = _sdir(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    
    payload = request.get_json(silent=True) or {}
    export_format = payload.get("format", "zip")
    include_labels = payload.get("include_labels", False)
    do_split = payload.get("split", False)
    
    try:
        # Build params for export functions
        params = {
            "include_labels": include_labels
        }
        
        if do_split:
            params["test_size"] = payload.get("test_size", 0.2)
            params["target_col"] = payload.get("target_col", "label")
        
        # Create output file
        if do_split:
            # Export train/test split
            output_dir = sdir / "export"
            success, message = image_export.export_split(df, sdir, output_dir, params)
            
            if not success:
                return jsonify(error=message), 400
            
            # Zip the output directory
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(output_dir))
            buf.seek(0)
            
            return send_file(
                buf, mimetype="application/zip", as_attachment=True,
                download_name=f"image_split_{sid}.zip"
            )
        else:
            # Single file export
            if export_format == "zip":
                output_path = sdir / f"export_{sid}.zip"
                success, message = image_export.export_to_zip(df, sdir, output_path, params)
                mimetype = "application/zip"
            elif export_format == "numpy":
                output_path = sdir / f"export_{sid}.npy"
                success, message = image_export.export_to_numpy(df, sdir, output_path, params)
                mimetype = "application/octet-stream"
            elif export_format == "pytorch":
                output_path = sdir / f"export_{sid}.pt"
                success, message = image_export.export_to_pytorch(df, sdir, output_path, params)
                mimetype = "application/octet-stream"
            else:
                return jsonify(error=f"unknown format: {export_format}"), 400
            
            if not success:
                return jsonify(error=message), 400
            
            return send_file(
                output_path, mimetype=mimetype, as_attachment=True,
                download_name=output_path.name
            )
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/image/thumbnail/<sid>/<image_id>")
def image_thumbnail(sid, image_id):
    """Serve a thumbnail (200px longest edge) for an image."""
    try:
        df = _load_current(sid)
        sdir = _sdir(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    
    # Find image by image_id
    row = df[df["image_id"] == image_id]
    if row.empty:
        return jsonify(error="image not found"), 404
    
    img_path = sdir / row.iloc[0]["path"]
    if not img_path.exists():
        return jsonify(error="image file not found"), 404
    
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            # Create thumbnail (200px longest edge)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            # Save to buffer
            buf = io.BytesIO()
            img_format = img.format or "PNG"
            img.save(buf, format=img_format)
            buf.seek(0)
            
            # Determine mimetype
            mimetype = f"image/{img_format.lower()}"
            if img_format.upper() == "JPEG":
                mimetype = "image/jpeg"
            
            response = send_file(buf, mimetype=mimetype)
            response.headers["Cache-Control"] = "max-age=3600"
            return response
    except Exception as e:
        return jsonify(error=f"failed to generate thumbnail: {e}"), 500


@app.get("/api/image/full/<sid>/<image_id>")
def image_full(sid, image_id):
    """Serve the full-resolution image."""
    try:
        df = _load_current(sid)
        sdir = _sdir(sid)
    except FileNotFoundError:
        return jsonify(error="session not found"), 404
    
    # Find image by image_id
    row = df[df["image_id"] == image_id]
    if row.empty:
        return jsonify(error="image not found"), 404
    
    img_path = sdir / row.iloc[0]["path"]
    if not img_path.exists():
        return jsonify(error="image file not found"), 404
    
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            img_format = img.format or "PNG"
            
            # Determine mimetype
            mimetype = f"image/{img_format.lower()}"
            if img_format.upper() == "JPEG":
                mimetype = "image/jpeg"
            
            response = send_file(img_path, mimetype=mimetype)
            response.headers["Cache-Control"] = "max-age=3600"
            return response
    except Exception as e:
        return jsonify(error=f"failed to serve image: {e}"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
