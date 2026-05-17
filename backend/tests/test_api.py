"""Integration tests against the Flask app using its test client."""

import io
import json
import zipfile

import pandas as pd
import pytest


# ==========================================================================
# UPLOAD + LOAD
# ==========================================================================

def _upload_csv(client, csv_text: str, filename: str = "test.csv"):
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(csv_text.encode("utf-8")), filename)},
        content_type="multipart/form-data",
    )


def test_index_serves_html(flask_client):
    r = flask_client.get("/")
    assert r.status_code == 200
    assert b"CleanML" in r.data


def test_upload_accepts_csv(flask_client):
    r = _upload_csv(flask_client, "a,b\n1,2\n3,4\n")
    assert r.status_code == 200
    data = r.get_json()
    assert "sid" in data
    assert data["rows"] == 2
    assert data["cols"] == 2


def test_upload_rejects_non_csv(flask_client):
    r = flask_client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"foo"), "x.txt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_upload_recognizes_nan_tokens(flask_client):
    """Mixed NaN tokens should auto-become real NaN on load."""
    csv = "name,age\nAlice,30\nBob,?\nCarol,N/A\nDan,\n"
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    profile = flask_client.get(f"/api/profile/{sid}").get_json()
    age = next(c for c in profile["columns"] if c["name"] == "age")
    assert age["missing"] == 3


# ==========================================================================
# PROFILE
# ==========================================================================

def test_profile_returns_quality_score(flask_client):
    r = _upload_csv(flask_client, "x,y\n1,a\n2,b\n3,c\n4,d\n5,e\n")
    sid = r.get_json()["sid"]
    p = flask_client.get(f"/api/profile/{sid}").get_json()
    assert "quality_score" in p
    assert "grade" in p
    assert "columns" in p


def test_profile_404_for_unknown_session(flask_client):
    r = flask_client.get("/api/profile/no-such-sid")
    assert r.status_code == 404


# ==========================================================================
# SUGGESTIONS
# ==========================================================================

def test_suggest_returns_ranked_list(flask_client):
    csv = ("a,b,const\n" +
           "\n".join(f"{i},x,c" for i in range(20)) +
           "\n,x,c\n,x,c\n")
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    s = flask_client.get(f"/api/suggest/{sid}").get_json()
    assert "suggestions" in s
    assert len(s["suggestions"]) > 0
    impacts = [it["impact"] for it in s["suggestions"]]
    rank = {"high": 3, "medium": 2, "low": 1}
    assert all(rank[impacts[i]] >= rank[impacts[i+1]] for i in range(len(impacts)-1))


# ==========================================================================
# CLEAN + PREVIEW-OP
# ==========================================================================

def test_clean_applies_op_and_updates_history(flask_client):
    r = _upload_csv(flask_client, "x\n1\n2\n3\n")
    sid = r.get_json()["sid"]
    r2 = flask_client.post(f"/api/clean/{sid}", json={
        "family": "missing", "strategy": "drop_column", "column": "x",
    })
    assert r2.status_code == 200
    h = flask_client.get(f"/api/history/{sid}").get_json()
    assert len(h["history"]) == 1


def test_preview_op_does_not_commit(flask_client):
    r = _upload_csv(flask_client, "x,y\n1,a\n2,b\n3,c\n")
    sid = r.get_json()["sid"]
    flask_client.post(f"/api/preview-op/{sid}", json={
        "family": "missing", "strategy": "drop_column", "column": "x",
    })
    # current dataset must still contain 'x'
    p = flask_client.get(f"/api/preview/{sid}").get_json()
    assert "x" in p["columns"]


def test_preview_op_returns_added_removed_cols(flask_client):
    csv = "color\nred\nblue\ngreen\nred\n"
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    r2 = flask_client.post(f"/api/preview-op/{sid}", json={
        "family": "encoders", "strategy": "onehot", "column": "color",
    })
    info = r2.get_json()
    assert "color" in info["removed_cols"]
    assert any("color_" in c for c in info["added_cols"])


# ==========================================================================
# MAGIC CLEAN
# ==========================================================================

def test_magic_clean_endpoint(flask_client, dirty_df):
    csv = dirty_df.to_csv(index=False)
    r = _upload_csv(flask_client, csv, "dirty.csv")
    sid = r.get_json()["sid"]
    r2 = flask_client.post(f"/api/magic/{sid}")
    assert r2.status_code == 200
    data = r2.get_json()
    assert len(data["applied"]) > 0
    assert data["profile"]["quality_score"] >= 60


# ==========================================================================
# UNDO + RESET
# ==========================================================================

def test_undo_reverts_last_op(flask_client):
    r = _upload_csv(flask_client, "x,y\n1,a\n2,b\n3,c\n")
    sid = r.get_json()["sid"]
    flask_client.post(f"/api/clean/{sid}", json={
        "family": "missing", "strategy": "drop_column", "column": "x",
    })
    flask_client.post(f"/api/undo/{sid}")
    p = flask_client.get(f"/api/preview/{sid}").get_json()
    assert "x" in p["columns"]


def test_reset_clears_history(flask_client):
    r = _upload_csv(flask_client, "x\n1\n2\n3\n")
    sid = r.get_json()["sid"]
    flask_client.post(f"/api/clean/{sid}", json={
        "family": "missing", "strategy": "drop_column", "column": "x",
    })
    flask_client.post(f"/api/reset/{sid}")
    h = flask_client.get(f"/api/history/{sid}").get_json()
    assert h["history"] == []


# ==========================================================================
# COLUMN ENDPOINTS (used by Viz)
# ==========================================================================

def test_column_endpoint_returns_numeric_values(flask_client):
    r = _upload_csv(flask_client, "x\n1\n2\n3\n4\n5\n")
    sid = r.get_json()["sid"]
    data = flask_client.get(f"/api/column/{sid}/x").get_json()
    assert data["kind"] == "numeric"
    assert sorted(data["values"]) == [1, 2, 3, 4, 5]


def test_column_endpoint_returns_categorical(flask_client):
    r = _upload_csv(flask_client, "g\na\nb\na\nc\nb\na\n")
    sid = r.get_json()["sid"]
    data = flask_client.get(f"/api/column/{sid}/g").get_json()
    assert data["kind"] == "categorical"
    assert set(data["categories"]) == {"a", "b", "c"}


def test_scatter_same_column_for_x_and_y(flask_client):
    """Regression: scatter(x=Rank, y=Rank) used to crash with AttributeError."""
    csv = "Rank,Score\n" + "\n".join(f"{i},{i*2}" for i in range(10))
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    r2 = flask_client.get(f"/api/scatter/{sid}?x=Rank&y=Rank")
    assert r2.status_code == 200
    data = r2.get_json()
    assert len(data["x_values"]) == len(data["y_values"])
    assert data["x_values"] == data["y_values"]


def test_scatter_different_columns(flask_client):
    csv = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(10))
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    data = flask_client.get(f"/api/scatter/{sid}?x=x&y=y").get_json()
    assert data["x_values"] == list(range(10))
    assert data["y_values"] == [i * 2 for i in range(10)]


def test_correlation_endpoint(flask_client):
    csv = "x,y,z\n" + "\n".join(f"{i},{i*2},{i+1}" for i in range(10))
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    d = flask_client.get(f"/api/correlation/{sid}").get_json()
    assert len(d["columns"]) == 3
    assert len(d["matrix"]) == 3


# ==========================================================================
# LABEL NORMALIZER
# ==========================================================================

def test_label_groups_endpoint(flask_client):
    csv = "g\nMale\nmale\nMALE\nFemale\nfemale\nFEMALE\n"
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    data = flask_client.get(f"/api/label-groups/{sid}/g").get_json()
    assert "groups" in data
    assert len(data["groups"]) >= 2


# ==========================================================================
# SPLIT + DOWNLOAD
# ==========================================================================

def test_split_creates_train_test(flask_client):
    csv = "x,y\n" + "\n".join(f"{i},{i % 2}" for i in range(50))
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    r2 = flask_client.post(f"/api/split/{sid}",
        json={"target": "y", "test_size": 0.2})
    data = r2.get_json()
    assert data["train_rows"] + data["test_rows"] == 50


def test_download_split_returns_zip(flask_client):
    csv = "x,y\n" + "\n".join(f"{i},{i % 2}" for i in range(50))
    r = _upload_csv(flask_client, csv)
    sid = r.get_json()["sid"]
    flask_client.post(f"/api/split/{sid}", json={"target": "y", "test_size": 0.2})
    r2 = flask_client.get(f"/api/download-split/{sid}")
    assert r2.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r2.data))
    assert "train.csv" in zf.namelist()
    assert "test.csv" in zf.namelist()


# ==========================================================================
# DOWNLOAD CSV + NOTEBOOK
# ==========================================================================

def test_download_csv(flask_client):
    r = _upload_csv(flask_client, "x\n1\n2\n3\n")
    sid = r.get_json()["sid"]
    r2 = flask_client.get(f"/api/download/{sid}")
    assert r2.status_code == 200
    assert b"x" in r2.data
    assert b"1" in r2.data


def test_download_notebook(flask_client):
    r = _upload_csv(flask_client, "x\n1\n2\n3\n")
    sid = r.get_json()["sid"]
    flask_client.post(f"/api/clean/{sid}", json={
        "family": "missing", "strategy": "drop_column", "column": "x",
    })
    r2 = flask_client.get(f"/api/notebook/{sid}")
    assert r2.status_code == 200
    nb = json.loads(r2.data)
    assert "cells" in nb
    assert any(c["cell_type"] == "code" for c in nb["cells"])


# ==========================================================================
# JSON SAFETY (NaN-as-null contract)
# ==========================================================================

def test_responses_contain_no_nan_literal(flask_client):
    """Regression: NaN in numeric cols used to produce invalid JSON."""
    csv = "x\n1\n\n3\n"          # row 2 missing → NaN
    r = _upload_csv(flask_client, csv)
    raw = r.data.decode()
    assert "NaN" not in raw      # contract: every NaN becomes JSON null
    sid = r.get_json()["sid"]
    raw2 = flask_client.get(f"/api/preview/{sid}").data.decode()
    assert "NaN" not in raw2


# ==========================================================================
# MERGE
# ==========================================================================

def test_merge_inspect_returns_columns(flask_client):
    other = "id,extra\n1,x\n2,y\n"
    r = flask_client.post("/api/merge/inspect",
        data={"file": (io.BytesIO(other.encode()), "other.csv")},
        content_type="multipart/form-data")
    assert r.status_code == 200
    data = r.get_json()
    assert "id" in data["columns"]
    assert "extra" in data["columns"]


def test_merge_apply_inner_join(flask_client):
    main = _upload_csv(flask_client, "id,name\n1,A\n2,B\n3,C\n")
    sid = main.get_json()["sid"]
    other = "id,extra\n1,x\n2,y\n5,z\n"
    r = flask_client.post(f"/api/merge/{sid}",
        data={
            "file": (io.BytesIO(other.encode()), "other.csv"),
            "left_on": "id", "right_on": "id", "how": "inner",
        }, content_type="multipart/form-data")
    assert r.status_code == 200
    data = r.get_json()
    assert data["rows"] == 2     # only ids 1 and 2 match
