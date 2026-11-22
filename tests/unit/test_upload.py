import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from main import app
from src.config import get_settings


@pytest.fixture
def client(tmp_path):
    fake = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        FILE_MAX_SIZE=1,  # 1 MB
        FILE_ALLOWED_TYPES=["txt", "md"],
        APP_NAME="test",
        APP_VERSION="0",
        FILE_DEFAULT_CHUNK_SIZE=200,
        FILE_DEFAULT_OVERLAP_SIZE=20,
    )
    app.dependency_overrides[get_settings] = lambda: fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_upload_ok(client, tmp_path):
    r = client.post("/api/data/upload/proj1", files={"file": ("a.txt", b"hello")})
    assert r.status_code == 200
    assert len(list((tmp_path / "raw" / "proj1").iterdir())) == 1


def test_rejects_wrong_type(client):
    r = client.post("/api/data/upload/proj1", files={"file": ("a.exe", b"x")})
    assert r.status_code == 415


def test_rejects_too_large(client, tmp_path):
    big = b"x" * (2 * 1024 * 1024)
    r = client.post("/api/data/upload/proj1", files={"file": ("a.txt", big)})
    assert r.status_code == 413
    assert not list((tmp_path / "raw" / "proj1").glob("*"))  # no partial file left


def test_filename_traversal_stays_inside_project_dir(client, tmp_path):
    client.post("/api/data/upload/proj1", files={"file": ("../../evil.txt", b"x")})
    assert not (tmp_path / "evil.txt").exists()
    assert len(list((tmp_path / "raw" / "proj1").iterdir())) == 1


def test_rejects_bad_project_id(client):
    r = client.post("/api/data/upload/bad.id", files={"file": ("a.txt", b"x")})
    assert r.status_code == 422


def test_process_unknown_project_gives_404(client):
    assert client.post("/api/data/process/nope").status_code == 404


def test_process_skips_done_files_unless_forced(client):
    client.post("/api/data/upload/p1", files={"file": ("a.txt", b"hello")})
    client.post("/api/data/process/p1")
    assert client.post("/api/data/process/p1").json()["files_skipped"] == 1
    assert client.post("/api/data/process/p1?force=true").json()["files_processed"] == 1


def test_reset_project_processed_keeps_raw(client, tmp_path):
    client.post("/api/data/upload/p1", files={"file": ("a.txt", b"hello")})
    client.post("/api/data/process/p1")
    client.delete("/api/data/p1?scope=processed")
    assert (tmp_path / "raw" / "p1").exists()
    assert not (tmp_path / "processed" / "p1").exists()


def test_reset_all_needs_confirm(client, tmp_path):
    client.post("/api/data/upload/p1", files={"file": ("a.txt", b"hello")})
    assert client.delete("/api/data").status_code == 400
    assert (tmp_path / "raw" / "p1").exists()


def test_reset_all_then_upload_again(client, tmp_path):
    client.post("/api/data/upload/p1", files={"file": ("a.txt", b"hello")})
    assert client.delete("/api/data?confirm=true").status_code == 200
    assert not (tmp_path / "raw").exists()
    assert client.post(
        "/api/data/upload/p1", files={"file": ("b.txt", b"new")}
    ).status_code == 200


def test_info_endpoint(client):
    assert client.get("/api/info").json()["name"] == "test"


def test_chunk_file_is_named_after_raw_file(client, tmp_path):
    stored = client.post(
        "/api/data/upload/p1", files={"file": ("a.txt", b"hello world")}
    ).json()["stored_filename"]
    client.post("/api/data/process/p1")

    chunk_file = tmp_path / "processed" / "p1" / f"{stored}.jsonl"
    assert chunk_file.exists()
    first = json.loads(chunk_file.read_text(encoding="utf-8").splitlines()[0])
    assert first["metadata"]["source"] == stored


def test_orphan_chunk_file_is_removed(client, tmp_path):
    stored = client.post(
        "/api/data/upload/p1", files={"file": ("a.txt", b"hello")}
    ).json()["stored_filename"]
    client.post("/api/data/process/p1")
    client.post("/api/data/upload/p1", files={"file": ("b.txt", b"other")})

    (tmp_path / "raw" / "p1" / stored).unlink()
    r = client.post("/api/data/process/p1").json()

    assert r["files_removed"] == 1
    assert not (tmp_path / "processed" / "p1" / f"{stored}.jsonl").exists()


def test_search_after_index(client):
    client.post(
        "/api/data/upload/p1", files={"file": ("a.txt", b"cats are small animals. " * 20)}
    )
    client.post("/api/data/process/p1")
    assert client.post("/api/data/index/p1").json()["chunks_indexed"] > 0

    r = client.post("/api/search/p1", json={"query": "cats", "top_k": 3})
    assert r.status_code == 200
    assert 1 <= len(r.json()["hits"]) <= 3


def test_search_unindexed_project_is_empty(client):
    r = client.post("/api/search/nope", json={"query": "x"})
    assert r.json() == {"hits": []}


def test_search_rejects_empty_query(client):
    assert client.post("/api/search/p1", json={"query": ""}).status_code == 422


def test_reset_all_removes_vectors(client):
    client.post("/api/data/upload/p1", files={"file": ("a.txt", b"cats " * 50)})
    client.post("/api/data/process/p1")
    client.post("/api/data/index/p1")
    assert client.delete("/api/data?confirm=true").json()["deleted"].count("vectors") == 1
    assert client.post("/api/search/p1", json={"query": "cats"}).json() == {"hits": []}


def test_reset_processed_also_removes_vectors(client):
    client.post("/api/data/upload/p1", files={"file": ("a.txt", b"cats " * 50)})
    client.post("/api/data/process/p1")
    client.post("/api/data/index/p1")
    client.delete("/api/data/p1?scope=processed")
    assert client.post("/api/search/p1", json={"query": "cats"}).json() == {"hits": []}

