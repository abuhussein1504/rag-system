from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from main import app
from src.utils.config import get_settings


@pytest.fixture
def client(tmp_path):
    fake = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        FILE_MAX_SIZE=1,  # 1 MB
        FILE_ALLOWED_TYPES=["txt", "md"],
        APP_NAME="test",
        APP_VERSION="0",
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

