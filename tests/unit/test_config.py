import pytest
from pydantic import ValidationError

from src.config import Settings

BASE = dict(
    APP_NAME="t", APP_VERSION="0", FILE_ALLOWED_TYPES=["txt"], FILE_MAX_SIZE=1,
    DATA_DIR="d", FILE_DEFAULT_CHUNK_SIZE=100, FILE_DEFAULT_OVERLAP_SIZE=10,
    EMBEDDING_BACKEND="gemini", EMBEDDING_MODEL_ID="m", EMBEDDING_MODEL_SIZE=8,
    VECTOR_DB_BACKEND="qdrant", VECTOR_DB_PATH="p",
)


def test_valid_settings():
    assert Settings(**{"_env_file": None, **BASE}).FILE_DEFAULT_OVERLAP_SIZE == 10


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(**{"_env_file": None, **BASE, "FILE_DEFAULT_OVERLAP_SIZE": 100})

