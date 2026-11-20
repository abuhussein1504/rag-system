import hashlib
from collections.abc import Callable
from pathlib import Path

from src.models.schemas import Document

Loader = Callable[[Path, str], list[Document]]
_LOADERS: dict[str, Loader] = {}


class UnsupportedFileError(Exception):
    pass


def register(*extensions: str) -> Callable[[Loader], Loader]:
    def decorator(fn: Loader) -> Loader:
        for ext in extensions:
            _LOADERS[ext] = fn
        return fn

    return decorator


def make_document(
    path: Path,
    project_id: str,
    text: str,
    *,
    key: str | int = 1,
    **metadata: object,
) -> Document:
    doc_id = hashlib.sha256(f"{path.name}:{key}".encode()).hexdigest()[:16]
    return Document(
        id=doc_id,
        project_id=project_id,
        source=path.name,
        text=text,
        metadata={"format": path.suffix.lower().lstrip("."), **metadata},
    )


def _ensure_loaded() -> None:
    from src.core.ingestion import loaders


def supported_extensions() -> set[str]:
    _ensure_loaded()
    return set(_LOADERS)


def load_file(path: Path, project_id: str) -> list[Document]:
    _ensure_loaded()
    ext = path.suffix.lower().lstrip(".")
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileError(ext)
    return loader(path, project_id)

