import re
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

READ_CHUNK_BYTES = 512 * 1024  # 512 KB
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{12}_")


class FileTypeNotSupportedError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


def original_name(stored_name: str) -> str:
    return _UPLOAD_PREFIX.sub("", stored_name)


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def sanitize_stem(filename: str) -> str:
    stem = re.sub(r"[^\w\-]", "_", Path(filename).stem)
    return stem[:50] or "file"


async def save_upload(
    file: UploadFile,
    dest_dir: Path,
    *,
    max_bytes: int,
    allowed_types: set[str],
) -> Path:

    original = file.filename or ""
    ext = get_extension(original)
    if ext not in allowed_types:
        raise FileTypeNotSupportedError(ext)

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{uuid.uuid4().hex[:12]}_{sanitize_stem(original)}.{ext}"

    size = 0
    try:
        async with aiofiles.open(path, "wb") as out:
            while chunk := await file.read(READ_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise FileTooLargeError(max_bytes)
                await out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path

