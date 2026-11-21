import random
import re
import time
from collections.abc import Callable


def _server_retry_delay(exc: Exception) -> float | None:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    return float(match.group(1)) + random.random() if match else None


def retry[T](
    fn: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    attempts: int = 5,
    base_delay: float = 1.0,
) -> T:
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not should_retry(exc):
                raise
            delay = _server_retry_delay(exc) or (base_delay * 2**attempt + random.random())
            time.sleep(delay)
    raise RuntimeError("unreachable")

