import random
import time
from collections.abc import Callable


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
            time.sleep(base_delay * 2**attempt + random.random())
    raise RuntimeError("unreachable")

