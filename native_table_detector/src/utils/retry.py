from __future__ import annotations

import time
from typing import Callable, TypeVar


T = TypeVar("T")


def retry(
    func: Callable[[], T],
    attempts: int = 3,
    wait_seconds: float = 0.5,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - utility function
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc

