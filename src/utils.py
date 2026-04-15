"""Cross-module helpers: math utilities, cache I/O, and a retry wrapper.

Factoring these out keeps the feature modules focused on domain logic and
removes several pieces of duplicated code:

    - ``l2_normalize`` used to live in both ``tuner`` and ``clusterer``.
    - ``content_hash`` / pickle cache load+save lived in both ``embedder`` and
      ``namer`` with identical semantics.
    - Exponential-backoff retry logic was copy-pasted across four call sites.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from openai import APIError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalise each row of `x` to unit L2 length.

    Zero-norm rows are replaced by the canonical basis vector ``e_0`` so that
    downstream code never divides by zero. The substitution keeps distance
    math well-defined; zero rows would otherwise trigger noisy
    `divide by zero` warnings inside scikit-learn's silhouette / cosine
    distance implementations.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    zero_mask = norms.flatten() == 0
    if zero_mask.any():
        logger.debug(
            "Replacing %d zero-norm rows with e_0 before normalisation",
            int(zero_mask.sum()),
        )
        x = x.copy()
        x[zero_mask, 0] = 1.0
        norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def content_hash(payload: str) -> str:
    """Stable SHA-256 hex digest for the given UTF-8 string."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pickle cache with JSON sidecar
# ---------------------------------------------------------------------------


def load_pickle_cache(path: Path) -> dict:
    """Load a pickle cache file; return an empty dict on any failure.

    Used by every module that persists API results (embeddings, annotations).
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            logger.warning("Cache has invalid format, rebuilding: %s", path)
            return {}
        return cache
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.warning("Cache load failed (%s); rebuilding: %s", exc, path)
        return {}


def save_pickle_cache(
    path: Path,
    cache: dict,
    *,
    also_json: bool = False,
) -> None:
    """Persist `cache` as pickle with an atomic replace.

    When ``also_json`` is True, a ``<stem>.json`` sidecar is written too so the
    contents are human-inspectable. The JSON path has the same parent directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.debug("Cache saved: %d entries -> %s", len(cache), path)

    if also_json:
        json_path = path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Exponential-backoff retry wrapper for OpenAI calls
# ---------------------------------------------------------------------------


# Default retriable error set for OpenAI chat/embedding endpoints.
OPENAI_RETRIABLE_ERRORS: tuple = (RateLimitError, APIError)


def call_with_exponential_backoff(
    fn: Callable[[], T],
    *,
    log_prefix: str,
    max_retries: int = 4,
    base_sec: float = 2.0,
    retriable: tuple = OPENAI_RETRIABLE_ERRORS,
) -> T:
    """Call ``fn()`` with retries on transient OpenAI errors.

    Args:
        fn: zero-argument callable that performs the API request. Typically
            wrapped in a ``lambda`` so the caller can compose the kwargs.
        log_prefix: short label prepended to warning logs (e.g. ``"embeddings"``,
            ``"annotation"``) to identify the call site.
        max_retries: number of attempts before giving up.
        base_sec: backoff base; actual wait is ``base_sec * 2**(attempt-1)``.
        retriable: exceptions that trigger a retry. Non-retriable exceptions
            propagate immediately.

    Returns:
        Whatever ``fn`` returns on the first successful attempt.

    Raises:
        RuntimeError: all retries failed on retriable errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except retriable as exc:
            wait = base_sec * (2 ** (attempt - 1))
            logger.warning(
                "%s: transient error (attempt=%d/%d): %s — waiting %.1fs",
                log_prefix, attempt, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"{log_prefix}: exhausted {max_retries} retries against the API"
    )
