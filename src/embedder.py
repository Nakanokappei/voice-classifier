"""OpenAI Embeddings retrieval — with cache.

Responsibilities:
    - Fetch embeddings in batches.
    - Cache per-text results locally (pickle, keyed by SHA-256 of the text).
    - Retry with exponential backoff on rate-limit/API errors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "text-embedding-3-small"
BATCH_SIZE: int = 100              # Within OpenAI's recommended batch size.
MAX_RETRIES: int = 5
BACKOFF_BASE_SEC: float = 2.0
MAX_CONCURRENCY: int = 8           # Number of parallel batches. Scale with your tier.


def get_embeddings(
    texts: list[str],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> np.ndarray:
    """Return the embedding vectors for the given texts.

    Args:
        texts: normalised input texts
        cache_dir: directory used to persist the embedding cache
        model: OpenAI embedding model name
        api_key: pass an API key explicitly; otherwise read from OPENAI_API_KEY

    Returns:
        `ndarray` with shape `(len(texts), D)`, matching the input order.

    Raises:
        RuntimeError: no API key is available, or retries were exhausted
    """
    if not texts:
        raise ValueError("Received an empty text list")

    cache_path = _cache_path_for(Path(cache_dir), model)
    cache: dict[str, np.ndarray] = _load_cache(cache_path)

    # Only ask the API about texts that aren't already cached.
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for idx, text in enumerate(texts):
        key = _hash_key(text)
        if key not in cache:
            missing_indices.append(idx)
            missing_texts.append(text)

    if missing_texts:
        logger.info(
            "Fetching embeddings: %d from API (cache hits: %d)",
            len(missing_texts),
            len(texts) - len(missing_texts),
        )
        client = _make_client(api_key)
        new_vectors = _fetch_batched(client, missing_texts, model)
        for text, vec in zip(missing_texts, new_vectors, strict=True):
            cache[_hash_key(text)] = vec
        _save_cache(cache_path, cache)
    else:
        logger.info("All %d items served from cache", len(texts))

    # Assemble the full array in the original input order.
    vectors = np.stack([cache[_hash_key(t)] for t in texts], axis=0)
    return vectors.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(api_key: str | None) -> OpenAI:
    """Build an OpenAI client.

    Raises:
        RuntimeError: when no API key can be resolved
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in .env or the environment."
        )
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    return OpenAI(api_key=key, timeout=timeout)


def _fetch_batched(client: OpenAI, texts: list[str], model: str) -> list[np.ndarray]:
    """Send texts to the API in parallel batches.

    Batches are sliced sequentially so ordering is deterministic. The actual
    HTTP requests run concurrently via `ThreadPoolExecutor` (the OpenAI client
    is thread-safe).
    """
    # Pre-slice batches with their starting indices so we can reassemble in order.
    batches: list[tuple[int, list[str]]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batches.append((start, texts[start : start + BATCH_SIZE]))

    # Store results keyed by their start index.
    results_by_start: dict[int, list[np.ndarray]] = {}
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_start = {
            executor.submit(_request_with_retry, client, batch, model): start
            for start, batch in batches
        }
        with tqdm(total=len(batches), desc="Embeddings", unit="batch") as pbar:
            for future in as_completed(future_to_start):
                start = future_to_start[future]
                batch_vectors = future.result()
                with results_lock:
                    results_by_start[start] = batch_vectors
                pbar.update(1)

    # Concatenate batches in their original order.
    results: list[np.ndarray] = []
    for start, _ in batches:
        results.extend(results_by_start[start])
    return results


def _request_with_retry(
    client: OpenAI, batch: list[str], model: str
) -> list[np.ndarray]:
    """Call `embeddings.create` with exponential backoff on retriable errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=model, input=batch)
            # response.data is guaranteed to be in input order.
            return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        except RateLimitError as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "Rate limited (attempt=%d/%d): %s — waiting %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
        except APIError as exc:
            # Retry 5xx / transient API errors too.
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "Transient API error (attempt=%d/%d): %s — waiting %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"OpenAI embeddings retrieval exceeded {MAX_RETRIES} retries"
    )


def _hash_key(text: str) -> str:
    """Stable cache key for a single text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """Per-model cache file path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize slashes in case of namespaced model names.
    safe_model = model.replace("/", "_")
    return cache_dir / f"embeddings_{safe_model}.pkl"


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """Load the pickle cache. Returns an empty dict if missing/corrupt."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            logger.warning("Cache has invalid format, rebuilding: %s", path)
            return {}
        logger.debug("Cache loaded: %d entries", len(cache))
        return cache
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.warning("Cache load failed (%s); discarding and rebuilding", exc)
        return {}


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    """Persist the pickle cache using an atomic replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.debug("Cache saved: %d entries -> %s", len(cache), path)
