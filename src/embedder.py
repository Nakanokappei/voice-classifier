"""Azure OpenAI Embeddings retrieval with caching.

Responsibilities:
    - Fetch embeddings in parallel batches.
    - Cache per-text results locally (pickle, keyed by SHA-256 of the text).
    - Retry transient errors with exponential backoff.

Shared helpers live in ``utils`` (cache I/O, content hashing, retry wrapper).
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import AzureOpenAI
from tqdm import tqdm

from . import utils

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION: str = "2024-10-21"
BATCH_SIZE: int = 100              # Within OpenAI's recommended batch size.
MAX_CONCURRENCY: int = 8           # Number of parallel batches. Scale with your tier.
MAX_RETRIES: int = 5
BACKOFF_BASE_SEC: float = 2.0


def _default_deployment() -> str:
    """Return the embedding deployment name from the environment."""
    return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")


def get_embeddings(
    texts: list[str],
    cache_dir: Path | str,
    model: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Return the embedding vectors for ``texts``, caching misses on disk.

    Args:
        texts: normalised input texts (duplicates should already be collapsed).
        cache_dir: directory used to persist the embedding cache.
        model: Azure OpenAI embedding deployment name. Defaults to
            ``AZURE_OPENAI_EMBEDDING_DEPLOYMENT`` from the environment.
        api_key: pass an API key explicitly; otherwise read from
            ``AZURE_OPENAI_API_KEY``.

    Returns:
        ``ndarray`` with shape ``(len(texts), D)``, matching the input order.

    Raises:
        RuntimeError: required Azure credentials are missing, or retries were
            exhausted.
        ValueError: deployment name cannot be resolved.
    """
    if not texts:
        raise ValueError("Received an empty text list")

    deployment = model or _default_deployment()
    if not deployment:
        raise ValueError(
            "Azure OpenAI embedding deployment name is not set. "
            "Pass --model or configure AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
        )

    cache_path = _cache_path_for(Path(cache_dir), deployment)
    cache: dict[str, np.ndarray] = utils.load_pickle_cache(cache_path)

    # Only ask the API about texts that aren't already cached.
    missing_texts = [t for t in texts if utils.content_hash(t) not in cache]

    if missing_texts:
        logger.info(
            "Fetching embeddings: %d from API (cache hits: %d)",
            len(missing_texts),
            len(texts) - len(missing_texts),
        )
        client = _make_azure_client(api_key)
        new_vectors = _embed_texts_in_parallel(client, missing_texts, deployment)
        for text, vec in zip(missing_texts, new_vectors, strict=True):
            cache[utils.content_hash(text)] = vec
        utils.save_pickle_cache(cache_path, cache)
    else:
        logger.info("All %d items served from cache", len(texts))

    # Assemble the full array in the original input order.
    vectors = np.stack([cache[utils.content_hash(t)] for t in texts], axis=0)
    return vectors.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_azure_client(api_key: str | None) -> AzureOpenAI:
    """Instantiate an ``AzureOpenAI`` client, honouring env / timeout settings."""
    key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Configure it in .env or the environment."
        )
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. "
            "Configure it in .env or the environment "
            "(e.g. https://<resource>.openai.azure.com)."
        )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
    timeout = float(os.getenv("AZURE_OPENAI_REQUEST_TIMEOUT", "60"))
    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version,
        timeout=timeout,
    )


def _embed_texts_in_parallel(
    client: AzureOpenAI,
    texts: list[str],
    model: str,
) -> list[np.ndarray]:
    """Split ``texts`` into batches and fetch them concurrently.

    The Azure OpenAI client is thread-safe, so several batches can be in
    flight at once. Results are reordered by starting index before being
    returned.
    """
    batches: list[tuple[int, list[str]]] = [
        (start, texts[start : start + BATCH_SIZE])
        for start in range(0, len(texts), BATCH_SIZE)
    ]
    results_by_start: dict[int, list[np.ndarray]] = {}
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_start = {
            executor.submit(_embed_batch, client, batch, model): start
            for start, batch in batches
        }
        with tqdm(total=len(batches), desc="Embeddings", unit="batch") as pbar:
            for future in as_completed(future_to_start):
                start = future_to_start[future]
                with results_lock:
                    results_by_start[start] = future.result()
                pbar.update(1)

    # Concatenate batches in their original order.
    results: list[np.ndarray] = []
    for start, _ in batches:
        results.extend(results_by_start[start])
    return results


def _embed_batch(
    client: AzureOpenAI, batch: list[str], model: str
) -> list[np.ndarray]:
    """Call ``embeddings.create`` for a single batch with retry/backoff."""
    def _call() -> list[np.ndarray]:
        response = client.embeddings.create(model=model, input=batch)
        # response.data is guaranteed to be in input order.
        return [
            np.asarray(item.embedding, dtype=np.float32)
            for item in response.data
        ]

    return utils.call_with_exponential_backoff(
        _call,
        log_prefix="embeddings",
        max_retries=MAX_RETRIES,
        base_sec=BACKOFF_BASE_SEC,
    )


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """Per-model cache file path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize slashes in case of namespaced model names.
    safe_model = model.replace("/", "_")
    return cache_dir / f"embeddings_{safe_model}.pkl"
