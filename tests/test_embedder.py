"""Tests for the embedder module; OpenAI API is mocked out."""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import embedder


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeEmbeddingItem:
    """Mimics the shape of response.data[i] in the OpenAI SDK."""

    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    """Mimics the return shape of embeddings.create()."""

    data: list[_FakeEmbeddingItem]


class _CountingFakeClient:
    """Fakes OpenAI.embeddings.create and records every call.

    A deterministic stub that vectorises each input by character code.
    Thread-safe and countable.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.call_count = 0
        self.requested_texts: list[list[str]] = []
        self._lock = threading.Lock()
        self.embeddings = self  # `.embeddings.create(...)` アクセスのため

    def create(self, model: str, input: list[str]) -> _FakeEmbeddingResponse:
        with self._lock:
            self.call_count += 1
            self.requested_texts.append(list(input))

        # 各テキストに対して、決定論的な D 次元ベクトルを返す
        data = []
        for text in input:
            # 文字コードベースで再現性のあるベクトルを作る
            vec = np.zeros(self.dim, dtype=np.float32)
            for idx, ch in enumerate(text):
                vec[idx % self.dim] += float(ord(ch) % 17)
            data.append(_FakeEmbeddingItem(embedding=vec.tolist()))
        return _FakeEmbeddingResponse(data=data)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_all_cache_hit_skips_api(tmp_path: Path) -> None:
    """When the cache holds everything, the API is never called."""
    texts = ["返品したい", "配送遅延", "商品破損"]
    dim = 4
    model = "test-model"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Pre-populate the cache
    cache = {
        embedder._hash_key(t): np.full(dim, i + 1, dtype=np.float32)
        for i, t in enumerate(texts)
    }
    cache_path = cache_dir / f"embeddings_{model}.pkl"
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)

    fake_client = _CountingFakeClient(dim=dim)
    with patch.object(embedder, "_make_client", return_value=fake_client):
        result = embedder.get_embeddings(texts, cache_dir=cache_dir, model=model)

    # API は呼ばれない
    assert fake_client.call_count == 0
    # Results come out in the original order
    assert result.shape == (3, dim)
    for i, _ in enumerate(texts):
        assert np.allclose(result[i], i + 1)


def test_all_cache_miss_calls_api_and_saves_cache(tmp_path: Path) -> None:
    """Without a cache, the API is called and results are persisted."""
    texts = ["alpha", "beta", "gamma"]
    model = "test-model"
    cache_dir = tmp_path / "cache"

    fake_client = _CountingFakeClient(dim=6)
    with patch.object(embedder, "_make_client", return_value=fake_client):
        result = embedder.get_embeddings(texts, cache_dir=cache_dir, model=model)

    # API が呼ばれて全件取得している
    assert fake_client.call_count >= 1
    # 全 3 件が 1 バッチで一度に送られたはず（BATCH_SIZE=100）
    all_requested = [t for batch in fake_client.requested_texts for t in batch]
    assert sorted(all_requested) == sorted(texts)

    # ファイルに保存された
    cache_path = cache_dir / f"embeddings_{model}.pkl"
    assert cache_path.exists()
    with cache_path.open("rb") as f:
        saved = pickle.load(f)
    assert len(saved) == 3
    for text in texts:
        assert embedder._hash_key(text) in saved

    # Return shape
    assert result.shape == (3, 6)


def test_partial_cache_only_queries_missing(tmp_path: Path) -> None:
    """With partial cache, only the missing texts hit the API."""
    cached_texts = ["text-A", "text-B"]
    new_texts = ["text-C", "text-D"]
    all_texts = cached_texts + new_texts
    dim = 5
    model = "partial"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # A, B だけ事前キャッシュ
    initial = {
        embedder._hash_key(t): np.full(dim, i, dtype=np.float32)
        for i, t in enumerate(cached_texts)
    }
    cache_path = cache_dir / f"embeddings_{model}.pkl"
    with cache_path.open("wb") as f:
        pickle.dump(initial, f)

    fake_client = _CountingFakeClient(dim=dim)
    with patch.object(embedder, "_make_client", return_value=fake_client):
        result = embedder.get_embeddings(all_texts, cache_dir=cache_dir, model=model)

    # 送られたテキストは C, D のみ
    sent = [t for batch in fake_client.requested_texts for t in batch]
    assert set(sent) == set(new_texts)

    # Output ordering matches input ordering
    assert result.shape == (4, dim)
    # First two rows come from the pre-seeded cache (0, 1)
    assert np.allclose(result[0], 0)
    assert np.allclose(result[1], 1)


def test_result_order_preserved_across_parallel_batches(tmp_path: Path) -> None:
    """Parallel batches preserve the original input order."""
    # BATCH_SIZE=100 means 350 texts become 4 batches
    texts = [f"unique-text-{i:04d}" for i in range(350)]
    model = "parallel"
    cache_dir = tmp_path / "cache"
    fake_client = _CountingFakeClient(dim=4)

    with patch.object(embedder, "_make_client", return_value=fake_client):
        result = embedder.get_embeddings(texts, cache_dir=cache_dir, model=model)

    # 少なくとも 2 バッチ（並列動作時は順不同で完了しうる）
    assert fake_client.call_count >= 2
    assert result.shape == (350, 4)

    # 決定論的ベクトルなので、完全一致で順序が保たれているか検証
    for idx, text in enumerate(texts):
        expected = np.zeros(4, dtype=np.float32)
        for j, ch in enumerate(text):
            expected[j % 4] += float(ord(ch) % 17)
        assert np.allclose(result[idx], expected), f"order corruption at row {idx}"


def test_empty_texts_raises(tmp_path: Path) -> None:
    """Empty list raises ValueError."""
    with pytest.raises(ValueError):
        embedder.get_embeddings([], cache_dir=tmp_path, model="x")


def test_missing_api_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing OPENAI_API_KEY raises RuntimeError when no key is passed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # _make_client is not called when everything hits cache,
    # so use an empty cache to force the API call and then fail before it
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        embedder.get_embeddings(["x"], cache_dir=tmp_path, model="y")


def test_corrupted_cache_is_discarded(tmp_path: Path) -> None:
    """A corrupt cache is discarded and rebuilt."""
    model = "corrupt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / f"embeddings_{model}.pkl"
    cache_path.write_bytes(b"not a valid pickle")

    fake_client = _CountingFakeClient(dim=3)
    with patch.object(embedder, "_make_client", return_value=fake_client):
        result = embedder.get_embeddings(["hello"], cache_dir=cache_dir, model=model)

    assert result.shape == (1, 3)
    # A fresh valid cache replaced the corrupt one
    with cache_path.open("rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, dict)
    assert embedder._hash_key("hello") in loaded


def test_rate_limit_triggers_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RateLimitError triggers retry and eventually succeeds."""
    from openai import RateLimitError

    attempts = {"count": 0}

    class _FlakeyClient:
        def __init__(self) -> None:
            self.embeddings = self

        def create(self, model: str, input: list[str]) -> _FakeEmbeddingResponse:
            attempts["count"] += 1
            if attempts["count"] == 1:
                # Only the first attempt is rate-limited
                raise RateLimitError(
                    message="Rate limit",
                    response=MagicMock(status_code=429),
                    body={"error": "rate limit"},
                )
            return _FakeEmbeddingResponse(
                data=[_FakeEmbeddingItem(embedding=[0.1] * 3) for _ in input]
            )

    # Shorten backoff to keep the test fast
    monkeypatch.setattr(embedder, "BACKOFF_BASE_SEC", 0.01)

    with patch.object(embedder, "_make_client", return_value=_FlakeyClient()):
        result = embedder.get_embeddings(
            ["a"], cache_dir=tmp_path / "cache", model="retry"
        )

    assert attempts["count"] == 2  # One failure + one success
    assert result.shape == (1, 3)
