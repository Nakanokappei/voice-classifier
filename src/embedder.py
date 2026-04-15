"""OpenAI Embeddings 取得 — キャッシュ付き.

責務:
    - テキストの埋め込みをバッチで取得
    - SHA-256 ハッシュをキーにローカルキャッシュ（pickle）
    - レートリミット時は指数バックオフでリトライ
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "text-embedding-3-small"
BATCH_SIZE: int = 100              # OpenAI の推奨バッチ上限の範囲内
MAX_RETRIES: int = 5
BACKOFF_BASE_SEC: float = 2.0


def get_embeddings(
    texts: list[str],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> np.ndarray:
    """テキストリストの埋め込みベクトルを返す.

    Args:
        texts: 正規化済みテキスト列
        cache_dir: キャッシュ保存ディレクトリ
        model: 使用する OpenAI 埋め込みモデル
        api_key: OPENAI_API_KEY を明示する場合

    Returns:
        ndarray shape=(len(texts), D) — `texts` と同じ順序

    Raises:
        RuntimeError: APIキー未設定、またはリトライ上限超過
    """
    if not texts:
        raise ValueError("空のテキストリストが渡されました")

    cache_path = _cache_path_for(Path(cache_dir), model)
    cache: dict[str, np.ndarray] = _load_cache(cache_path)

    # まずキャッシュに無いテキストだけを API に問い合わせ
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for idx, text in enumerate(texts):
        key = _hash_key(text)
        if key not in cache:
            missing_indices.append(idx)
            missing_texts.append(text)

    if missing_texts:
        logger.info(
            "埋め込みをAPI取得: %d件 (キャッシュヒット: %d件)",
            len(missing_texts),
            len(texts) - len(missing_texts),
        )
        client = _make_client(api_key)
        new_vectors = _fetch_batched(client, missing_texts, model)
        for text, vec in zip(missing_texts, new_vectors, strict=True):
            cache[_hash_key(text)] = vec
        _save_cache(cache_path, cache)
    else:
        logger.info("全 %d 件がキャッシュヒット", len(texts))

    # キャッシュから全件を取り出して ndarray に組み立てる
    vectors = np.stack([cache[_hash_key(t)] for t in texts], axis=0)
    return vectors.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(api_key: str | None) -> OpenAI:
    """OpenAI クライアントを生成する.

    Raises:
        RuntimeError: APIキーが取得できない場合
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY が未設定です. .env または環境変数で指定してください"
        )
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    return OpenAI(api_key=key, timeout=timeout)


def _fetch_batched(client: OpenAI, texts: list[str], model: str) -> list[np.ndarray]:
    """テキストをバッチに分割して API 呼び出し."""
    results: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Embeddings",
        unit="batch",
    ):
        batch = texts[start : start + BATCH_SIZE]
        batch_vectors = _request_with_retry(client, batch, model)
        results.extend(batch_vectors)
    return results


def _request_with_retry(
    client: OpenAI, batch: list[str], model: str
) -> list[np.ndarray]:
    """指数バックオフ付きで embeddings.create を呼ぶ."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=model, input=batch)
            # response.data は入力順に並ぶ保証がある
            return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        except RateLimitError as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "レートリミット (attempt=%d/%d): %s — %.1f秒待機",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
        except APIError as exc:
            # 5xx など一時エラーもリトライ
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "API一時エラー (attempt=%d/%d): %s — %.1f秒待機",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"OpenAI embeddings 取得がリトライ上限 {MAX_RETRIES} を超えました")


def _hash_key(text: str) -> str:
    """テキストを安定キー化."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """モデル別キャッシュファイルパス."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # モデル名にスラッシュが入らない前提. 念のためサニタイズ
    safe_model = model.replace("/", "_")
    return cache_dir / f"embeddings_{safe_model}.pkl"


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """pickle キャッシュを読み込む. 無ければ空辞書."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            logger.warning("キャッシュ形式が不正です. 新規作成します: %s", path)
            return {}
        logger.debug("キャッシュ読込: %d エントリ", len(cache))
        return cache
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.warning("キャッシュ読込失敗 (%s). 破棄して新規作成します", exc)
        return {}


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    """pickle キャッシュを保存（原子的に差し替え）."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.debug("キャッシュ保存: %d エントリ -> %s", len(cache), path)
