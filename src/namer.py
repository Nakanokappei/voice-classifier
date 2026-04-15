"""クラスタ名のLLM自動生成.

責務:
    - 各クラスタの代表テキストを入力に、OpenAI Chat Completions で短いラベルを生成
    - SHA-256 ハッシュをキーにキャッシュ（代表テキストが同じなら再生成しない）
    - 並列リクエストで全クラスタ分を短時間に処理

使い方:
    names = generate_cluster_names(
        summaries=summaries,
        cache_dir="cache",
        model="gpt-4o-mini",
    )
    # names: dict[cluster_id -> "短いラベル"]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm

from .clusterer import ClusterSummary, NOISE_LABEL

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gpt-4o-mini"
MAX_CONCURRENCY: int = 5
MAX_RETRIES: int = 4
BACKOFF_BASE_SEC: float = 2.0

# プロンプト. 日本語で短いラベルを要求
_SYSTEM_PROMPT = (
    "あなたはカスタマーサポート対応履歴を分析する専門家です. "
    "与えられた代表テキスト群から、そのクラスタを最もよく表す短いラベルを1つだけ生成してください. "
    "制約:\n"
    "- 10〜20文字の日本語\n"
    "- 名詞句で終わる（例: 『充電できない』ではなく『充電不良』）\n"
    "- 具体的かつ業務的な用語を優先\n"
    "- 説明文や修飾語は付けない\n"
    "- 絵文字・引用符・余計な記号は使わない"
)


def generate_cluster_names(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[int, str]:
    """各クラスタのラベルを生成して `{cluster_id: name}` を返す.

    Args:
        summaries: `clusterer.summarize_clusters` の戻り値
        cache_dir: 生成済みラベルのキャッシュ先
        model: OpenAI Chat モデル名
        api_key: 明示的に API キーを渡す場合

    Returns:
        cluster_id -> 生成されたラベル. ノイズクラスタは「未分類」固定
    """
    if not summaries:
        return {}

    cache_path = _cache_path_for(Path(cache_dir), model)
    cache: dict[str, str] = _load_cache(cache_path)

    # キャッシュヒット判定
    tasks: list[tuple[int, str, list[str]]] = []  # (cluster_id, key, rep_texts)
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            continue
        if not summary.representative_texts:
            continue
        key = _hash_key(summary.representative_texts)
        tasks.append((summary.cluster_id, key, summary.representative_texts))

    # 未キャッシュ分だけ API 呼び出し
    pending = [(cid, key, texts) for cid, key, texts in tasks if key not in cache]
    if pending:
        logger.info(
            "クラスタ名生成: %d件をAPI取得 (キャッシュヒット: %d件)",
            len(pending),
            len(tasks) - len(pending),
        )
        client = _make_client(api_key)
        new_names = _fetch_parallel(client, pending, model)
        cache.update(new_names)
        _save_cache(cache_path, cache)
    else:
        logger.info("クラスタ名生成: 全 %d 件がキャッシュヒット", len(tasks))

    # 組み立て（ノイズは固定ラベル）
    result: dict[int, str] = {}
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            result[summary.cluster_id] = "未分類"
            continue
        if not summary.representative_texts:
            result[summary.cluster_id] = f"クラスタ #{summary.cluster_id}"
            continue
        key = _hash_key(summary.representative_texts)
        result[summary.cluster_id] = cache.get(
            key, f"クラスタ #{summary.cluster_id}"
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(api_key: str | None) -> OpenAI:
    """OpenAI クライアントを生成."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY が未設定です. .env または環境変数で指定してください"
        )
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    return OpenAI(api_key=key, timeout=timeout)


def _fetch_parallel(
    client: OpenAI,
    pending: list[tuple[int, str, list[str]]],
    model: str,
) -> dict[str, str]:
    """各クラスタのラベルを並列に問い合わせる."""
    results: dict[str, str] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_meta = {
            executor.submit(_request_with_retry, client, texts, model): (cid, key)
            for cid, key, texts in pending
        }
        with tqdm(total=len(pending), desc="Naming", unit="cluster", leave=False) as pbar:
            for future in as_completed(future_to_meta):
                cid, key = future_to_meta[future]
                try:
                    name = future.result()
                except Exception as exc:  # noqa: BLE001 — 名前生成失敗は致命的にしない
                    logger.warning("クラスタ %d のラベル生成に失敗: %s", cid, exc)
                    name = f"クラスタ #{cid}"
                with lock:
                    results[key] = name
                pbar.update(1)
    return results


def _request_with_retry(
    client: OpenAI, rep_texts: list[str], model: str
) -> str:
    """指数バックオフ付きで Chat Completions を呼ぶ."""
    user_prompt = _build_user_prompt(rep_texts)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=30,
            )
            content = response.choices[0].message.content or ""
            return _sanitize_label(content)
        except RateLimitError:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "naming レートリミット (attempt=%d/%d), %.1f秒待機",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except APIError as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "naming APIエラー (attempt=%d/%d): %s, %.1f秒待機",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"クラスタ名生成がリトライ上限 {MAX_RETRIES} を超えました"
    )


def _build_user_prompt(rep_texts: list[str]) -> str:
    """ユーザメッセージを組み立てる."""
    bullets = "\n".join(f"- {t}" for t in rep_texts)
    return f"以下はあるクラスタの代表的な対応記録です:\n{bullets}\n\nこのクラスタのラベルを1つ出力してください."


def _sanitize_label(raw: str) -> str:
    """LLM 出力を短いラベルにクリーニング."""
    label = raw.strip()
    # クォート・引用符を除去
    for ch in ('"', "'", "「", "」", "『", "』", "`"):
        label = label.replace(ch, "")
    # 改行で分割し、最初の非空行のみ
    for line in label.splitlines():
        line = line.strip()
        if line:
            label = line
            break
    # 長すぎる場合は先頭 30 字にトランケート
    if len(label) > 30:
        label = label[:30]
    if not label:
        label = "無題"
    return label


def _hash_key(rep_texts: list[str]) -> str:
    """代表テキスト列を安定キー化."""
    payload = "\n".join(rep_texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """モデル別キャッシュファイルパス."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    return cache_dir / f"cluster_names_{safe_model}.pkl"


def _load_cache(path: Path) -> dict[str, str]:
    """pickle キャッシュを読み込む. 無ければ空辞書."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            logger.warning("名前キャッシュ形式が不正. 新規作成: %s", path)
            return {}
        return cache
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.warning("名前キャッシュ読込失敗 (%s). 破棄して新規作成", exc)
        return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    """pickle キャッシュを保存（原子的に差し替え）. 副次的に JSON 版も残す."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    # デバッグ用途の人間可読 JSON も同ディレクトリに残す
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
