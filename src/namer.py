"""クラスタのLLM自動アノテーション (短いラベル + 要約テキスト).

責務:
    - 各クラスタの代表テキストを入力に、OpenAI Chat Completions でラベルと
      要約テキストを **1 回の呼び出しで両方** 取得する（JSON モード活用）
    - SHA-256 ハッシュをキーにキャッシュ（代表テキストが同じなら再生成しない）
    - 並列リクエストで全クラスタ分を短時間に処理

label: 10〜20字の短い名詞句. 表や見出しで使う
summary: 1〜3文の要約. レポート本文の「代表テキスト」として使う

使い方:
    annotations = generate_cluster_annotations(
        summaries=summaries,
        cache_dir="cache",
        model="gpt-4o-mini",
    )
    # annotations[cluster_id] -> ClusterAnnotation(label=..., summary=...)
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
from dataclasses import dataclass
from pathlib import Path

from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm

from .clusterer import ClusterSummary, NOISE_LABEL

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gpt-4o-mini"
MAX_CONCURRENCY: int = 5
MAX_RETRIES: int = 4
BACKOFF_BASE_SEC: float = 2.0

# システムプロンプト. label と summary を JSON で同時に要求
_SYSTEM_PROMPT = (
    "あなたはカスタマーサポート対応履歴を分析する専門家です. "
    "与えられた代表テキスト群からクラスタを特徴付ける短いラベルと、"
    "そのクラスタを説明する要約を JSON で返してください.\n\n"
    "出力スキーマ（厳守）:\n"
    "{\n"
    '  "label": "<10〜20 文字の日本語. 名詞句. 具体的・業務的な用語>",\n'
    '  "summary": "<代表テキストを総合した 1〜3 文の日本語. 問い合わせ内容の傾向を述べる>"\n'
    "}\n\n"
    "制約:\n"
    "- label は修飾語・絵文字・引用符・余計な記号を含めない\n"
    "- summary は代表テキストから読み取れる事実のみを記述し、推測・助言は避ける\n"
    "- 入力が英語でもラベル・要約は日本語で返す\n"
    "- 必ず有効な JSON のみを返す（余計な文字・コードブロック囲みは厳禁）"
)


@dataclass
class ClusterAnnotation:
    """LLM が生成したクラスタのアノテーション.

    Attributes:
        label: 10〜20 字の短いラベル（cluster_name として利用）
        summary: クラスタの代表として表示する 1〜3 文の要約テキスト
    """

    label: str
    summary: str


def generate_cluster_annotations(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[int, ClusterAnnotation]:
    """各クラスタに対して label と summary をまとめて生成.

    Args:
        summaries: `clusterer.summarize_clusters` の戻り値
        cache_dir: 生成済みアノテーションのキャッシュ先
        model: OpenAI Chat モデル名
        api_key: 明示的に API キーを渡す場合

    Returns:
        cluster_id -> ClusterAnnotation. ノイズクラスタは「未分類」固定ラベル
    """
    if not summaries:
        return {}

    cache_path = _cache_path_for(Path(cache_dir), model)
    cache: dict[str, dict[str, str]] = _load_cache(cache_path)

    # キャッシュヒット判定
    tasks: list[tuple[int, str, list[str]]] = []  # (cluster_id, key, rep_texts)
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            continue
        if not summary.representative_texts:
            continue
        key = _hash_key(summary.representative_texts)
        tasks.append((summary.cluster_id, key, summary.representative_texts))

    pending = [(cid, key, texts) for cid, key, texts in tasks if key not in cache]
    if pending:
        logger.info(
            "クラスタアノテーション生成: %d件をAPI取得 (キャッシュヒット: %d件)",
            len(pending),
            len(tasks) - len(pending),
        )
        client = _make_client(api_key)
        new_annotations = _fetch_parallel(client, pending, model)
        cache.update(new_annotations)
        _save_cache(cache_path, cache)
    else:
        logger.info("クラスタアノテーション生成: 全 %d 件がキャッシュヒット", len(tasks))

    # 組み立て
    result: dict[int, ClusterAnnotation] = {}
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            result[summary.cluster_id] = ClusterAnnotation(
                label="未分類",
                summary="密度の低い領域に位置し、いずれのクラスタにも属さなかったサンプル.",
            )
            continue
        if not summary.representative_texts:
            result[summary.cluster_id] = ClusterAnnotation(
                label=f"クラスタ #{summary.cluster_id}",
                summary="（代表テキストなし）",
            )
            continue
        key = _hash_key(summary.representative_texts)
        entry = cache.get(key)
        if entry is None:
            result[summary.cluster_id] = ClusterAnnotation(
                label=f"クラスタ #{summary.cluster_id}",
                summary="（生成失敗）",
            )
        else:
            result[summary.cluster_id] = ClusterAnnotation(
                label=entry.get("label", f"クラスタ #{summary.cluster_id}"),
                summary=entry.get("summary", ""),
            )

    return result


def generate_cluster_names(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[int, str]:
    """後方互換: cluster_id -> label のみ返すラッパ.

    内部では `generate_cluster_annotations` を呼び、summary も同時生成される.
    """
    annotations = generate_cluster_annotations(summaries, cache_dir, model, api_key)
    return {cid: ann.label for cid, ann in annotations.items()}


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
) -> dict[str, dict[str, str]]:
    """各クラスタの label/summary を並列に問い合わせる."""
    results: dict[str, dict[str, str]] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_meta = {
            executor.submit(_request_with_retry, client, texts, model): (cid, key)
            for cid, key, texts in pending
        }
        with tqdm(
            total=len(pending), desc="Annotating", unit="cluster", leave=False
        ) as pbar:
            for future in as_completed(future_to_meta):
                cid, key = future_to_meta[future]
                try:
                    annotation = future.result()
                except Exception as exc:  # noqa: BLE001 — 生成失敗は致命的にしない
                    logger.warning(
                        "クラスタ %d のアノテーション生成に失敗: %s", cid, exc
                    )
                    annotation = {
                        "label": f"クラスタ #{cid}",
                        "summary": "（生成失敗）",
                    }
                with lock:
                    results[key] = annotation
                pbar.update(1)
    return results


def _request_with_retry(
    client: OpenAI, rep_texts: list[str], model: str
) -> dict[str, str]:
    """指数バックオフ付きで Chat Completions を呼び、label+summary の JSON を取得."""
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
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return _parse_annotation_json(content)
        except RateLimitError:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "annotation レートリミット (attempt=%d/%d), %.1f秒待機",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except APIError as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "annotation APIエラー (attempt=%d/%d): %s, %.1f秒待機",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"クラスタアノテーション生成がリトライ上限 {MAX_RETRIES} を超えました"
    )


def _build_user_prompt(rep_texts: list[str]) -> str:
    """ユーザメッセージを組み立てる."""
    bullets = "\n".join(f"- {t}" for t in rep_texts)
    return (
        "以下はあるクラスタの代表的な対応記録です:\n"
        f"{bullets}\n\n"
        "このクラスタの label と summary を JSON で出力してください."
    )


def _parse_annotation_json(raw: str) -> dict[str, str]:
    """LLM 出力（JSON 文字列）から label / summary を抽出."""
    text = raw.strip()
    # 念のため ```json ... ``` を剥がす（JSON モードでは通常出ない想定）
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("annotation JSON パース失敗: %s / raw=%r", exc, text[:120])
        return {"label": "解析失敗", "summary": text[:200]}

    label = _sanitize_label(str(payload.get("label", "")))
    summary = _sanitize_summary(str(payload.get("summary", "")))
    return {"label": label, "summary": summary}


def _sanitize_label(raw: str) -> str:
    """ラベルをクリーニング（短く・引用符除去）."""
    label = raw.strip()
    for ch in ('"', "'", "「", "」", "『", "』", "`"):
        label = label.replace(ch, "")
    for line in label.splitlines():
        line = line.strip()
        if line:
            label = line
            break
    if len(label) > 30:
        label = label[:30]
    if not label:
        label = "無題"
    return label


def _sanitize_summary(raw: str) -> str:
    """要約テキストをクリーニング."""
    summary = raw.strip()
    # 先頭末尾の引用符を除去
    for ch in ('"', "`"):
        if summary.startswith(ch) and summary.endswith(ch):
            summary = summary[1:-1].strip()
    # 2 行以上あれば空白で連結（レポートでは 1 段落扱い）
    summary = " ".join(line.strip() for line in summary.splitlines() if line.strip())
    # 過剰に長い場合は 400 字で切る
    if len(summary) > 400:
        summary = summary[:400] + "…"
    if not summary:
        summary = "（要約なし）"
    return summary


def _hash_key(rep_texts: list[str]) -> str:
    """代表テキスト列を安定キー化.

    注意: プロンプトスキーマが変わる場合、
    キャッシュ互換性を壊すため別モデル名・別キャッシュファイルが必要.
    """
    payload = "\n".join(rep_texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """モデル別キャッシュファイルパス.

    v2 suffix: label+summary スキーマを明示し、v1 の label 単体キャッシュとは
    別ファイルで管理.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    return cache_dir / f"cluster_annotations_v2_{safe_model}.pkl"


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    """pickle キャッシュを読み込む. 無ければ空辞書."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            logger.warning("annotation キャッシュ形式が不正. 新規作成: %s", path)
            return {}
        return cache
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.warning("annotation キャッシュ読込失敗 (%s). 破棄して新規作成", exc)
        return {}


def _save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    """pickle キャッシュを保存（原子的に差し替え）. JSON 版も並列保存."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    # デバッグ用途の人間可読 JSON
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
