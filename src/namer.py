"""クラスタのLLM自動アノテーション (ラベル + 要約 + 重複解消).

処理の流れ:

    1. **データセットの意味推定** (infer_dataset_context)
       入力CSVから埋め込み対象テキストをランダムに 5 件サンプルし、
       LLM にこのデータセットの業務分野や性質を推定させる. 後続の
       ラベル生成プロンプトに埋め込むグラウンディング情報として使う.

    2. **並列ラベル生成** (generate_cluster_annotations)
       全クラスタを同時進行で LLM に問い合わせ、それぞれ label と summary を
       JSON で受け取る. プロンプトにはデータセットの意味を埋め込み、
       「○○システムのカテゴリとして相応しい粒度のラベル」という制約を与える.

    3. **重複解消ループ** (resolve_label_duplicates)
       生成後に label の重複をチェック. 重複していた場合:
       - クラスタサイズが大きい方は現在の label をそのまま維持
       - 小さい方は、両方のクラスタの代表テキストを LLM に提示し、
         「他方との差異が分かるラベル」を再生成
       - 全 label がユニークになるまで繰り返す（最大 3 周）

責務上の注意:
    - キャッシュは initial 生成分のみ. 重複解消で再生成された label は
      その場限りで扱う（同じ入力でも既存 label 群との相対関係で変わるため）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm

from .clusterer import ClusterSummary, NOISE_LABEL

logger = logging.getLogger(__name__)

# OpenAI の公式推奨: GPT-5.4 nano は分類・データ抽出・ランキング・サブエージェント
# 用途で速度とコストの両立が期待できるとされる. クラスタラベリングは正に該当.
DEFAULT_MODEL: str = "gpt-5.4-nano"


# ---------------------------------------------------------------------------
# Model-specific API adapters
# ---------------------------------------------------------------------------
#
# OpenAI のモデルはシリーズごとに Chat Completions API の許容パラメータが異なる:
#   - GPT-5 系, o-series: `max_completion_tokens` を使う. `max_tokens` はエラー
#   - o-series (o1, o3): `temperature` も未サポート（固定値）
#   - GPT-4o, GPT-4, GPT-3.5: 従来の `max_tokens` が通る
#
# 新モデルが出るたびに個別対応するのは辛いので、モデル名プレフィックスで判定する.
# 判定ロジックは `_build_chat_kwargs` に集約.


def _uses_max_completion_tokens(model: str) -> bool:
    """新 API 仕様（max_completion_tokens）を使うモデル判定."""
    prefixes = ("gpt-5", "o1", "o3", "o4")
    return any(model.startswith(p) for p in prefixes)


def _supports_custom_temperature(model: str) -> bool:
    """temperature パラメータを受け付けるか."""
    # o-series は temperature 1.0 固定. 他は自由.
    reasoning_prefixes = ("o1", "o3", "o4")
    return not any(model.startswith(p) for p in reasoning_prefixes)


def _supports_json_mode(model: str) -> bool:
    """response_format={'type': 'json_object'} が使えるか.

    GPT-4o, GPT-5, o-series, gpt-4-turbo は対応.
    gpt-3.5 は古いバージョンだと非対応だが一般利用される 0125 以降は OK.
    保守的に「大体対応している」前提で True を返し、失敗時はプロンプトだけで回避.
    """
    return True


def _build_chat_kwargs(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.3,
    json_mode: bool = True,
) -> dict:
    """モデル別 API 仕様差を吸収して Chat Completions の kwargs を組み立てる."""
    kwargs: dict = {"model": model, "messages": messages}

    # トークン上限パラメータ
    if _uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # temperature（o-series は省略）
    if _supports_custom_temperature(model):
        kwargs["temperature"] = temperature

    # JSON モード
    if json_mode and _supports_json_mode(model):
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs
MAX_CONCURRENCY: int = 8
MAX_RETRIES: int = 4
BACKOFF_BASE_SEC: float = 2.0

# 重複解消の最大反復回数. 3 周で収束しない場合は手動調整を促す
MAX_DEDUP_ITERATIONS: int = 3

# データセット意味推定用のサンプル件数
DATASET_SAMPLE_SIZE: int = 5


@dataclass
class ClusterAnnotation:
    """LLM が生成したクラスタのアノテーション.

    Attributes:
        label: 10〜20 字の短いラベル（cluster_name として利用）
        summary: クラスタの代表として表示する 1〜3 文の要約テキスト
    """

    label: str
    summary: str


@dataclass
class DatasetContext:
    """データセットの意味推定結果.

    Attributes:
        domain: 業務分野（例: "家電修理受付", "SaaSカスタマーサポート"）
        granularity_hint: クラスタ粒度の指針（例: "故障症状レベルで分類"）
        sample_texts: 推定に使った実サンプル（ログ用、再現性のため保持）
    """

    domain: str
    granularity_hint: str
    sample_texts: list[str] = field(default_factory=list)

    def grounding_hint(self) -> str:
        """プロンプトに埋め込むグラウンディング指示文を生成."""
        return (
            f"このデータセットは「{self.domain}」であり、"
            f"{self.granularity_hint}. "
            "抽象的すぎる一般語（「問題」「不具合」単独など）は避け、"
            "このデータセットのトピックとして具体的な名詞句をラベルとして返すこと."
        )


# ---------------------------------------------------------------------------
# Step 1: Dataset context inference
# ---------------------------------------------------------------------------


def infer_dataset_context(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    sample_size: int = DATASET_SAMPLE_SIZE,
    seed: int = 42,
    api_key: str | None = None,
) -> DatasetContext:
    """入力テキスト列からランダムに数件を選び、LLM にデータセットの意味を推定させる.

    Args:
        texts: 埋め込み対象の正規化済みテキストリスト（重複集約後）
        model: 推定に使う Chat モデル
        sample_size: サンプリングする件数
        seed: 乱数シード（同入力なら同結果）
        api_key: 明示 API キー

    Returns:
        DatasetContext (domain / granularity_hint / sample_texts)
    """
    if not texts:
        logger.warning("テキストが空のためデータセット意味推定をスキップ")
        return DatasetContext(
            domain="不明", granularity_hint="具体的な業務用語で分類する"
        )

    # 重複を除いた上でサンプリング（同じ表現が繰り返し出る影響を排除）
    unique_texts = list(dict.fromkeys(texts))
    rng = _np_rng(seed)
    n = min(sample_size, len(unique_texts))
    indices = rng.choice(len(unique_texts), size=n, replace=False)
    samples = [unique_texts[int(i)] for i in indices]

    client = _make_client(api_key)
    bullets = "\n".join(f"- {_trim_for_prompt(t)}" for t in samples)
    system = (
        "あなたはデータセットの性質を見抜く分析家です. "
        "与えられた代表的なレコード数件から、このデータセットの業務分野と、"
        "分類粒度の指針を JSON で返してください.\n\n"
        "出力スキーマ（厳守）:\n"
        "{\n"
        '  "domain": "<業務分野. 例: \'家電修理受付データ\' '
        '\'SaaSのカスタマーサポート問い合わせ\' など具体的に>",\n'
        '  "granularity_hint": "<このデータセットのクラスタリング結果に'
        'つけるラベルの粒度指針. 例: \'故障症状レベルで分類\' '
        '\'問い合わせ種別ごとに分類\'>"\n'
        "}\n\n"
        "必ず有効な JSON のみを返す."
    )
    user = f"以下はデータセットからランダム抽出した {n} 件のレコードです:\n{bullets}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                **_build_chat_kwargs(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
            )
            raw = response.choices[0].message.content or "{}"
            payload = _safe_json_loads(raw)
            domain = str(payload.get("domain", "")).strip() or "不明な分野"
            hint = (
                str(payload.get("granularity_hint", "")).strip()
                or "具体的な業務用語で分類する"
            )
            logger.info("データセット推定: domain=%s hint=%s", domain, hint)
            return DatasetContext(
                domain=domain, granularity_hint=hint, sample_texts=samples
            )
        except (RateLimitError, APIError) as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "dataset context API エラー (attempt=%d/%d): %s, %.1f秒待機",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    logger.warning("データセット意味推定がリトライ上限超過、フォールバック使用")
    return DatasetContext(
        domain="不明", granularity_hint="具体的な業務用語で分類する",
        sample_texts=samples,
    )


# ---------------------------------------------------------------------------
# Step 2: Parallel annotation generation
# ---------------------------------------------------------------------------


def generate_cluster_annotations(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    dataset_context: DatasetContext | None = None,
) -> dict[int, ClusterAnnotation]:
    """各クラスタに対して label と summary をまとめて生成.

    Args:
        summaries: `clusterer.summarize_clusters` の戻り値
        cache_dir: キャッシュ先
        model: OpenAI Chat モデル名
        api_key: 明示的に API キーを渡す場合
        dataset_context: グラウンディング情報. None なら generic プロンプト

    Returns:
        cluster_id -> ClusterAnnotation. ノイズは固定ラベル
    """
    if not summaries:
        return {}

    cache_path = _cache_path_for(Path(cache_dir), model)
    cache: dict[str, dict[str, str]] = _load_cache(cache_path)

    # キャッシュキーは代表テキスト + grounding ヒントを含める
    # （データセットが違えば同じ rep_texts でも違うラベルが期待されるため）
    grounding_salt = dataset_context.grounding_hint() if dataset_context else ""

    tasks: list[tuple[int, str, list[str]]] = []
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            continue
        if not summary.representative_texts:
            continue
        key = _hash_key(summary.representative_texts, grounding_salt)
        tasks.append((summary.cluster_id, key, summary.representative_texts))

    pending = [(cid, key, texts) for cid, key, texts in tasks if key not in cache]
    if pending:
        logger.info(
            "クラスタアノテーション生成: %d件をAPI取得 (キャッシュヒット: %d件)",
            len(pending), len(tasks) - len(pending),
        )
        client = _make_client(api_key)
        new_annotations = _fetch_parallel(
            client, pending, model, dataset_context
        )
        cache.update(new_annotations)
        _save_cache(cache_path, cache)
    else:
        logger.info(
            "クラスタアノテーション生成: 全 %d 件がキャッシュヒット", len(tasks)
        )

    # 結果を組み立て
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
        key = _hash_key(summary.representative_texts, grounding_salt)
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


# ---------------------------------------------------------------------------
# Step 3: Duplicate resolution
# ---------------------------------------------------------------------------


def resolve_label_duplicates(
    summaries: list[ClusterSummary],
    annotations: dict[int, ClusterAnnotation],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    dataset_context: DatasetContext | None = None,
    max_iterations: int = MAX_DEDUP_ITERATIONS,
) -> dict[int, ClusterAnnotation]:
    """ラベル重複を解消する.

    戦略:
    - 同じ label を持つクラスタが複数ある場合、サイズが大きい方は据え置き
    - サイズが小さい方（同率なら cluster_id の若い順）は両者の代表テキストを
      LLM に提示して差別化したラベルを再生成
    - 1 周分解消後、さらに別の組で重複が生まれる可能性があるためループ

    Args:
        summaries: clusterer.summarize_clusters の戻り値
        annotations: generate_cluster_annotations の出力
        model / api_key / dataset_context: LLM 呼び出し用
        max_iterations: 最大反復回数（デフォ 3）

    Returns:
        重複解消後の新しい annotations dict（入力を破壊しない）
    """
    # ノイズ以外だけを対象に
    size_by_cid = {s.cluster_id: s.size for s in summaries}
    reps_by_cid = {s.cluster_id: s.representative_texts for s in summaries}
    working = dict(annotations)  # shallow copy

    for iteration in range(1, max_iterations + 1):
        # 現在の label ごとにクラスタ ID を集める（ノイズは対象外）
        groups: dict[str, list[int]] = defaultdict(list)
        for cid, ann in working.items():
            if cid == NOISE_LABEL:
                continue
            groups[ann.label].append(cid)

        duplicates = {
            label: cids for label, cids in groups.items() if len(cids) >= 2
        }
        if not duplicates:
            if iteration > 1:
                logger.info(
                    "重複解消完了: %d 周目で全 label がユニーク化",
                    iteration,
                )
            break

        logger.info(
            "重複解消 iteration %d: %d 件の重複 label を検出",
            iteration, len(duplicates),
        )

        # 再生成対象を決定
        regeneration_tasks: list[tuple[int, list[str], int, list[str]]] = []
        for label, cids in duplicates.items():
            # サイズ降順で並べ、先頭（最大）を据え置き、それ以外を再生成
            ordered = sorted(
                cids,
                key=lambda c: (-size_by_cid.get(c, 0), c),
            )
            keeper = ordered[0]
            keeper_reps = reps_by_cid.get(keeper, [])
            for losing in ordered[1:]:
                losing_reps = reps_by_cid.get(losing, [])
                if not losing_reps:
                    continue
                regeneration_tasks.append(
                    (losing, losing_reps, keeper, keeper_reps)
                )
            logger.debug(
                "  重複 '%s': keeper=#%d(size=%d), 再生成=%s",
                label, keeper, size_by_cid.get(keeper, 0),
                [c for c in ordered[1:]],
            )

        if not regeneration_tasks:
            break

        # LLM に差別化ラベルを並列で問い合わせ
        client = _make_client(api_key)
        new_labels = _differentiate_labels(
            client, regeneration_tasks, model, dataset_context
        )

        # 反映
        for cid, (new_label, new_summary) in new_labels.items():
            old = working[cid]
            working[cid] = ClusterAnnotation(
                label=new_label,
                # summary は元のを保持（差別化対象は label のみ）
                summary=old.summary if new_summary is None else new_summary,
            )
    else:
        logger.warning(
            "重複解消が %d 周でも収束しませんでした. 手動調整を推奨",
            max_iterations,
        )

    return working


# ---------------------------------------------------------------------------
# 後方互換: generate_cluster_names
# ---------------------------------------------------------------------------


def generate_cluster_names(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[int, str]:
    """後方互換: cluster_id -> label のみ返すラッパ."""
    annotations = generate_cluster_annotations(
        summaries, cache_dir, model, api_key
    )
    return {cid: ann.label for cid, ann in annotations.items()}


# ---------------------------------------------------------------------------
# Internal: OpenAI client and retry
# ---------------------------------------------------------------------------


def _make_client(api_key: str | None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY が未設定です. .env または環境変数で指定してください"
        )
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    return OpenAI(api_key=key, timeout=timeout)


def _np_rng(seed: int):
    """numpy.random.Generator を遅延インポート（tests から利用）."""
    import numpy as np

    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Internal: initial annotation
# ---------------------------------------------------------------------------


def _build_system_prompt(context: DatasetContext | None) -> str:
    """データセット文脈を反映した system プロンプトを生成."""
    grounding = context.grounding_hint() if context else ""
    base = (
        "あなたはデータセットを分析する専門家です. "
        "与えられた代表テキスト群からクラスタを特徴付ける短いラベルと、"
        "そのクラスタを説明する要約を JSON で返してください.\n\n"
    )
    if grounding:
        base += f"【データセット文脈】\n{grounding}\n\n"
    base += (
        "出力スキーマ（厳守）:\n"
        "{\n"
        '  "label": "<10〜20 文字の日本語. 名詞句. 具体的・業務的な用語>",\n'
        '  "summary": "<代表テキストを総合した 1〜3 文の日本語. '
        '問い合わせ内容の傾向を述べる>"\n'
        "}\n\n"
        "制約:\n"
        "- label は修飾語・絵文字・引用符・余計な記号を含めない\n"
        "- summary は代表テキストから読み取れる事実のみを記述し、"
        "推測・助言は避ける\n"
        "- 入力が英語でもラベル・要約は日本語で返す\n"
        "- 必ず有効な JSON のみを返す（コードブロック囲みは厳禁）"
    )
    return base


def _fetch_parallel(
    client: OpenAI,
    pending: list[tuple[int, str, list[str]]],
    model: str,
    context: DatasetContext | None,
) -> dict[str, dict[str, str]]:
    """各クラスタの label/summary を並列に問い合わせる."""
    results: dict[str, dict[str, str]] = {}
    lock = threading.Lock()
    system_prompt = _build_system_prompt(context)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_meta = {
            executor.submit(
                _request_annotation, client, texts, model, system_prompt
            ): (cid, key)
            for cid, key, texts in pending
        }
        with tqdm(
            total=len(pending), desc="Annotating", unit="cluster", leave=False
        ) as pbar:
            for future in as_completed(future_to_meta):
                cid, key = future_to_meta[future]
                try:
                    annotation = future.result()
                except Exception as exc:  # noqa: BLE001
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


def _request_annotation(
    client: OpenAI,
    rep_texts: list[str],
    model: str,
    system_prompt: str,
) -> dict[str, str]:
    """指数バックオフ付きで Chat Completions を呼ぶ."""
    bullets = "\n".join(f"- {_trim_for_prompt(t)}" for t in rep_texts)
    user_prompt = (
        "以下はあるクラスタの代表的なレコードです:\n"
        f"{bullets}\n\n"
        "このクラスタの label と summary を JSON で出力してください."
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                **_build_chat_kwargs(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=400,
                )
            )
            content = response.choices[0].message.content or "{}"
            return _parse_annotation_json(content)
        except (RateLimitError, APIError) as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "annotation API エラー (attempt=%d/%d): %s, %.1f秒待機",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"クラスタアノテーション生成がリトライ上限 {MAX_RETRIES} を超えました"
    )


# ---------------------------------------------------------------------------
# Internal: duplicate label differentiation
# ---------------------------------------------------------------------------


def _differentiate_labels(
    client: OpenAI,
    tasks: list[tuple[int, list[str], int, list[str]]],
    model: str,
    context: DatasetContext | None,
) -> dict[int, tuple[str, str | None]]:
    """重複した label を差別化する再生成を並列実行.

    Args:
        tasks: (losing_cid, losing_reps, keeper_cid, keeper_reps) のリスト
    Returns:
        losing_cid -> (新 label, 新 summary or None)
    """
    results: dict[int, tuple[str, str | None]] = {}
    lock = threading.Lock()
    grounding = context.grounding_hint() if context else ""

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_cid = {
            executor.submit(
                _request_differentiated_label,
                client, losing_reps, keeper_reps, model, grounding,
            ): losing_cid
            for losing_cid, losing_reps, _, keeper_reps in tasks
        }
        with tqdm(
            total=len(tasks), desc="Dedup labels", unit="cluster", leave=False
        ) as pbar:
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    label, summary = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "クラスタ %d の差別化ラベル生成に失敗: %s", cid, exc
                    )
                    label, summary = f"クラスタ #{cid}", None
                with lock:
                    results[cid] = (label, summary)
                pbar.update(1)
    return results


def _request_differentiated_label(
    client: OpenAI,
    losing_reps: list[str],
    keeper_reps: list[str],
    model: str,
    grounding: str,
) -> tuple[str, str | None]:
    """他クラスタと差別化したラベルを要求する LLM 呼び出し."""
    losing_bullets = "\n".join(f"- {_trim_for_prompt(t)}" for t in losing_reps)
    keeper_bullets = "\n".join(f"- {_trim_for_prompt(t)}" for t in keeper_reps)

    system = (
        "あなたはデータセットを分析する専門家です. "
        "2 つのクラスタ A と B があり、自動生成されたラベルが重複しています. "
        "ここで要求されているのはクラスタ A の再ラベル付けで、B との違いを"
        "明確に表現する必要があります.\n\n"
    )
    if grounding:
        system += f"【データセット文脈】\n{grounding}\n\n"
    system += (
        "出力スキーマ（厳守）:\n"
        "{\n"
        '  "label": "<10〜20 文字の日本語. クラスタ A を B と区別できる具体名詞句>",\n'
        '  "summary": "<任意. クラスタ A の 1〜3 文要約. 省略可>"\n'
        "}\n\n"
        "制約:\n"
        "- label は必ずクラスタ B のものと明確に異なる表現にする\n"
        "- 抽象語（「問題」「不具合」単独）で逃げない\n"
        "- 必ず有効な JSON のみを返す"
    )
    user = (
        "=== クラスタ A（こちらに新しいラベルを付けてください）===\n"
        f"{losing_bullets}\n\n"
        "=== クラスタ B（参考. 既存のラベルを持つ側）===\n"
        f"{keeper_bullets}\n\n"
        "クラスタ A の新しい label と（任意で）summary を JSON で出力してください."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                **_build_chat_kwargs(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.35,
                    max_tokens=400,
                )
            )
            content = response.choices[0].message.content or "{}"
            parsed = _parse_annotation_json(content)
            label = parsed["label"]
            # summary は任意なので空のままなら None を返す
            summary = parsed["summary"] if parsed.get("summary") else None
            return label, summary
        except (RateLimitError, APIError) as exc:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "dedup API エラー (attempt=%d/%d): %s, %.1f秒待機",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError("差別化ラベル生成がリトライ上限超過")


# ---------------------------------------------------------------------------
# Internal: parsing / sanitization / caching
# ---------------------------------------------------------------------------


def _parse_annotation_json(raw: str) -> dict[str, str]:
    """LLM 出力 JSON から label / summary を抽出."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]
    payload = _safe_json_loads(text)
    if payload is None:
        return {"label": "解析失敗", "summary": text[:200]}

    label = _sanitize_label(str(payload.get("label", "")))
    summary = _sanitize_summary(str(payload.get("summary", "")))
    return {"label": label, "summary": summary}


def _safe_json_loads(text: str) -> dict | None:
    """JSON パース失敗時は None を返す."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON パース失敗: %s / raw=%r", exc, text[:120])
        return None


def _sanitize_label(raw: str) -> str:
    """ラベルをクリーニング."""
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
    for ch in ('"', "`"):
        if summary.startswith(ch) and summary.endswith(ch):
            summary = summary[1:-1].strip()
    summary = " ".join(line.strip() for line in summary.splitlines() if line.strip())
    if len(summary) > 400:
        summary = summary[:400] + "…"
    if not summary:
        summary = "（要約なし）"
    return summary


def _trim_for_prompt(text: str, max_chars: int = 500) -> str:
    """プロンプトに含めるテキストを過剰な長さから保護."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _hash_key(rep_texts: list[str], salt: str = "") -> str:
    """代表テキスト列を安定キー化. grounding 文脈も混ぜて別キーに."""
    payload = "\n".join(rep_texts) + "||" + salt
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """モデル別キャッシュファイルパス.

    v3 suffix: grounding 文脈を salt に入れた新スキーマで v2 とは別ファイル.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    return cache_dir / f"cluster_annotations_v3_{safe_model}.pkl"


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers for tests: expose unused-looking imports
# ---------------------------------------------------------------------------


def _label_frequencies(annotations: dict[int, ClusterAnnotation]) -> Counter:
    """テスト向け. 重複確認に使うラベル頻度."""
    return Counter(ann.label for ann in annotations.values())
