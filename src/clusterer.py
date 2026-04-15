"""クラスタリング結果の後処理 — 代表テキストの抽出.

責務:
    - 各クラスタの重心を算出し、重心に近い上位 N 件を代表として選ぶ
    - ノイズラベル（-1）は「未分類」としてまとめる
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NOISE_LABEL: int = -1


@dataclass
class ClusterSummary:
    """単一クラスタのサマリ情報.

    Attributes:
        cluster_id: クラスタID（ノイズは -1）
        size: クラスタ内のサンプル数
        representative_indices: 重心に近い順のインデックス（`df` 行番号基準）
        representative_texts: 代表テキスト（正規化後）
    """

    cluster_id: int
    size: int
    representative_indices: list[int]
    representative_texts: list[str]


def summarize_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: int = 5,
) -> list[ClusterSummary]:
    """各クラスタの代表テキストを抽出してサマリ化.

    Args:
        df: `_normalized_text` 列を持つ DataFrame（`labels` と同じ行順）
        embeddings: shape=(N, D) の埋め込み行列
        labels: 各サンプルのクラスタID配列 shape=(N,)
        top_k: 各クラスタから抽出する代表件数

    Returns:
        クラスタIDの昇順に並べた ClusterSummary のリスト. ノイズが存在する場合は末尾に含まれる
    """
    if len(df) != embeddings.shape[0] or len(df) != len(labels):
        raise ValueError(
            f"行数不整合: df={len(df)} embeddings={embeddings.shape[0]} labels={len(labels)}"
        )

    # 事前に正規化（centroid 計算と cosine 距離で一貫させる）
    normalized = _l2_normalize(embeddings)

    summaries: list[ClusterSummary] = []
    unique_labels = sorted(set(labels.tolist()))
    # ノイズは末尾にまとめる
    ordered = [c for c in unique_labels if c != NOISE_LABEL]
    if NOISE_LABEL in unique_labels:
        ordered.append(NOISE_LABEL)

    for cluster_id in ordered:
        member_mask = labels == cluster_id
        member_indices = np.where(member_mask)[0]
        size = int(member_indices.size)

        if cluster_id == NOISE_LABEL:
            # ノイズは代表テキストを選ばず、件数のみ保持
            summaries.append(
                ClusterSummary(
                    cluster_id=NOISE_LABEL,
                    size=size,
                    representative_indices=[],
                    representative_texts=[],
                )
            )
            continue

        # 重心に近い上位 top_k を抽出
        rep_indices = _pick_representatives(
            normalized[member_indices], member_indices, top_k
        )
        rep_texts = [str(df.iloc[i]["_normalized_text"]) for i in rep_indices]

        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                size=size,
                representative_indices=[int(i) for i in rep_indices],
                representative_texts=rep_texts,
            )
        )
        logger.debug("cluster=%d size=%d reps=%d", cluster_id, size, len(rep_indices))

    return summaries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化. ゼロ行は正準基底 `e_0` に置換.

    tuner._l2_normalize と同一仕様（責務分離のため各モジュールに配置）.
    下流の `member_vectors @ centroid` で divide-by-zero を踏まないよう、
    ゼロノルムを根絶する.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    zero_mask = norms.flatten() == 0
    if zero_mask.any():
        x = x.copy()
        x[zero_mask, 0] = 1.0
        norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def _pick_representatives(
    member_vectors: np.ndarray,
    member_indices: np.ndarray,
    top_k: int,
) -> list[int]:
    """クラスタ重心に最も近い上位 top_k のインデックスを返す.

    重心は member_vectors の平均を L2 正規化したもの.
    近さはコサイン類似度の降順（= コサイン距離の昇順）.
    """
    if member_vectors.shape[0] == 0:
        return []

    centroid = member_vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        # 全ベクトルが相殺するケースは稀. 先頭から返す
        return member_indices[:top_k].tolist()
    centroid /= centroid_norm

    # member_vectors は _l2_normalize 済みなのでゼロ行は存在しない前提.
    # ただし数値精度で微小なオーバーフロー等が起きうるため防御的に errstate で抑制
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        similarities = member_vectors @ centroid

    # NaN/Inf が万一混入しても最下位に送る
    similarities = np.nan_to_num(
        similarities, nan=-np.inf, posinf=np.inf, neginf=-np.inf
    )
    order = np.argsort(-similarities)[:top_k]
    return [int(member_indices[i]) for i in order]
