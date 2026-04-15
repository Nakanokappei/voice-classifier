"""自動チューナー — 候補手法を走査し最適なクラスタ設定を選ぶ.

責務:
    - L2 正規化した埋め込みに対し KMeans / DBSCAN / HDBSCAN を試行
    - 各候補のシルエットスコア（コサイン）を計算
    - 最大スコアの設定を `BestConfig` として返却
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# HDBSCAN はオプショナル（環境によって導入困難なことがある）
try:  # pragma: no cover - 環境依存
    import hdbscan  # type: ignore[import-not-found]

    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HDBSCAN_AVAILABLE = False
    logger.info("hdbscan 未導入のため HDBSCAN 候補はスキップされます")


Algorithm = Literal["kmeans", "dbscan", "hdbscan"]

# シルエット評価時のサブサンプリング上限（スコア計算の計算量抑制）
SILHOUETTE_SAMPLE_CAP: int = 2000

# スコア閾値
SCORE_GOOD: float = 0.40
SCORE_WARN: float = 0.20

# 再現性のための乱数種
RANDOM_STATE: int = 42


@dataclass
class BestConfig:
    """チューニング結果.

    Attributes:
        algorithm: 採用アルゴリズム名
        params: 採用パラメータ（辞書）
        silhouette: シルエットスコア（cosine, サブサンプル評価値）
        labels: 全サンプルに対するクラスタラベル（ノイズは -1）
        n_clusters: 有効なクラスタ数（ノイズを除く）
        n_noise: ノイズ点数
        all_trials: 参考. 評価した全候補のスコアログ
    """

    algorithm: Algorithm
    params: dict[str, Any]
    silhouette: float
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    all_trials: list[dict[str, Any]] = field(default_factory=list)

    @property
    def quality_flag(self) -> Literal["good", "warn", "poor"]:
        """スコアに基づく品質判定."""
        if self.silhouette >= SCORE_GOOD:
            return "good"
        if self.silhouette >= SCORE_WARN:
            return "warn"
        return "poor"


def find_best_clustering(
    embeddings: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 20,
) -> BestConfig:
    """埋め込み行列に対して最適なクラスタリング設定を探す.

    Args:
        embeddings: shape=(N, D) の埋め込み行列
        min_clusters: KMeans 候補の下限 K
        max_clusters: KMeans 候補の上限 K

    Returns:
        BestConfig

    Raises:
        ValueError: サンプル数が min_clusters を下回る
    """
    n_samples = embeddings.shape[0]
    if n_samples < max(2, min_clusters):
        raise ValueError(
            f"サンプル数 {n_samples} がクラスタリングに不足 (min_clusters={min_clusters})"
        )

    # コサイン類似度でユークリッド距離と等価換算するため L2 正規化
    normalized = _l2_normalize(embeddings)

    # サブサンプル用インデックスを固定（評価の一貫性のため全候補で同じサブサンプルを使う）
    sample_indices = _sample_indices(n_samples, SILHOUETTE_SAMPLE_CAP)

    trials: list[dict[str, Any]] = []
    trials.extend(_try_kmeans(normalized, sample_indices, min_clusters, max_clusters))
    trials.extend(_try_dbscan(normalized, sample_indices))
    if _HDBSCAN_AVAILABLE:
        trials.extend(_try_hdbscan(normalized, sample_indices))

    # 有効な候補のみを残す
    valid = [t for t in trials if t["silhouette"] is not None and not math.isnan(t["silhouette"])]
    if not valid:
        raise RuntimeError("有効なクラスタリング候補が得られませんでした")

    best = max(valid, key=lambda t: t["silhouette"])
    logger.info(
        "採用: %s params=%s score=%.4f clusters=%d noise=%d",
        best["algorithm"],
        best["params"],
        best["silhouette"],
        best["n_clusters"],
        best["n_noise"],
    )

    return BestConfig(
        algorithm=best["algorithm"],
        params=best["params"],
        silhouette=float(best["silhouette"]),
        labels=best["labels"],
        n_clusters=int(best["n_clusters"]),
        n_noise=int(best["n_noise"]),
        all_trials=trials,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化. ゼロベクトルはそのまま."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _sample_indices(n: int, cap: int) -> np.ndarray:
    """スコア計算用のインデックス配列を決定."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.choice(n, size=cap, replace=False)


def _evaluate_silhouette(
    normalized: np.ndarray, labels: np.ndarray, sample_indices: np.ndarray
) -> float | None:
    """ノイズを除外した上でシルエット（cosine）を計算."""
    sub_labels = labels[sample_indices]
    mask = sub_labels != -1
    if mask.sum() < 2:
        return None
    unique = np.unique(sub_labels[mask])
    if len(unique) < 2:
        return None
    try:
        return float(
            silhouette_score(
                normalized[sample_indices][mask],
                sub_labels[mask],
                metric="cosine",
            )
        )
    except ValueError as exc:
        logger.debug("silhouette 計算失敗: %s", exc)
        return None


def _count_clusters(labels: np.ndarray) -> tuple[int, int]:
    """(有効クラスタ数, ノイズ数) を返す."""
    unique = set(labels.tolist())
    n_noise = int((labels == -1).sum())
    n_clusters = len(unique - {-1})
    return n_clusters, n_noise


def _try_kmeans(
    normalized: np.ndarray,
    sample_indices: np.ndarray,
    min_k: int,
    max_k: int,
) -> list[dict[str, Any]]:
    """KMeans を候補 K で走査."""
    n_samples = normalized.shape[0]
    # データ件数に合わせて妥当な探索範囲に補正
    upper = min(max_k, max(min_k, int(math.ceil(math.sqrt(n_samples)))))
    lower = max(2, min_k)
    upper = max(upper, lower)

    trials: list[dict[str, Any]] = []
    for k in range(lower, upper + 1):
        if k >= n_samples:
            break
        model = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
        labels = model.fit_predict(normalized)
        n_clusters, n_noise = _count_clusters(labels)
        score = _evaluate_silhouette(normalized, labels, sample_indices)
        trials.append(
            {
                "algorithm": "kmeans",
                "params": {"k": k},
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": score,
            }
        )
        logger.debug("kmeans k=%d score=%s", k, score)
    return trials


def _try_dbscan(
    normalized: np.ndarray,
    sample_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """DBSCAN を eps グリッドで走査. min_samples はデータ量から決定."""
    n_samples = normalized.shape[0]
    min_samples = max(3, int(math.ceil(math.log(max(n_samples, 2)))))
    eps_grid = np.linspace(0.15, 0.60, 10)

    trials: list[dict[str, Any]] = []
    for eps in eps_grid:
        model = DBSCAN(eps=float(eps), min_samples=min_samples, metric="cosine")
        labels = model.fit_predict(normalized)
        n_clusters, n_noise = _count_clusters(labels)
        # ノイズしか無い / 1クラスタしかできない場合は除外
        if n_clusters < 2:
            trials.append(
                {
                    "algorithm": "dbscan",
                    "params": {"eps": float(eps), "min_samples": min_samples},
                    "labels": labels,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "silhouette": None,
                }
            )
            continue
        score = _evaluate_silhouette(normalized, labels, sample_indices)
        trials.append(
            {
                "algorithm": "dbscan",
                "params": {"eps": float(eps), "min_samples": min_samples},
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": score,
            }
        )
        logger.debug(
            "dbscan eps=%.2f min_samples=%d n_clusters=%d score=%s",
            eps,
            min_samples,
            n_clusters,
            score,
        )
    return trials


def _try_hdbscan(
    normalized: np.ndarray,
    sample_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """HDBSCAN を min_cluster_size の候補で走査."""
    n_samples = normalized.shape[0]
    candidates = sorted(
        {
            max(5, int(math.ceil(n_samples / 200))),
            max(10, int(math.ceil(n_samples / 100))),
            max(20, int(math.ceil(n_samples / 50))),
        }
    )

    trials: list[dict[str, Any]] = []
    for min_cluster_size in candidates:
        if min_cluster_size >= n_samples:
            continue
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",  # L2 正規化済みデータはユークリッド ≈ cosine
        )
        labels = model.fit_predict(normalized)
        n_clusters, n_noise = _count_clusters(labels)
        if n_clusters < 2:
            trials.append(
                {
                    "algorithm": "hdbscan",
                    "params": {"min_cluster_size": min_cluster_size},
                    "labels": labels,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "silhouette": None,
                }
            )
            continue
        score = _evaluate_silhouette(normalized, labels, sample_indices)
        trials.append(
            {
                "algorithm": "hdbscan",
                "params": {"min_cluster_size": min_cluster_size},
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": score,
            }
        )
        logger.debug(
            "hdbscan min_cluster_size=%d n_clusters=%d score=%s",
            min_cluster_size,
            n_clusters,
            score,
        )
    return trials
