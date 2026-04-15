"""自動チューナー — 候補手法を走査し最適なクラスタ設定を選ぶ.

参考: Chatbot Knowledge Preparation System の parameter_search.py
    - サンプル 1500 件でスイープ（O(n²)シルエットが現実的な上限）
    - 離散的・人が読めるパラメータグリッド
    - 1) サンプルでスイープ → 2) 勝者パラメータでフル再実行 の2相構成

責務:
    - L2 正規化した埋め込みに対し MiniBatchKMeans / DBSCAN / HDBSCAN を試行
    - 各候補のシルエットスコア（コサイン）を計算
    - 最大スコアの設定を `BestConfig` として返却（labels は全点分）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

# HDBSCAN はオプショナル（環境によって導入困難なことがある）
try:  # pragma: no cover - 環境依存
    import hdbscan  # type: ignore[import-not-found]

    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HDBSCAN_AVAILABLE = False
    logger.info("hdbscan 未導入のため HDBSCAN 候補はスキップされます")


Algorithm = Literal["kmeans", "dbscan", "hdbscan"]

# CKPS 準拠のスイープサンプル数. シルエット(cosine)の O(n²) 計算を許容できる上限
SWEEP_SAMPLE_SIZE: int = 1500

# 高次元埋め込み（例: 1536d）は距離の集中で cosine 類似度が狭い帯域に固まり
# KMeans/DBSCAN/HDBSCAN の識別力が大幅に落ちる. PCA で ~30 次元に落とすと
# 意味のある距離分散が戻る（LLM Topic Modeler の ClusteringService.swift 参照）
PCA_TARGET_DIM: int = 30
PCA_DIM_THRESHOLD: int = 50

# K 候補（CKPS を踏襲しつつ、小データ向けに 2, 3 も保持）
K_CANDIDATES: tuple[int, ...] = (2, 3, 5, 7, 10, 15, 20, 30, 50, 80)

# HDBSCAN の min_cluster_size 候補（CKPS 準拠）
HDBSCAN_MCS_CANDIDATES: tuple[int, ...] = (5, 10, 15, 20, 30, 50, 80, 100)
HDBSCAN_MIN_SAMPLES: int = 5

# DBSCAN eps グリッド（L2 正規化後の euclidean スケール）
# 正規化後: euclidean² = 2 × (1 - cos_sim). cos_sim=0.75 なら eps ≈ 0.707
DBSCAN_EPS_GRID: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.70)
DBSCAN_MIN_SAMPLES: int = 5

# 実用性フィルタ: ノイズ率がこの値を超える候補は選択対象外（高スコアでも無意味）
MAX_NOISE_RATIO_FOR_SELECTION: float = 0.5

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
        silhouette: シルエットスコア（cosine, 最終ラベル全点に対する評価）
        labels: 全サンプルに対するクラスタラベル（ノイズは -1）
        n_clusters: 有効なクラスタ数（ノイズを除く）
        n_noise: ノイズ点数
        all_trials: 参考. スイープで評価した全候補のスコアログ
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

    フェーズ1: SWEEP_SAMPLE_SIZE 件のサブサンプルで全候補を走査しスコア評価.
    フェーズ2: 勝者アルゴリズム・パラメータでフルデータにラベルを付与.

    Args:
        embeddings: shape=(N, D) の埋め込み行列
        min_clusters: KMeans 候補の下限 K（スイープフィルタ）
        max_clusters: KMeans 候補の上限 K（スイープフィルタ）

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

    # 次元の呪い対策: 高次元ならば PCA で ~30次元に削減して再正規化.
    # 同一モデルで fit した PCA を phase1/phase2 で使い回す
    reduced, pca_model = _reduce_dimensions(normalized)
    logger.info(
        "tuner: dim %d -> %d (pca=%s)",
        embeddings.shape[1],
        reduced.shape[1],
        pca_model is not None,
    )

    # フェーズ1: サンプリング → スイープ
    sample_indices = _sample_indices(n_samples, SWEEP_SAMPLE_SIZE, seed=RANDOM_STATE)
    sample = reduced[sample_indices]
    logger.info(
        "tuner: phase1 sweep on %d / %d points (hdbscan=%s)",
        sample.shape[0],
        n_samples,
        _HDBSCAN_AVAILABLE,
    )

    trials: list[dict[str, Any]] = []
    trials.extend(_sweep_kmeans(sample, min_clusters, max_clusters))
    trials.extend(_sweep_dbscan(sample))
    if _HDBSCAN_AVAILABLE:
        trials.extend(_sweep_hdbscan(sample))

    # 候補フィルタ:
    # 1. シルエット計算が成立 (silhouette != None)
    # 2. ノイズ率が閾値以下（少数点で高スコアを取る病的ケースを排除）
    valid = [
        t for t in trials
        if t["silhouette"] is not None
        and t["silhouette"] > -1.0
        and (t["n_noise"] / sample.shape[0]) <= MAX_NOISE_RATIO_FOR_SELECTION
    ]
    if not valid:
        # フォールバック: フィルタを緩めて、スコアがある候補から選ぶ
        logger.warning(
            "全候補でノイズ率 > %.0f%%. フィルタを緩めて選択します",
            MAX_NOISE_RATIO_FOR_SELECTION * 100,
        )
        valid = [t for t in trials if t["silhouette"] is not None and t["silhouette"] > -1.0]
    if not valid:
        raise RuntimeError("有効なクラスタリング候補が得られませんでした")

    winner = max(valid, key=lambda t: t["silhouette"])
    logger.info(
        "phase1 winner: %s params=%s sample_score=%.4f",
        winner["algorithm"],
        winner["params"],
        winner["silhouette"],
    )

    # フェーズ2: 勝者設定で PCA 後の空間でフルデータにラベル付与
    logger.info("phase2: apply winner to full data (N=%d)", n_samples)
    final_labels = _apply_to_full(
        normalized=reduced,
        sample=sample,
        sample_labels=winner["sample_labels"],
        sample_indices=sample_indices,
        algorithm=winner["algorithm"],
        params=winner["params"],
    )

    n_clusters, n_noise = _count_clusters(final_labels)
    final_score = _evaluate_silhouette(
        normalized=reduced,
        labels=final_labels,
        max_points=SWEEP_SAMPLE_SIZE,
    )
    if final_score is None:
        # フル反映で単一クラスタ化した場合のフォールバック
        final_score = float("-inf")

    logger.info(
        "採用: %s params=%s final_score=%.4f clusters=%d noise=%d",
        winner["algorithm"],
        winner["params"],
        final_score,
        n_clusters,
        n_noise,
    )

    return BestConfig(
        algorithm=winner["algorithm"],
        params=winner["params"],
        silhouette=final_score,
        labels=final_labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        all_trials=[_trial_log(t) for t in trials],
    )


# ---------------------------------------------------------------------------
# Sweep — 各手法ごとにサブサンプルで走査
# ---------------------------------------------------------------------------


def _sweep_kmeans(
    sample: np.ndarray, min_k: int, max_k: int
) -> list[dict[str, Any]]:
    """MiniBatchKMeans を K 候補で走査."""
    n = sample.shape[0]
    candidates = [
        k for k in K_CANDIDATES if max(2, min_k) <= k <= min(max_k, n - 1)
    ]
    if not candidates:
        candidates = [min(max(2, min_k), n - 1)]

    trials: list[dict[str, Any]] = []
    for k in tqdm(candidates, desc="KMeans sweep", unit="K", leave=False):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=3,
            batch_size=min(512, max(128, n // 4)),
            max_iter=100,
        )
        labels = model.fit_predict(sample)
        score = _silhouette_on_sample(sample, labels)
        trials.append(
            {
                "algorithm": "kmeans",
                "params": {"k": k},
                "sample_labels": labels,
                "silhouette": score,
                "n_clusters": len(set(labels) - {-1}),
                "n_noise": int((labels == -1).sum()),
            }
        )
        logger.debug("sweep kmeans k=%d score=%s", k, score)
    return trials


def _sweep_dbscan(sample: np.ndarray) -> list[dict[str, Any]]:
    """DBSCAN を eps グリッドで走査."""
    trials: list[dict[str, Any]] = []
    for eps in tqdm(DBSCAN_EPS_GRID, desc="DBSCAN sweep", unit="eps", leave=False):
        model = DBSCAN(
            eps=float(eps),
            min_samples=DBSCAN_MIN_SAMPLES,
            metric="euclidean",  # L2正規化済みなので euclidean で OK
            n_jobs=-1,
        )
        labels = model.fit_predict(sample)
        score = _silhouette_on_sample(sample, labels)
        trials.append(
            {
                "algorithm": "dbscan",
                "params": {
                    "eps": float(eps),
                    "min_samples": DBSCAN_MIN_SAMPLES,
                },
                "sample_labels": labels,
                "silhouette": score,
                "n_clusters": len(set(labels) - {-1}),
                "n_noise": int((labels == -1).sum()),
            }
        )
        logger.debug("sweep dbscan eps=%.2f score=%s", eps, score)
    return trials


def _sweep_hdbscan(sample: np.ndarray) -> list[dict[str, Any]]:
    """HDBSCAN を min_cluster_size 候補で走査."""
    trials: list[dict[str, Any]] = []
    n = sample.shape[0]
    candidates = [m for m in HDBSCAN_MCS_CANDIDATES if m < n]
    if not candidates:
        candidates = [max(2, n // 4)]

    for mcs in tqdm(candidates, desc="HDBSCAN sweep", unit="mcs", leave=False):
        model = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = model.fit_predict(sample)
        score = _silhouette_on_sample(sample, labels)
        trials.append(
            {
                "algorithm": "hdbscan",
                "params": {
                    "min_cluster_size": mcs,
                    "min_samples": HDBSCAN_MIN_SAMPLES,
                },
                "sample_labels": labels,
                "silhouette": score,
                "n_clusters": len(set(labels) - {-1}),
                "n_noise": int((labels == -1).sum()),
            }
        )
        logger.debug("sweep hdbscan mcs=%d score=%s", mcs, score)
    return trials


# ---------------------------------------------------------------------------
# Phase 2 — 勝者設定でフルデータへ
# ---------------------------------------------------------------------------


def _apply_to_full(
    normalized: np.ndarray,
    sample: np.ndarray,
    sample_labels: np.ndarray,
    sample_indices: np.ndarray,
    algorithm: Algorithm,
    params: dict[str, Any],
) -> np.ndarray:
    """勝者設定をフルデータに適用してラベルを得る.

    いずれの手法も **フルデータで再実行** する方針（サンプル伝播だと
    サンプルが局所密領域しか捉えなかった場合に大量ノイズ化するため）.

    - KMeans: MiniBatchKMeans を全点で再 fit + predict
    - DBSCAN: ball_tree で euclidean、n_jobs=-1
    - HDBSCAN: core_dist_n_jobs=-1 で euclidean
    """
    if normalized.shape[0] == sample.shape[0]:
        # N が既にサンプル以下なら、サンプルのラベルをそのまま返す
        return sample_labels.astype(np.int64)

    logger.info("phase2 running %s on full N=%d", algorithm, normalized.shape[0])

    if algorithm == "kmeans":
        model = MiniBatchKMeans(
            n_clusters=int(params["k"]),
            random_state=RANDOM_STATE,
            n_init=3,
            batch_size=min(2048, max(512, normalized.shape[0] // 20)),
            max_iter=100,
        )
        model.fit(normalized)
        return model.predict(normalized).astype(np.int64)

    if algorithm == "dbscan":
        model = DBSCAN(
            eps=float(params["eps"]),
            min_samples=int(params["min_samples"]),
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1,
        )
        return model.fit_predict(normalized).astype(np.int64)

    # hdbscan
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    return model.fit_predict(normalized).astype(np.int64)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化. ゼロベクトルはそのまま."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _reduce_dimensions(
    x: np.ndarray,
    target_dim: int = PCA_TARGET_DIM,
    threshold: int = PCA_DIM_THRESHOLD,
) -> tuple[np.ndarray, PCA | None]:
    """PCA で次元削減し、削減後に再 L2 正規化する.

    入力次元が `threshold` 以下ならそのまま返す（削減不要）.
    返り値は (削減後の配列, fit 済み PCA モデル). PCA をかけなかった場合は
    2要素目が None.
    """
    n, dim = x.shape
    if dim <= threshold:
        return x, None
    n_components = min(target_dim, dim, max(1, n - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(x)
    # PCA 後は単位長でなくなるため再正規化（cosine 等価性を維持）
    reduced = _l2_normalize(reduced)
    return reduced.astype(np.float32, copy=False), pca


def _sample_indices(n: int, cap: int, seed: int) -> np.ndarray:
    """評価・fit 用のインデックス配列を決定."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=cap, replace=False)


def _silhouette_on_sample(sample: np.ndarray, labels: np.ndarray) -> float | None:
    """サンプル全点に対してシルエット（cosine）を計算.

    ノイズ（-1）を除外後、有効クラスタが 2 未満なら None.
    """
    mask = labels != -1
    if mask.sum() < 2:
        return None
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return None
    try:
        return float(
            silhouette_score(sample[mask], labels[mask], metric="cosine")
        )
    except ValueError as exc:
        logger.debug("silhouette 計算失敗: %s", exc)
        return None


def _evaluate_silhouette(
    normalized: np.ndarray, labels: np.ndarray, max_points: int
) -> float | None:
    """フルデータのラベルを、最大 max_points 点のサブサンプルで評価."""
    n = normalized.shape[0]
    if n <= max_points:
        return _silhouette_on_sample(normalized, labels)

    rng = np.random.default_rng(RANDOM_STATE + 7)
    idx = rng.choice(n, size=max_points, replace=False)
    return _silhouette_on_sample(normalized[idx], labels[idx])


def _count_clusters(labels: np.ndarray) -> tuple[int, int]:
    """(有効クラスタ数, ノイズ数) を返す."""
    unique = set(labels.tolist())
    n_noise = int((labels == -1).sum())
    n_clusters = len(unique - {-1})
    return n_clusters, n_noise


def _trial_log(trial: dict[str, Any]) -> dict[str, Any]:
    """all_trials に格納するコンパクトな辞書（sample_labels は除外）."""
    return {
        "algorithm": trial["algorithm"],
        "params": trial["params"],
        "silhouette": trial["silhouette"],
        "n_clusters": int(trial["n_clusters"]),
        "n_noise": int(trial["n_noise"]),
    }
