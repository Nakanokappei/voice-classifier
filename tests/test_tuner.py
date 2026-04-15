"""tuner モジュールのテスト — 合成データで最適解選択を確認."""

from __future__ import annotations

import numpy as np
import pytest

from src import tuner


def _synthetic_blobs(
    centers: list[list[float]],
    per_cluster: int = 30,
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """中心点の周りに球状ノイズでサンプルを生成."""
    rng = np.random.default_rng(seed)
    arrays: list[np.ndarray] = []
    for c in centers:
        center = np.asarray(c, dtype=np.float32)
        samples = rng.normal(
            loc=center,
            scale=noise_std,
            size=(per_cluster, len(c)),
        ).astype(np.float32)
        arrays.append(samples)
    return np.vstack(arrays)


def test_finds_three_clusters_on_clean_blobs() -> None:
    """明瞭に分離した 3 クラスタで最適解が見つかること."""
    centers = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    X = _synthetic_blobs(centers, per_cluster=40)

    best = tuner.find_best_clustering(X, min_clusters=2, max_clusters=10)

    # 3 クラスタが検出されること
    assert best.n_clusters == 3
    # シルエットは十分高いはず
    assert best.silhouette > 0.5
    # ラベルは入力行数と一致
    assert best.labels.shape == (X.shape[0],)


def test_raises_when_too_few_samples() -> None:
    """サンプル不足なら ValueError."""
    X = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        tuner.find_best_clustering(X, min_clusters=2)


def test_quality_flag_thresholds() -> None:
    """quality_flag の判定境界."""
    labels = np.zeros(3, dtype=int)
    good = tuner.BestConfig(
        algorithm="kmeans", params={}, silhouette=0.45,
        labels=labels, n_clusters=2, n_noise=0,
    )
    warn = tuner.BestConfig(
        algorithm="kmeans", params={}, silhouette=0.30,
        labels=labels, n_clusters=2, n_noise=0,
    )
    poor = tuner.BestConfig(
        algorithm="kmeans", params={}, silhouette=0.10,
        labels=labels, n_clusters=2, n_noise=0,
    )
    assert good.quality_flag == "good"
    assert warn.quality_flag == "warn"
    assert poor.quality_flag == "poor"


def test_all_trials_are_recorded() -> None:
    """試行履歴が all_trials に残ること."""
    X = _synthetic_blobs([[1.0, 0.0], [0.0, 1.0]], per_cluster=20)
    best = tuner.find_best_clustering(X, min_clusters=2, max_clusters=5)
    # KMeans だけで複数の K を試しているはず
    kmeans_trials = [t for t in best.all_trials if t["algorithm"] == "kmeans"]
    assert len(kmeans_trials) >= 2
