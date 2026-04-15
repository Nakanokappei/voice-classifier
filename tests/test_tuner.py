"""Tests for the tuner module using synthetic blobs."""

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
    """Sample points around each centroid with spherical noise."""
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
    """Three well-separated clusters are recovered under the insight target.

    The production default is ``faq`` which prefers 30-80 clusters, so on a
    tiny synthetic dataset of 120 rows it would deliberately over-split
    rather than recover the true three-cluster structure. For algorithmic
    correctness tests we use ``insight`` (pure silhouette maximisation).
    """
    centers = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    X = _synthetic_blobs(centers, per_cluster=40)

    best = tuner.find_best_clustering(
        X, min_clusters=2, max_clusters=10, target="insight"
    )

    # Three clusters are detected
    assert best.n_clusters == 3
    # Silhouette should be comfortably high
    assert best.silhouette > 0.5
    # Labels match the number of input rows
    assert best.labels.shape == (X.shape[0],)


def test_raises_when_too_few_samples() -> None:
    """Too few samples raises ValueError."""
    X = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        tuner.find_best_clustering(X, min_clusters=2)


def test_quality_flag_thresholds() -> None:
    """Threshold boundaries for quality_flag."""
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
    """All trials are recorded in all_trials."""
    X = _synthetic_blobs([[1.0, 0.0], [0.0, 1.0]], per_cluster=20)
    best = tuner.find_best_clustering(X, min_clusters=2, max_clusters=5)
    # KMeans alone should try multiple K values
    kmeans_trials = [t for t in best.all_trials if t["algorithm"] == "kmeans"]
    assert len(kmeans_trials) >= 2


def test_granularity_fit_rewards_target_range() -> None:
    """FAQ granularity: in-range = 1.0, under = steep penalty, over = gentle."""
    # FAQ target range is 30-80.
    assert tuner._granularity_fit(50, "faq") == 1.0
    assert tuner._granularity_fit(30, "faq") == 1.0
    assert tuner._granularity_fit(80, "faq") == 1.0
    # Under-clustering is penalised hard.
    assert tuner._granularity_fit(10, "faq") < 0.5
    assert tuner._granularity_fit(1, "faq") < 0.1 or tuner._granularity_fit(1, "faq") == 0.1
    # Over-clustering is softer.
    over = tuner._granularity_fit(120, "faq")
    assert 0.3 <= over < 1.0
    # Insight ignores cluster count entirely.
    for n in (2, 20, 100, 500):
        assert tuner._granularity_fit(n, "insight") == 1.0


def test_share_penalty_respects_budget() -> None:
    """share_penalty: ≤ budget returns 1.0; above it tapers toward the floor."""
    # FAQ budget is 0.10.
    assert tuner._share_penalty(0.05, "faq") == 1.0
    assert tuner._share_penalty(0.10, "faq") == 1.0
    assert tuner._share_penalty(0.15, "faq") < 1.0
    # Extreme share hits the floor, not zero.
    assert tuner._share_penalty(1.0, "faq") >= 0.2
    # Insight has effectively no limit (budget 1.0).
    assert tuner._share_penalty(0.5, "insight") == 1.0


def test_faq_target_prefers_finer_granularity_than_insight() -> None:
    """Two candidates with similar silhouette but very different cluster counts.

    Under the FAQ target, the candidate landing in the 30-80 range should win
    over a lower-count candidate even if its silhouette is slightly lower.
    Under insight, the higher-silhouette candidate wins regardless.
    """
    big_blob = _synthetic_blobs(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        per_cluster=10,
    )
    # Fabricate two trials on the same sample.
    import numpy as np
    labels_few = np.concatenate([np.zeros(15), np.ones(15)]).astype(int)
    labels_many = np.tile(np.arange(30), 1)  # 30 unique clusters of 1
    trial_few = {
        "algorithm": "kmeans",
        "params": {"k": 2},
        "sample_labels": labels_few,
        "silhouette": 0.45,
        "n_clusters": 2,
        "n_noise": 0,
    }
    trial_many = {
        "algorithm": "kmeans",
        "params": {"k": 30},
        "sample_labels": labels_many,
        "silhouette": 0.40,  # slightly lower
        "n_clusters": 30,
        "n_noise": 0,
    }
    scored_few_faq = tuner._target_score(trial_few, sample_size=30, target="faq")
    scored_many_faq = tuner._target_score(trial_many, sample_size=30, target="faq")
    assert scored_many_faq > scored_few_faq, "FAQ target should prefer more granularity"

    scored_few_insight = tuner._target_score(
        trial_few, sample_size=30, target="insight"
    )
    scored_many_insight = tuner._target_score(
        trial_many, sample_size=30, target="insight"
    )
    assert scored_few_insight > scored_many_insight, (
        "insight target should prefer higher silhouette regardless of count"
    )


def test_leiden_sweep_runs_when_available() -> None:
    """When the Leiden backend is installed, it produces trials across the grid."""
    if not tuner._LEIDEN_AVAILABLE:
        import pytest

        pytest.skip("Leiden backend not installed")
    X = _synthetic_blobs(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        per_cluster=30,
    )
    best = tuner.find_best_clustering(X, min_clusters=2, max_clusters=10)

    leiden_trials = [t for t in best.all_trials if t["algorithm"] == "leiden"]
    # Every resolution in the grid should have produced a trial.
    assert len(leiden_trials) == len(tuner.LEIDEN_RESOLUTION_GRID)
    # Clear cluster structure should yield at least two valid clusters for
    # at least one resolution.
    assert any(t["n_clusters"] >= 2 for t in leiden_trials)
