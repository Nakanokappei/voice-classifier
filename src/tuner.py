"""Auto-tuner — sweeps candidate methods and picks the best clustering config.

References:
    - `Chatbot Knowledge Preparation System` / `worker/src/steps/parameter_search.py`:
      sample 1,500 points (ceiling for O(n²) silhouette), discrete human-readable
      parameter grids, and a two-phase flow (sweep on sample → re-run with the
      winning config on the full data).

Responsibilities:
    - L2-normalise the embedding matrix; optionally reduce dimensionality via PCA.
    - Run MiniBatchKMeans / DBSCAN / HDBSCAN over the sample.
    - Compute silhouette score (cosine) for each candidate.
    - Return the winning configuration as ``BestConfig``, with labels for every row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from . import utils

logger = logging.getLogger(__name__)

# HDBSCAN is optional; it is occasionally painful to install on some platforms.
try:  # pragma: no cover - environment-dependent
    import hdbscan  # type: ignore[import-not-found]

    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HDBSCAN_AVAILABLE = False
    logger.info("hdbscan is not installed; HDBSCAN candidates will be skipped")


Algorithm = Literal["kmeans", "dbscan", "hdbscan"]

# Sample size for the sweep. 1,500 is the practical ceiling for O(n²) cosine silhouette.
SWEEP_SAMPLE_SIZE: int = 1500

# High-dimensional embeddings (e.g. 1,536d) suffer from distance concentration,
# which collapses cosine similarities into a narrow band and destroys the
# discriminative power of KMeans/DBSCAN/HDBSCAN. Projecting onto ~30 PCA
# components restores meaningful variance.
# See: `LLM Topic Modeler / Sources/Services/ClusteringService.swift`.
PCA_TARGET_DIM: int = 30
PCA_DIM_THRESHOLD: int = 50

# K candidate set (inherits CKPS's grid; keeps small K too for tiny datasets).
K_CANDIDATES: tuple[int, ...] = (2, 3, 5, 7, 10, 15, 20, 30, 50, 80)

# HDBSCAN ``min_cluster_size`` candidates (matches CKPS).
HDBSCAN_MCS_CANDIDATES: tuple[int, ...] = (5, 10, 15, 20, 30, 50, 80, 100)
HDBSCAN_MIN_SAMPLES: int = 5

# DBSCAN ``eps`` grid on the L2-normalised euclidean scale.
# After normalisation: euclidean² = 2 * (1 - cos_sim). Example: cos_sim=0.75 -> eps ≈ 0.707.
DBSCAN_EPS_GRID: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.70)
DBSCAN_MIN_SAMPLES: int = 5

# Usability filter: drop candidates whose noise ratio exceeds this threshold.
# A high score on "6 tiny clusters carved out of 1,500 points with 97% noise"
# is numerically valid but operationally useless.
MAX_NOISE_RATIO_FOR_SELECTION: float = 0.5

# Score thresholds for the quality flag.
SCORE_GOOD: float = 0.40
SCORE_WARN: float = 0.20

# Fixed random seed for reproducibility across runs.
RANDOM_STATE: int = 42


@dataclass
class BestConfig:
    """Tuning result.

    Attributes:
        algorithm: chosen algorithm name
        params: chosen parameters (dict)
        silhouette: final silhouette score (cosine) on the full data
        labels: cluster labels for every row; noise is -1
        n_clusters: number of valid clusters (noise excluded)
        n_noise: number of noise points
        all_trials: score log for every swept candidate (pre-filter)
        sweep_sample_size: points used during the sweep (for noise ratio reporting)
        dim_before_pca: dimension before PCA; equals ``dim_after_pca`` when PCA skipped
        dim_after_pca: dimension after PCA
    """

    algorithm: Algorithm
    params: dict[str, Any]
    silhouette: float
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    all_trials: list[dict[str, Any]] = field(default_factory=list)
    sweep_sample_size: int = 0
    dim_before_pca: int = 0
    dim_after_pca: int = 0

    @property
    def quality_flag(self) -> Literal["good", "warn", "poor"]:
        """Quality classification based on the silhouette score."""
        if self.silhouette >= SCORE_GOOD:
            return "good"
        if self.silhouette >= SCORE_WARN:
            return "warn"
        return "poor"

    @property
    def max_noise_ratio(self) -> float:
        """Noise-ratio filter threshold used during selection (surfaced in reports)."""
        return MAX_NOISE_RATIO_FOR_SELECTION


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def find_best_clustering(
    embeddings: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 20,
) -> BestConfig:
    """Search for the best clustering configuration over the embedding matrix.

    Phase 1: sweep every candidate on a ``SWEEP_SAMPLE_SIZE`` subsample and score it.
    Phase 2: re-run the winning (algorithm, params) on the full data to label it.
    """
    n_samples = embeddings.shape[0]
    if n_samples < max(2, min_clusters):
        raise ValueError(
            f"Only {n_samples} samples, not enough for clustering "
            f"(min_clusters={min_clusters})"
        )

    # L2-normalise so euclidean distance is monotonic with cosine distance.
    normalized = utils.l2_normalize(embeddings)

    # Curse-of-dimensionality mitigation: project to ~30 PCA dims and renormalise.
    # The same PCA model is reused across phase1 and phase2.
    reduced, pca_model = _reduce_dimensions(normalized)
    logger.info(
        "tuner: dim %d -> %d (pca=%s)",
        embeddings.shape[1], reduced.shape[1], pca_model is not None,
    )

    # Phase 1: sampling -> sweep.
    sample_indices = _sample_indices(n_samples, SWEEP_SAMPLE_SIZE, seed=RANDOM_STATE)
    sample = reduced[sample_indices]
    logger.info(
        "tuner: phase1 sweep on %d / %d points (hdbscan=%s)",
        sample.shape[0], n_samples, _HDBSCAN_AVAILABLE,
    )

    trials = _run_all_sweeps(sample, min_clusters, max_clusters)

    winner = _select_winner(trials, sample_size=sample.shape[0])
    logger.info(
        "phase1 winner: %s params=%s sample_score=%.4f",
        winner["algorithm"], winner["params"], winner["silhouette"],
    )

    # Phase 2: apply the winning config to the full (PCA-reduced) data.
    logger.info("phase2: apply winner to full data (N=%d)", n_samples)
    final_labels = _fit_winner_on_full_data(
        normalized=reduced,
        sample_size=sample.shape[0],
        sample_labels=winner["sample_labels"],
        algorithm=winner["algorithm"],
        params=winner["params"],
    )

    n_clusters, n_noise = _count_clusters(final_labels)
    final_score = _evaluate_silhouette_on_subsample(
        normalized=reduced,
        labels=final_labels,
        max_points=SWEEP_SAMPLE_SIZE,
    )
    if final_score is None:
        # Collapses to a single cluster → score unusable; mark as -inf.
        final_score = float("-inf")

    logger.info(
        "Selected: %s params=%s final_score=%.4f clusters=%d noise=%d",
        winner["algorithm"], winner["params"], final_score, n_clusters, n_noise,
    )

    return BestConfig(
        algorithm=winner["algorithm"],
        params=winner["params"],
        silhouette=final_score,
        labels=final_labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        all_trials=[_trial_log(t) for t in trials],
        sweep_sample_size=int(sample.shape[0]),
        dim_before_pca=int(embeddings.shape[1]),
        dim_after_pca=int(reduced.shape[1]),
    )


def _run_all_sweeps(
    sample: np.ndarray, min_clusters: int, max_clusters: int
) -> list[dict[str, Any]]:
    """Run KMeans / DBSCAN / (optional) HDBSCAN sweeps, returning one flat trial list."""
    trials: list[dict[str, Any]] = []
    trials.extend(_sweep_kmeans(sample, min_clusters, max_clusters))
    trials.extend(_sweep_dbscan(sample))
    if _HDBSCAN_AVAILABLE:
        trials.extend(_sweep_hdbscan(sample))
    return trials


def _select_winner(
    trials: list[dict[str, Any]],
    sample_size: int,
) -> dict[str, Any]:
    """Apply the usability filter and pick the highest-scoring candidate.

    If every candidate trips the noise-ratio filter, log a warning and fall
    back to scoring alone so the pipeline still produces output.
    """
    def _has_score(t: dict[str, Any]) -> bool:
        return t["silhouette"] is not None and t["silhouette"] > -1.0

    within_noise_budget = [
        t for t in trials
        if _has_score(t) and (t["n_noise"] / sample_size) <= MAX_NOISE_RATIO_FOR_SELECTION
    ]
    if within_noise_budget:
        return max(within_noise_budget, key=lambda t: t["silhouette"])

    logger.warning(
        "All candidates have noise ratio > %.0f%%. Relaxing the filter.",
        MAX_NOISE_RATIO_FOR_SELECTION * 100,
    )
    fallback = [t for t in trials if _has_score(t)]
    if not fallback:
        raise RuntimeError("No valid clustering candidate was produced")
    return max(fallback, key=lambda t: t["silhouette"])


# ---------------------------------------------------------------------------
# Sweep helpers (each wraps a method-specific model factory in a common loop)
# ---------------------------------------------------------------------------


def _run_sweep(
    *,
    sample: np.ndarray,
    algorithm: Algorithm,
    candidates: list[Any],
    build_model: Callable[[Any], Any],
    params_of: Callable[[Any], dict[str, Any]],
    desc: str,
    unit: str,
) -> list[dict[str, Any]]:
    """Generic sweep loop shared by all three algorithms.

    For each candidate, instantiate the model, fit+predict on the sample,
    and record a uniformly-shaped trial dict.
    """
    trials: list[dict[str, Any]] = []
    for candidate in tqdm(candidates, desc=desc, unit=unit, leave=False):
        model = build_model(candidate)
        labels = model.fit_predict(sample)
        trials.append(_build_trial(algorithm, params_of(candidate), labels, sample))
    return trials


def _build_trial(
    algorithm: Algorithm,
    params: dict[str, Any],
    labels: np.ndarray,
    sample: np.ndarray,
) -> dict[str, Any]:
    """Assemble the standard trial dict consumed by the selector and reporter."""
    n_clusters, n_noise = _count_clusters(labels)
    return {
        "algorithm": algorithm,
        "params": params,
        "sample_labels": labels,
        "silhouette": _silhouette_on_sample(sample, labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


def _sweep_kmeans(
    sample: np.ndarray, min_k: int, max_k: int
) -> list[dict[str, Any]]:
    """Sweep MiniBatchKMeans over the K candidate set."""
    n = sample.shape[0]
    candidates = [k for k in K_CANDIDATES if max(2, min_k) <= k <= min(max_k, n - 1)]
    if not candidates:
        candidates = [min(max(2, min_k), n - 1)]

    def _build(k: int) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=3,
            batch_size=min(512, max(128, n // 4)),
            max_iter=100,
        )

    return _run_sweep(
        sample=sample,
        algorithm="kmeans",
        candidates=candidates,
        build_model=_build,
        params_of=lambda k: {"k": k},
        desc="KMeans sweep",
        unit="K",
    )


def _sweep_dbscan(sample: np.ndarray) -> list[dict[str, Any]]:
    """Sweep DBSCAN over the ``eps`` grid."""

    def _build(eps: float) -> DBSCAN:
        return DBSCAN(
            eps=float(eps),
            min_samples=DBSCAN_MIN_SAMPLES,
            # Euclidean is fine because the input is already L2-normalised.
            metric="euclidean",
            n_jobs=-1,
        )

    return _run_sweep(
        sample=sample,
        algorithm="dbscan",
        candidates=list(DBSCAN_EPS_GRID),
        build_model=_build,
        params_of=lambda eps: {"eps": float(eps), "min_samples": DBSCAN_MIN_SAMPLES},
        desc="DBSCAN sweep",
        unit="eps",
    )


def _sweep_hdbscan(sample: np.ndarray) -> list[dict[str, Any]]:
    """Sweep HDBSCAN over the ``min_cluster_size`` candidate set."""
    n = sample.shape[0]
    candidates = [m for m in HDBSCAN_MCS_CANDIDATES if m < n]
    if not candidates:
        candidates = [max(2, n // 4)]

    def _build(mcs: int) -> "hdbscan.HDBSCAN":
        return hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )

    return _run_sweep(
        sample=sample,
        algorithm="hdbscan",
        candidates=candidates,
        build_model=_build,
        params_of=lambda mcs: {
            "min_cluster_size": mcs,
            "min_samples": HDBSCAN_MIN_SAMPLES,
        },
        desc="HDBSCAN sweep",
        unit="mcs",
    )


# ---------------------------------------------------------------------------
# Phase 2: apply the winner to the full data
# ---------------------------------------------------------------------------


def _fit_winner_on_full_data(
    normalized: np.ndarray,
    sample_size: int,
    sample_labels: np.ndarray,
    algorithm: Algorithm,
    params: dict[str, Any],
) -> np.ndarray:
    """Run the winning config on the full data and return labels.

    Always refit rather than propagating labels from the sample. Propagation
    from a 1.5k sample to 8k+ points tends to classify 80-90% of rows as noise
    when the sample doesn't span the full manifold.
    """
    if normalized.shape[0] == sample_size:
        # Sample equals the whole dataset — nothing to refit.
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


def _reduce_dimensions(
    x: np.ndarray,
    target_dim: int = PCA_TARGET_DIM,
    threshold: int = PCA_DIM_THRESHOLD,
) -> tuple[np.ndarray, PCA | None]:
    """PCA to ``target_dim`` components, then re-normalise to unit length.

    Returns ``(reduced_array, fitted_pca_or_None)``. PCA is skipped when the
    input dimensionality is at or below ``threshold``.
    """
    n, dim = x.shape
    if dim <= threshold:
        return x, None
    n_components = min(target_dim, dim, max(1, n - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(x)
    # Re-normalise after projection to keep cosine equivalence.
    reduced = utils.l2_normalize(reduced)
    return reduced.astype(np.float32, copy=False), pca


def _sample_indices(n: int, cap: int, seed: int) -> np.ndarray:
    """Return the index array used for evaluation / fitting."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=cap, replace=False)


def _silhouette_on_sample(sample: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute cosine silhouette on the whole sample (noise excluded).

    Returns ``None`` when fewer than 2 valid clusters remain after dropping noise.
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
        logger.debug("silhouette computation failed: %s", exc)
        return None


def _evaluate_silhouette_on_subsample(
    normalized: np.ndarray, labels: np.ndarray, max_points: int
) -> float | None:
    """Evaluate silhouette on at most ``max_points`` rows (random subsample)."""
    n = normalized.shape[0]
    if n <= max_points:
        return _silhouette_on_sample(normalized, labels)

    rng = np.random.default_rng(RANDOM_STATE + 7)
    idx = rng.choice(n, size=max_points, replace=False)
    return _silhouette_on_sample(normalized[idx], labels[idx])


def _count_clusters(labels: np.ndarray) -> tuple[int, int]:
    """Return ``(number_of_valid_clusters, noise_count)``."""
    unique = set(labels.tolist())
    n_noise = int((labels == -1).sum())
    n_clusters = len(unique - {-1})
    return n_clusters, n_noise


def _trial_log(trial: dict[str, Any]) -> dict[str, Any]:
    """Compact dict suitable for ``all_trials`` (drops heavy fields like sample_labels)."""
    return {
        "algorithm": trial["algorithm"],
        "params": trial["params"],
        "silhouette": trial["silhouette"],
        "n_clusters": int(trial["n_clusters"]),
        "n_noise": int(trial["n_noise"]),
    }
