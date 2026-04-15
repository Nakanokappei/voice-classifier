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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from . import utils

logger = logging.getLogger(__name__)

# HDBSCAN is optional; it is occasionally painful to install on some platforms.
try:  # pragma: no cover - environment-dependent
    import hdbscan  # type: ignore[import-not-found]

    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HDBSCAN_AVAILABLE = False
    logger.info("hdbscan is not installed; HDBSCAN candidates will be skipped")

# Leiden requires three packages; each is optional.  Missing any of them
# disables the leiden sweep without failing the rest of the pipeline.
try:  # pragma: no cover - environment-dependent
    import hnswlib  # type: ignore[import-not-found]
    import igraph as _ig  # type: ignore[import-not-found]
    import leidenalg as _la  # type: ignore[import-not-found]

    _LEIDEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LEIDEN_AVAILABLE = False
    logger.info(
        "hnswlib / python-igraph / leidenalg not installed; "
        "Leiden candidates will be skipped"
    )


Algorithm = Literal["kmeans", "hdbscan", "leiden"]

# Number of points used to fit each candidate during the sweep.
#
# Measurement (data/input/customer_support_tickets.csv, 14k unique rows):
#   N      phase3 time    silhouette scoring time
#   1500   1.0s           11ms / candidate
#   3000   2.5s           39ms / candidate
#   5000   5.8s           122ms / candidate
#   8000   13.5s          309ms / candidate
#
# Most of the cost at large N is the O(n²) silhouette score, so we cap
# scoring via sklearn's `sample_size=` parameter (see SILHOUETTE_SAMPLE_CAP)
# and keep the fit sample big enough to represent small clusters.
# N=5000 covers typical datasets while keeping phase 3 under ~4 seconds.
SWEEP_SAMPLE_SIZE: int = 5000

# Silhouette scoring is capped at this many rows via `silhouette_score
# (sample_size=...)` so O(n²) pairwise distances stay bounded regardless of
# SWEEP_SAMPLE_SIZE. 2000 gives stable scores (measured std < 0.01) at
# roughly 17 ms per candidate.
SILHOUETTE_SAMPLE_CAP: int = 2000

# If the dataset is no larger than this, skip subsampling entirely and run
# the sweep on the full data. Between SWEEP_SAMPLE_SIZE and this threshold,
# the added runtime of using all rows is small compared to the variance
# introduced by discarding up to half the data.
SWEEP_FULL_DATA_THRESHOLD: int = 10000

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

# Leiden community detection on an HNSW k-NN graph (inspired by BERTopic /
# CKPS). Resolution controls the granularity: lower resolution = fewer, larger
# communities; higher resolution = more, smaller communities.
LEIDEN_RESOLUTION_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0, 1.3, 1.7)
# k-NN neighbours per node in the HNSW graph.  The default of 15 tracks CKPS.
LEIDEN_DEFAULT_NEIGHBOURS: int = 15
# HNSW index construction parameters.  These rarely need tuning.
LEIDEN_HNSW_EF_CONSTRUCTION: int = 200
LEIDEN_HNSW_M: int = 16

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

    # Phase 1: sample + sweep.
    # For "small" datasets (≤ SWEEP_FULL_DATA_THRESHOLD) we keep all rows so
    # the winning config is chosen on the true distribution. Beyond that we
    # subsample to SWEEP_SAMPLE_SIZE to keep the sweep interactive.
    if n_samples <= SWEEP_FULL_DATA_THRESHOLD:
        sample_indices = np.arange(n_samples)
    else:
        sample_indices = _sample_indices(
            n_samples, SWEEP_SAMPLE_SIZE, seed=RANDOM_STATE
        )
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
    """Run every available sweep, returning one flat trial list."""
    trials: list[dict[str, Any]] = []
    trials.extend(_sweep_kmeans(sample, min_clusters, max_clusters))
    if _HDBSCAN_AVAILABLE:
        trials.extend(_sweep_hdbscan(sample))
    if _LEIDEN_AVAILABLE:
        trials.extend(_sweep_leiden(sample))
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
) -> list[dict[str, Any]]:
    """Generic sweep loop shared by all three algorithms.

    For each candidate, instantiate the model, fit+predict on the sample,
    and record a uniformly-shaped trial dict.

    Note: no per-candidate progress bar is rendered here. The whole sweep
    completes in a few seconds on typical input sizes, and the pipeline
    shows a spinner on the parent "[3/N]" line for visual feedback.
    """
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
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
    )


def _sweep_leiden(sample: np.ndarray) -> list[dict[str, Any]]:
    """Sweep Leiden community detection over the resolution grid.

    A single HNSW k-NN graph is built once for the sample, then Leiden is run
    at each resolution.  The graph-construction cost is amortised across the
    whole sweep; only the Leiden partition is re-computed per resolution.
    """
    n = sample.shape[0]
    # Guard against pathological cases where the sample is smaller than k.
    n_neighbors = min(LEIDEN_DEFAULT_NEIGHBOURS, max(2, n - 1))
    graph = _build_knn_graph(sample, n_neighbors=n_neighbors)

    trials: list[dict[str, Any]] = []
    for resolution in LEIDEN_RESOLUTION_GRID:
        labels = _leiden_partition(graph, resolution=float(resolution))
        trials.append(
            _build_trial(
                algorithm="leiden",
                params={"resolution": float(resolution), "n_neighbors": n_neighbors},
                labels=labels,
                sample=sample,
            )
        )
    return trials


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

    if algorithm == "leiden":
        # Rebuild the k-NN graph on the full data and re-run Leiden at the
        # winning resolution.  HNSW keeps this O(N log N) instead of O(N²).
        n_neighbors = int(params.get("n_neighbors", LEIDEN_DEFAULT_NEIGHBOURS))
        # For very small datasets, clamp k so hnswlib has something to build.
        n_neighbors = min(n_neighbors, max(2, normalized.shape[0] - 1))
        graph = _build_knn_graph(normalized, n_neighbors=n_neighbors)
        return _leiden_partition(graph, resolution=float(params["resolution"]))

    # hdbscan
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    return model.fit_predict(normalized).astype(np.int64)


# ---------------------------------------------------------------------------
# Leiden helpers (HNSW k-NN graph + Leiden community detection)
# ---------------------------------------------------------------------------


def _build_knn_graph(
    vectors: np.ndarray,
    n_neighbors: int,
) -> "_ig.Graph":
    """Build an HNSW k-NN graph with cosine similarity edge weights.

    Steps:
      1. Index the vectors in an HNSW graph (`space="cosine"`).
      2. Query the k nearest neighbours for each point.
      3. Turn the result into an igraph weighted graph whose edge weights
         are cosine similarities in ``[0, 1]``.

    The returned graph feeds directly into ``_leiden_partition``.
    """
    n_points, dim = vectors.shape

    # HNSW index (approximate k-NN, O(N log N)).
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=n_points,
        ef_construction=LEIDEN_HNSW_EF_CONSTRUCTION,
        M=LEIDEN_HNSW_M,
    )
    # hnswlib needs a float32 contiguous array.
    index.add_items(
        np.ascontiguousarray(vectors, dtype=np.float32),
        np.arange(n_points),
    )
    # Set query-time ef; 2x k (clamped to a floor) is a solid default.
    index.set_ef(max(n_neighbors * 2, 50))

    neighbour_indices, neighbour_distances = index.knn_query(
        np.ascontiguousarray(vectors, dtype=np.float32),
        k=n_neighbors,
    )

    # Build the edge list. hnswlib returns cosine *distance* in [0, 2];
    # convert to cosine *similarity* in [0, 1] for the edge weights.
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n_points):
        for j_idx in range(n_neighbors):
            j = int(neighbour_indices[i][j_idx])
            if i == j:
                # Skip the self-loop that hnswlib always returns.
                continue
            similarity = max(1.0 - float(neighbour_distances[i][j_idx]), 0.0)
            edges.append((i, j))
            weights.append(similarity)

    graph = _ig.Graph(n=n_points, edges=edges, directed=False)
    graph.es["weight"] = weights
    # Collapse duplicate edges (the graph is undirected but hnswlib may
    # report i→j and j→i separately); keep the stronger weight.
    graph.simplify(combine_edges="max")
    return graph


def _leiden_partition(
    graph: "_ig.Graph", resolution: float
) -> np.ndarray:
    """Run Leiden community detection and return a numpy label array."""
    partition = _la.find_partition(
        graph,
        _la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )
    return np.array(partition.membership, dtype=np.int64)


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
    """Compute cosine silhouette on the sample (noise excluded).

    Uses sklearn's built-in ``sample_size`` to cap the O(n²) pairwise distance
    matrix so cost stays roughly constant regardless of ``sample.shape[0]``.

    Returns ``None`` when fewer than 2 valid clusters remain after dropping noise.
    """
    mask = labels != -1
    if mask.sum() < 2:
        return None
    labels_valid = labels[mask]
    if len(np.unique(labels_valid)) < 2:
        return None
    x_valid = sample[mask]
    # silhouette_score requires sample_size ≤ number of available rows.
    scoring_cap = min(SILHOUETTE_SAMPLE_CAP, x_valid.shape[0])
    try:
        return float(
            silhouette_score(
                x_valid,
                labels_valid,
                metric="cosine",
                sample_size=scoring_cap,
                random_state=RANDOM_STATE + 7,
            )
        )
    except ValueError as exc:
        logger.debug("silhouette computation failed: %s", exc)
        return None


def _evaluate_silhouette_on_subsample(
    normalized: np.ndarray, labels: np.ndarray, max_points: int
) -> float | None:
    """Evaluate silhouette on at most ``max_points`` rows (random subsample).

    The internal `_silhouette_on_sample` already caps scoring via
    ``SILHOUETTE_SAMPLE_CAP``; this outer subsample is an additional guard for
    very large full-data arrays so we don't pass 100k+ rows into the routine.
    """
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
