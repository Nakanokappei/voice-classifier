"""Post-processing of clustering results — representative text extraction.

Responsibilities:
    - Compute each cluster's centroid and pick the top-N rows closest to it.
    - Noise labels (-1) are grouped as "unassigned" without representative picks.
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
    """Per-cluster summary.

    Attributes:
        cluster_id: cluster id (-1 denotes noise)
        size: number of samples in the cluster
        representative_indices: row indices ordered by distance to the centroid
        representative_texts: corresponding normalised texts
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
    """Produce per-cluster summaries by picking the top-K rows nearest each centroid.

    Args:
        df: DataFrame containing `_normalized_text` (row-aligned with `labels`)
        embeddings: shape (N, D) embedding matrix
        labels: cluster id for each row, shape (N,)
        top_k: number of representative rows to pick per cluster

    Returns:
        ClusterSummary list sorted by ascending cluster id; noise (if any) is
        appended at the end.
    """
    if len(df) != embeddings.shape[0] or len(df) != len(labels):
        raise ValueError(
            f"Length mismatch: df={len(df)} embeddings={embeddings.shape[0]} "
            f"labels={len(labels)}"
        )

    # Normalise up-front so centroid math and cosine distance stay consistent.
    normalized = _l2_normalize(embeddings)

    summaries: list[ClusterSummary] = []
    unique_labels = sorted(set(labels.tolist()))
    # Put noise at the tail.
    ordered = [c for c in unique_labels if c != NOISE_LABEL]
    if NOISE_LABEL in unique_labels:
        ordered.append(NOISE_LABEL)

    for cluster_id in ordered:
        member_mask = labels == cluster_id
        member_indices = np.where(member_mask)[0]
        size = int(member_indices.size)

        if cluster_id == NOISE_LABEL:
            # Noise carries no representative picks — only the count.
            summaries.append(
                ClusterSummary(
                    cluster_id=NOISE_LABEL,
                    size=size,
                    representative_indices=[],
                    representative_texts=[],
                )
            )
            continue

        # Pick the top-K rows closest to the centroid.
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
    """Normalise rows to unit L2 length; replace zero-norm rows with `e_0`.

    Mirrors the rule used in `tuner._l2_normalize` so that downstream math
    (`member_vectors @ centroid`) never encounters divide-by-zero.
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
    """Return the top-K indices closest to the cluster centroid.

    Centroid = L2-normalised mean of `member_vectors`. Closeness is ranked by
    descending cosine similarity (equivalent to ascending cosine distance).
    """
    if member_vectors.shape[0] == 0:
        return []

    centroid = member_vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        # Extremely rare: all vectors cancel out. Fall back to input order.
        return member_indices[:top_k].tolist()
    centroid /= centroid_norm

    # `member_vectors` is already normalised, so zero rows do not occur here.
    # Guard the matmul against stray numerical noise anyway.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        similarities = member_vectors @ centroid

    # Push any NaN/Inf to the back of the ranking.
    similarities = np.nan_to_num(
        similarities, nan=-np.inf, posinf=np.inf, neginf=-np.inf
    )
    order = np.argsort(-similarities)[:top_k]
    return [int(member_indices[i]) for i in order]
