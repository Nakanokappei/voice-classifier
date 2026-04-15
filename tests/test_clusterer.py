"""Tests for the clusterer module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import clusterer


def test_summarize_picks_centroid_closest_texts() -> None:
    """各クラスタで重心に近い順に代表テキストが返ること.

    Note: after L2 normalisation, the centroid approximates the angular mean,
    so the axis-aligned point is the one closest to the centroid.
    """
    # Two clusters around [1,0] and [0,1], symmetric around +y/-y
    vectors = np.array(
        [
            # cluster 0 — arranged so the angular mean points to [1, 0]
            [1.0, 0.0],      # Centroid direction (nearest)
            [1.0, 0.3],      # +y side
            [1.0, -0.3],     # -y side (cancels out)
            # cluster 1 — same but pointing to [0, 1]
            [0.0, 1.0],
            [0.3, 1.0],
            [-0.3, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    df = pd.DataFrame(
        {
            "_normalized_text": [
                "c0 center",
                "c0 plus",
                "c0 minus",
                "c1 center",
                "c1 plus",
                "c1 minus",
            ],
        }
    )

    summaries = clusterer.summarize_clusters(df, vectors, labels, top_k=2)

    # cluster_id 昇順に並ぶ
    assert summaries[0].cluster_id == 0
    assert summaries[1].cluster_id == 1

    # The axis-aligned "center" ranks first
    assert summaries[0].representative_texts[0] == "c0 center"
    assert summaries[1].representative_texts[0] == "c1 center"
    assert len(summaries[0].representative_texts) == 2


def test_noise_cluster_is_appended_at_end_without_reps() -> None:
    """Noise clusters go to the tail with no representatives."""
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],  # only this point is noise
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, -1])
    df = pd.DataFrame(
        {"_normalized_text": ["a", "b", "c", "noise"]}
    )

    summaries = clusterer.summarize_clusters(df, vectors, labels, top_k=3)

    assert summaries[-1].cluster_id == -1
    assert summaries[-1].size == 1
    assert summaries[-1].representative_texts == []


def test_shape_mismatch_raises() -> None:
    """Mismatched row counts raise ValueError."""
    df = pd.DataFrame({"_normalized_text": ["a", "b"]})
    vectors = np.zeros((3, 2), dtype=np.float32)
    labels = np.array([0, 0, 0])

    try:
        clusterer.summarize_clusters(df, vectors, labels)
    except ValueError as exc:
        assert "Length mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError was not raised")
