"""clusterer モジュールのテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import clusterer


def test_summarize_picks_centroid_closest_texts() -> None:
    """各クラスタで重心に近い順に代表テキストが返ること.

    注: L2正規化後の重心は角度平均に近いため、対称に散らしたサンプルの
    中央値（= 重心方向）を "close" として配置している.
    """
    # 2 クラスタ: [1,0] 方向系と [0,1] 方向系（+y/-y に対称に散らす）
    vectors = np.array(
        [
            # cluster 0 — 角度平均が [1, 0] 方向になる対称配置
            [1.0, 0.0],      # 重心方向（= 一番近い）
            [1.0, 0.3],      # +y 側
            [1.0, -0.3],     # -y 側（対称に相殺）
            # cluster 1 — 角度平均が [0, 1] 方向になる対称配置
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

    # 重心方向の "center" が 1 位で選ばれる
    assert summaries[0].representative_texts[0] == "c0 center"
    assert summaries[1].representative_texts[0] == "c1 center"
    assert len(summaries[0].representative_texts) == 2


def test_noise_cluster_is_appended_at_end_without_reps() -> None:
    """ノイズクラスタは末尾で、代表テキストは空."""
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],  # これだけノイズ
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
    """行数が一致しない場合は ValueError."""
    df = pd.DataFrame({"_normalized_text": ["a", "b"]})
    vectors = np.zeros((3, 2), dtype=np.float32)
    labels = np.array([0, 0, 0])

    try:
        clusterer.summarize_clusters(df, vectors, labels)
    except ValueError as exc:
        assert "行数不整合" in str(exc)
    else:
        raise AssertionError("ValueError が送出されませんでした")
