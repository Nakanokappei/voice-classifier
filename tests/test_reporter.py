"""reporter モジュールのテスト — 4ファイル出力と分類ロジック."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import reporter
from src.clusterer import ClusterSummary
from src.tuner import BestConfig


def _make_best(trials: list[dict]) -> BestConfig:
    """テスト用の BestConfig を合成."""
    return BestConfig(
        algorithm="kmeans",
        params={"k": 3},
        silhouette=0.35,
        labels=np.array([0, 0, 1, 1, 2, 2]),
        n_clusters=3,
        n_noise=0,
        all_trials=trials,
        sweep_sample_size=1500,
        dim_before_pca=1536,
        dim_after_pca=30,
    )


def _make_df_and_summaries() -> tuple[pd.DataFrame, list[ClusterSummary]]:
    df = pd.DataFrame(
        {
            "_normalized_text": ["a0", "a1", "b0", "b1", "c0", "c1"],
            "_duplicate_count": [1, 1, 1, 1, 1, 1],
        }
    )
    summaries = [
        ClusterSummary(cluster_id=0, size=2, representative_indices=[0, 1], representative_texts=["a0", "a1"]),
        ClusterSummary(cluster_id=1, size=2, representative_indices=[2, 3], representative_texts=["b0", "b1"]),
        ClusterSummary(cluster_id=2, size=2, representative_indices=[4, 5], representative_texts=["c0", "c1"]),
    ]
    return df, summaries


def test_write_report_produces_four_files(tmp_path: Path) -> None:
    """report.md / parameter_search.md / clusters.csv / params.json の4つが生成される."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35, "n_clusters": 3, "n_noise": 0},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df,
        labels=best.labels,
        summaries=summaries,
        best=best,
        input_path="/tmp/input.csv",
        text_col="対応内容",
    )

    for name in ("report.md", "parameter_search.md", "clusters.csv", "params.json"):
        assert (tmp_path / name).exists(), f"{name} が生成されていない"


def test_report_md_excludes_trial_history(tmp_path: Path) -> None:
    """report.md にスイープ詳細テーブルが含まれないこと（parameter_search.md に分離）."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35, "n_clusters": 3, "n_noise": 0},
        {"algorithm": "dbscan", "params": {"eps": 0.45, "min_samples": 5}, "silhouette": 0.7, "n_clusters": 2, "n_noise": 1450},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries,
        best=best, input_path="/tmp/in.csv", text_col="text",
    )

    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    # 試行詳細テーブルの構造要素（列ヘッダや除外セクションの見出し）は残らない
    assert "試行履歴" not in report_text
    assert "採用候補ランキング" not in report_text
    assert "ノイズ率フィルタ" not in report_text
    # DBSCAN 候補の詳細行は report.md に出ない
    assert "eps=0.45" not in report_text
    # parameter_search.md への参照がある
    assert "parameter_search.md" in report_text


def test_parameter_search_separates_accepted_and_rejected(tmp_path: Path) -> None:
    """parameter_search.md が採用/除外を分類して表示すること."""
    df, summaries = _make_df_and_summaries()
    trials = [
        # 採用候補（ノイズ率 0%）
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35, "n_clusters": 3, "n_noise": 0},
        # ノイズ率超過（97%）
        {"algorithm": "dbscan", "params": {"eps": 0.35, "min_samples": 5}, "silhouette": 0.86, "n_clusters": 6, "n_noise": 1457},
        # 退化（シルエット算出不能）
        {"algorithm": "dbscan", "params": {"eps": 0.25, "min_samples": 5}, "silhouette": None, "n_clusters": 1, "n_noise": 1495},
    ]
    best = _make_best(trials=trials)

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries,
        best=best, input_path="/tmp/in.csv", text_col="text",
    )

    search_text = (tmp_path / "parameter_search.md").read_text(encoding="utf-8")
    # 3つのセクションが存在
    assert "採用候補ランキング" in search_text
    assert "除外: ノイズ率フィルタ" in search_text
    assert "除外: クラスタ構造が成立せず" in search_text
    # 採用候補は ✓ マーク
    assert "✓ 採用" in search_text
    # 各候補が該当セクションに入っている
    assert "eps=0.350" in search_text and "97" in search_text  # noise ratio displayed
    assert "eps=0.250" in search_text


def test_params_json_includes_search_metadata(tmp_path: Path) -> None:
    """params.json に探索メタ情報（PCA・サンプル数・フィルタ閾値）が記録される."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35, "n_clusters": 3, "n_noise": 0},
    ])
    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries,
        best=best, input_path="/tmp/in.csv", text_col="text",
    )

    with (tmp_path / "params.json").open() as f:
        payload = json.load(f)
    assert payload["sweep_sample_size"] == 1500
    assert payload["dim_before_pca"] == 1536
    assert payload["dim_after_pca"] == 30
    assert payload["max_noise_ratio_filter"] == best.max_noise_ratio
