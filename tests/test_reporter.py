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


def test_write_report_produces_expected_files(tmp_path: Path) -> None:
    """report.md / parameter_search.md / clusters.csv / <入力名>_classified.csv / params.json が生成される."""
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

    # parameter_search.md は廃止. parameter_search.html は format によらず常時生成.
    expected = [
        "report.md",
        "parameter_search.html",
        "clusters.csv",
        "input_classified.csv",  # `input.csv` の stem が input
        "params.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"{name} が生成されていない"
    # parameter_search.md は生成されない
    assert not (tmp_path / "parameter_search.md").exists()


def test_clusters_csv_is_cluster_list(tmp_path: Path) -> None:
    """clusters.csv はクラスタ1件=1行のサマリ. 元データ行単位ではない."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/tmp/my_data.csv", text_col="text",
        cluster_names={0: "Aグループ", 1: "Bグループ", 2: "Cグループ"},
    )

    import pandas as pd
    clusters = pd.read_csv(tmp_path / "clusters.csv")
    # 行数 = クラスタ数
    assert len(clusters) == 3
    # 必須列
    for col in ("cluster_id", "cluster_name", "size", "rep_1"):
        assert col in clusters.columns
    # 最大サイズのクラスタが先頭（全てサイズ2なのでIDで確認）
    assert set(clusters["cluster_id"]) == {0, 1, 2}
    # 名前が反映されている
    assert "Aグループ" in clusters["cluster_name"].tolist()


def test_classified_csv_name_includes_input_stem(tmp_path: Path) -> None:
    """classified CSV は <入力ファイル名>_classified.csv の形式."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/some/path/repair_full.csv", text_col="text",
    )

    assert (tmp_path / "repair_full_classified.csv").exists()
    # 行数 = 元データ件数
    import pandas as pd
    classified = pd.read_csv(tmp_path / "repair_full_classified.csv")
    assert len(classified) == len(df)
    assert "cluster_id" in classified.columns


def test_cluster_list_places_noise_at_end(tmp_path: Path) -> None:
    """clusters.csv はノイズクラスタを末尾に配置する."""
    import numpy as np
    df = pd.DataFrame({"_normalized_text": ["a", "b", "c", "d"],
                       "_duplicate_count": [1, 1, 1, 1]})
    summaries_with_noise = [
        ClusterSummary(cluster_id=0, size=2, representative_indices=[0, 1],
                       representative_texts=["a", "b"]),
        ClusterSummary(cluster_id=1, size=1, representative_indices=[2],
                       representative_texts=["c"]),
        ClusterSummary(cluster_id=-1, size=1,
                       representative_indices=[], representative_texts=[]),
    ]
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 2}, "silhouette": 0.3,
         "n_clusters": 2, "n_noise": 1},
    ])
    best.labels = np.array([0, 0, 1, -1])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries_with_noise, best=best,
        input_path="/t/data.csv", text_col="text",
    )

    clusters = pd.read_csv(tmp_path / "clusters.csv")
    # 最終行がノイズ
    assert int(clusters.iloc[-1]["cluster_id"]) == -1
    # ノイズは「未分類」名
    assert clusters.iloc[-1]["cluster_name"] == "未分類"


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

    # parameter_search は HTML 一本化されたので、HTML 内の Markdown 変換結果を検査
    search_text = (tmp_path / "parameter_search.html").read_text(encoding="utf-8")
    # 3つのセクションが存在（h2 として変換される）
    assert "採用候補ランキング" in search_text
    assert "除外: ノイズ率フィルタ" in search_text
    assert "除外: クラスタ構造が成立せず" in search_text
    # 採用候補は ✓ マーク
    assert "✓ 採用" in search_text
    # 各候補が該当セクションに入っている
    assert "eps=0.350" in search_text and "97" in search_text
    assert "eps=0.250" in search_text
    # SVG チャートが先頭に埋め込まれている
    assert "<svg" in search_text
    assert "シルエットスコア" in search_text
    assert "クラスタ数" in search_text


def test_html_format_produces_html_files(tmp_path: Path) -> None:
    """output_format='html' で HTML ファイルが生成され、Markdown は生成されない."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/tmp/in.csv", text_col="text",
        output_format="html",
    )

    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "parameter_search.html").exists()
    # Markdown は生成しない
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "parameter_search.md").exists()

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    # 基本構造が含まれる
    assert "<!DOCTYPE html>" in html
    assert "<title>" in html
    assert "<table>" in html  # 概要テーブルなど
    # CSS がインライン埋め込み
    assert "<style>" in html
    assert ":root" in html


def test_both_format_produces_md_and_html_for_report(tmp_path: Path) -> None:
    """output_format='both' は report を md+html、parameter_search は html のみ."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/tmp/in.csv", text_col="text",
        output_format="both",
    )

    # report は両形式
    for name in ("report.md", "report.html"):
        assert (tmp_path / name).exists(), f"{name} が生成されていない"
    # parameter_search は HTML のみ
    assert (tmp_path / "parameter_search.html").exists()
    assert not (tmp_path / "parameter_search.md").exists()


def test_report_uses_llm_summary_as_main_representative(tmp_path: Path) -> None:
    """cluster_annotations が渡されたとき、要約がメイン代表テキストとして表示される."""
    from src.namer import ClusterAnnotation

    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])
    annotations = {
        0: ClusterAnnotation(
            label="Aグループ",
            summary="クラスタ0は A 系の問い合わせが多く見られます.",
        ),
        1: ClusterAnnotation(
            label="Bグループ",
            summary="クラスタ1は B 系の問い合わせが中心です.",
        ),
        2: ClusterAnnotation(
            label="Cグループ",
            summary="クラスタ2は C 系の問い合わせをまとめたものです.",
        ),
    }

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/tmp/x.csv", text_col="text",
        cluster_annotations=annotations,
    )

    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    # 要約が代表テキストとして出ている
    assert "クラスタ0は A 系の問い合わせが多く見られます" in report_text
    # 重心付近の実データセクションが常時可視で存在
    assert "**重心に近い実データ" in report_text
    # 折りたたみではなく直接可視
    assert "<details>" not in report_text
    # ラベルが見出しに反映
    assert "Aグループ" in report_text


def test_clusters_csv_includes_summary_column_when_annotations_given(
    tmp_path: Path,
) -> None:
    """cluster_annotations があると clusters.csv に summary 列が増える."""
    from src.namer import ClusterAnnotation

    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])
    annotations = {
        0: ClusterAnnotation(label="A", summary="Aの要約文"),
        1: ClusterAnnotation(label="B", summary="Bの要約文"),
        2: ClusterAnnotation(label="C", summary="Cの要約文"),
    }

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/tmp/x.csv", text_col="text",
        cluster_annotations=annotations,
    )

    import pandas as pd
    clusters = pd.read_csv(tmp_path / "clusters.csv")
    assert "summary" in clusters.columns
    assert "Aの要約文" in clusters["summary"].tolist()


def test_clusters_csv_omits_summary_without_annotations(tmp_path: Path) -> None:
    """annotations 無しの時は summary 列を追加しない（列が増えないこと）."""
    df, summaries = _make_df_and_summaries()
    best = _make_best(trials=[
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35,
         "n_clusters": 3, "n_noise": 0},
    ])

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries, best=best,
        input_path="/tmp/x.csv", text_col="text",
    )

    import pandas as pd
    clusters = pd.read_csv(tmp_path / "clusters.csv")
    assert "summary" not in clusters.columns


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
