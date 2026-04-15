"""Tests for the reporter module — covers all output files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import reporter
from src.clusterer import ClusterSummary
from src.tuner import BestConfig


def _make_best(trials: list[dict]) -> BestConfig:
    """Assemble a BestConfig for tests."""
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
    """All expected files are produced."""
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

    # parameter_search.md is retired; parameter_search.html is always produced.
    expected = [
        "report.md",
        "parameter_search.html",
        "clusters.csv",
        "input_classified.csv",  # stem of input.csv is "input"
        "params.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"{name} was not generated"
    # parameter_search.md is never generated
    assert not (tmp_path / "parameter_search.md").exists()


def test_clusters_csv_is_cluster_list(tmp_path: Path) -> None:
    """clusters.csv is one-row-per-cluster, not per raw row."""
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
    # Row count matches cluster count
    assert len(clusters) == 3
    # Required columns
    for col in ("cluster_id", "cluster_name", "size", "rep_1"):
        assert col in clusters.columns
    # Largest cluster comes first (ties broken by id here)
    assert set(clusters["cluster_id"]) == {0, 1, 2}
    # Names are reflected in output
    assert "Aグループ" in clusters["cluster_name"].tolist()


def test_classified_csv_name_includes_input_stem(tmp_path: Path) -> None:
    """classified CSV uses <input stem>_classified.csv naming."""
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
    # Row count matches the original data
    import pandas as pd
    classified = pd.read_csv(tmp_path / "repair_full_classified.csv")
    assert len(classified) == len(df)
    assert "cluster_id" in classified.columns


def test_cluster_list_places_noise_at_end(tmp_path: Path) -> None:
    """Noise cluster is appended last in clusters.csv."""
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
    # Noise row comes last.
    assert int(clusters.iloc[-1]["cluster_id"]) == -1
    # Noise label is the English placeholder.
    assert clusters.iloc[-1]["cluster_name"] == "Unassigned"


def test_report_md_excludes_trial_history(tmp_path: Path) -> None:
    """Trial detail tables stay out of report.md (moved to parameter_search)."""
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
    # Search-detail section titles and candidate rows stay out of report.md.
    assert "Accepted Candidates" not in report_text
    assert "Rejected" not in report_text
    assert "eps=0.45" not in report_text
    # It does point to the HTML parameter-search report.
    assert "parameter_search.html" in report_text


def test_parameter_search_separates_accepted_and_rejected(tmp_path: Path) -> None:
    """parameter_search.html shows accepted/rejected sections."""
    df, summaries = _make_df_and_summaries()
    trials = [
        # Accepted candidate (noise ratio 0%)
        {"algorithm": "kmeans", "params": {"k": 3}, "silhouette": 0.35, "n_clusters": 3, "n_noise": 0},
        # Noise ratio over threshold (97%)
        {"algorithm": "dbscan", "params": {"eps": 0.35, "min_samples": 5}, "silhouette": 0.86, "n_clusters": 6, "n_noise": 1457},
        # Degenerate (silhouette not computable)
        {"algorithm": "dbscan", "params": {"eps": 0.25, "min_samples": 5}, "silhouette": None, "n_clusters": 1, "n_noise": 1495},
    ]
    best = _make_best(trials=trials)

    reporter.write_report(
        output_dir=tmp_path,
        df=df, labels=best.labels, summaries=summaries,
        best=best, input_path="/tmp/in.csv", text_col="text",
    )

    # parameter_search is HTML-only; inspect the rendered HTML for each section.
    search_text = (tmp_path / "parameter_search.html").read_text(encoding="utf-8")
    # Three section headings are present (rendered as <h2>).
    assert "Accepted Candidates" in search_text
    assert "Rejected — Noise-Ratio Filter" in search_text
    assert "Rejected — No Cluster Structure" in search_text
    # Selected candidate has the ✓ marker.
    assert "✓ Selected" in search_text
    # Each trial lands in its own section with its noise ratio.
    assert "eps=0.350" in search_text and "97" in search_text
    assert "eps=0.250" in search_text
    # The SVG chart is embedded at the top.
    assert "<svg" in search_text
    assert "Silhouette score" in search_text
    assert "Cluster count" in search_text


def test_html_format_produces_html_files(tmp_path: Path) -> None:
    """output_format=.html. produces HTML only; no Markdown emitted."""
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
    # No Markdown is emitted
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "parameter_search.md").exists()

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    # Basic HTML structure is present
    assert "<!DOCTYPE html>" in html
    assert "<title>" in html
    assert "<table>" in html  # Summary table etc.
    # CSS is embedded inline
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

    # report is emitted in both formats
    for name in ("report.md", "report.html"):
        assert (tmp_path / name).exists(), f"{name} が生成されていない"
    # parameter_search is HTML-only
    assert (tmp_path / "parameter_search.html").exists()
    assert not (tmp_path / "parameter_search.md").exists()


def test_report_uses_llm_summary_as_main_representative(tmp_path: Path) -> None:
    """When cluster_annotations are provided, the summary is used as the main rep text."""
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
    # The summary is shown as the representative text.
    assert "クラスタ0は A 系の問い合わせが多く見られます" in report_text
    # The raw-data section is always visible (not collapsed in <details>).
    assert "**Raw data near centroid" in report_text
    assert "<details>" not in report_text
    # The label appears in the cluster heading.
    assert "Aグループ" in report_text


def test_clusters_csv_includes_summary_column_when_annotations_given(
    tmp_path: Path,
) -> None:
    """With annotations, clusters.csv gains a summary column."""
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
    """Without annotations, no summary column is added."""
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
    """params.json records search metadata (PCA, sample count, filter threshold)."""
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
