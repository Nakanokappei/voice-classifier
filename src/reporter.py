"""Report writing — emits the four output artifacts.

Generated files:
    - report.md              Clustering result (selected config, per-cluster reps)
    - parameter_search.html  Parameter-search report with a dual-axis chart on top
    - clusters.csv           One row per cluster: id, name, size, summary, rep_1..N
    - <input>_classified.csv Original data plus `cluster_id` (+ `cluster_name` if
                             annotations are available)
    - params.json            Machine-readable metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .clusterer import ClusterSummary, NOISE_LABEL
from .namer import ClusterAnnotation
from .tuner import BestConfig

logger = logging.getLogger(__name__)

QUALITY_LABEL: dict[str, str] = {
    "good": "Good",
    "warn": "Acceptable (with caveats)",
    "poor": "Needs review",
}

OutputFormat = Literal["md", "html", "both"]

# Embedded CSS for HTML reports — kept inline so the file is self-contained.
_HTML_CSS = """
:root {
  --fg: #1f2328;
  --muted: #57606a;
  --accent: #0969da;
  --warn: #bf8700;
  --bad: #cf222e;
  --bg: #ffffff;
  --bg-alt: #f6f8fa;
  --border: #d0d7de;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans",
               "Noto Sans CJK JP", "Segoe UI", sans-serif;
  color: var(--fg); background: var(--bg);
  max-width: 960px; margin: 2rem auto; padding: 0 1.25rem;
  line-height: 1.6;
}
h1, h2, h3 { border-bottom: 1px solid var(--border); padding-bottom: .3em; }
h1 { font-size: 1.7rem; }
h2 { font-size: 1.35rem; margin-top: 2rem; }
h3 { font-size: 1.1rem; border-bottom: none; color: var(--accent); }
table { border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .92rem; }
th, td { border: 1px solid var(--border); padding: .45rem .7rem; text-align: left; }
th { background: var(--bg-alt); }
tr:nth-child(even) td { background: var(--bg-alt); }
code { background: var(--bg-alt); padding: .1em .35em; border-radius: 3px;
       font-size: .9em; }
blockquote { border-left: 4px solid var(--warn); background: #fff8c5;
             padding: .6rem 1rem; margin: 1rem 0; border-radius: 3px; }
ol li, ul li { margin: .2rem 0; }
hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
.footer { color: var(--muted); font-size: .85rem; margin-top: 3rem;
          text-align: right; }
""".strip()


def write_report(
    output_dir: Path | str,
    df: pd.DataFrame,
    labels: np.ndarray,
    summaries: list[ClusterSummary],
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
    cluster_names: dict[int, str] | None = None,
    cluster_annotations: dict[int, ClusterAnnotation] | None = None,
    output_format: OutputFormat = "md",
) -> None:
    """Write the output artifacts under `output_dir`.

    Args:
        output_dir: target directory (created if missing)
        df: `loader.load_csv` result
        labels: per-row cluster ids
        summaries: `clusterer.summarize_clusters` result
        best: `tuner.find_best_clustering` result
        input_path: input CSV path (embedded in the report)
        text_col: target text column name (embedded in the report)
        cluster_names: optional cluster_id -> label mapping
        cluster_annotations: optional cluster_id -> ClusterAnnotation mapping
        output_format: "md" / "html" / "both" for the report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = cluster_annotations or {}

    # Explicit cluster_names wins; otherwise derive from annotations.
    if cluster_names:
        names = cluster_names
    else:
        names = {cid: ann.label for cid, ann in annotations.items()}

    # Always-written machine-readable artifacts.
    input_stem = Path(str(input_path)).stem
    _write_classified_rows_csv(
        output_dir / f"{input_stem}_classified.csv", df, labels, names
    )
    _write_cluster_list_csv(
        output_dir / "clusters.csv", summaries, names, annotations
    )
    _write_params_json(output_dir / "params.json", best, input_path, text_col, names)

    # Markdown body for both clustering and parameter-search reports.
    clustering_md = _build_clustering_report_md(
        df=df, summaries=summaries, best=best,
        input_path=input_path, text_col=text_col,
        cluster_names=names, cluster_annotations=annotations,
    )
    parameter_search_md = _build_parameter_search_md(
        best=best, input_path=input_path, text_col=text_col,
    )

    want_md = output_format in ("md", "both")
    want_html = output_format in ("html", "both")

    # Clustering result: toggle md/html per `--format`.
    if want_md:
        (output_dir / "report.md").write_text(clustering_md, encoding="utf-8")
    if want_html:
        _write_html(
            output_dir / "report.html",
            clustering_md,
            "Clustering Result Report",
        )

    # Parameter-search: always HTML-only (so we can embed the SVG chart).
    chart_svg = _build_parameter_search_chart_svg(best)
    _write_html(
        output_dir / "parameter_search.html",
        parameter_search_md,
        "Parameter Search Report",
        body_prefix=chart_svg,
    )

    logger.info("Report written: %s (format=%s)", output_dir, output_format)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_classified_rows_csv(
    path: Path,
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> None:
    """Write the original DataFrame annotated with cluster_id / cluster_name.

    The `<input>_classified.csv` naming keeps multiple runs distinguishable when
    viewing several datasets side by side.
    """
    out = df.copy()
    out["cluster_id"] = labels
    if cluster_names:
        out["cluster_name"] = [cluster_names.get(int(cid), "") for cid in labels]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.debug("%s written: %d rows", path.name, len(out))


def _write_cluster_list_csv(
    path: Path,
    summaries: list[ClusterSummary],
    cluster_names: dict[int, str],
    annotations: dict[int, ClusterAnnotation] | None = None,
) -> None:
    """Write the cluster-summary CSV (one row per cluster).

    Columns:
        cluster_id, cluster_name, size, summary, rep_1, rep_2, ..., rep_N

    - `summary`: LLM-generated summary (only present when annotations exist)
    - `rep_1..N`: raw rows closest to the centroid (verification data)

    Sort order:
        1. Non-noise clusters sorted by descending size.
        2. Noise (cluster_id == -1) is appended at the end.
    """
    annotations = annotations or {}
    max_reps = max(
        (len(s.representative_texts) for s in summaries),
        default=0,
    )

    non_noise = sorted(
        [s for s in summaries if s.cluster_id != NOISE_LABEL],
        key=lambda s: s.size,
        reverse=True,
    )
    noise = [s for s in summaries if s.cluster_id == NOISE_LABEL]
    ordered = non_noise + noise

    # The summary column only appears when annotations are available.
    has_summaries = bool(annotations)
    columns = ["cluster_id", "cluster_name", "size"]
    if has_summaries:
        columns.append("summary")
    columns.extend(f"rep_{i + 1}" for i in range(max_reps))

    rows: list[dict[str, object]] = []
    for summary in ordered:
        name = cluster_names.get(summary.cluster_id, "")
        if summary.cluster_id == NOISE_LABEL and not name:
            name = "Unassigned"

        row: dict[str, object] = {
            "cluster_id": int(summary.cluster_id),
            "cluster_name": name,
            "size": int(summary.size),
        }
        if has_summaries:
            ann = annotations.get(summary.cluster_id)
            row["summary"] = ann.summary if ann else ""
        for i in range(max_reps):
            rep = (
                summary.representative_texts[i]
                if i < len(summary.representative_texts)
                else ""
            )
            row[f"rep_{i + 1}"] = rep
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.debug("%s written: %d clusters", path.name, len(df))


def _write_params_json(
    path: Path,
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
    cluster_names: dict[int, str],
) -> None:
    """Write the machine-readable params/metadata JSON."""
    payload: dict[str, Any] = {
        "input": str(input_path),
        "text_col": text_col,
        "algorithm": best.algorithm,
        "params": best.params,
        "silhouette_score": round(best.silhouette, 6),
        "quality": best.quality_flag,
        "n_clusters": best.n_clusters,
        "n_noise": best.n_noise,
        "sweep_sample_size": best.sweep_sample_size,
        "dim_before_pca": best.dim_before_pca,
        "dim_after_pca": best.dim_after_pca,
        "max_noise_ratio_filter": best.max_noise_ratio,
        "trials": [
            {
                "algorithm": t["algorithm"],
                "params": t["params"],
                "silhouette": _safe_float(t["silhouette"]),
                "n_clusters": int(t["n_clusters"]),
                "n_noise": int(t["n_noise"]),
            }
            for t in best.all_trials
        ],
    }
    if cluster_names:
        payload["cluster_names"] = {
            str(cid): name for cid, name in cluster_names.items()
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Clustering result report (Markdown body)
# ---------------------------------------------------------------------------


def _build_clustering_report_md(
    df: pd.DataFrame,
    summaries: list[ClusterSummary],
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
    cluster_names: dict[int, str],
    cluster_annotations: dict[int, ClusterAnnotation] | None = None,
) -> str:
    """Build the Markdown body for the clustering result report.

    Per-cluster layout:
        - Heading with cluster id, label, and size.
        - "Representative Text" — the LLM summary when annotations exist,
          otherwise an explanatory placeholder.
        - "Raw data near centroid (N items)" — always visible verification data.
    """
    annotations = cluster_annotations or {}
    lines: list[str] = []
    lines.append("# Clustering Result Report")
    lines.append("")

    # Loud warning for poor quality scores.
    if best.quality_flag == "poor":
        lines.append("> ⚠️ **Silhouette score is low — review recommended.**")
        lines.append(">")
        lines.append(
            "> The data may not have a clear cluster structure. See "
            "`parameter_search.html` for the full sweep details."
        )
        lines.append("")

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Input | `{input_path}` |")
    lines.append(f"| Text column | `{text_col}` |")
    lines.append(f"| Effective rows | {len(df)} |")
    lines.append(f"| Chosen algorithm | **{best.algorithm}** |")
    lines.append(f"| Parameters | `{_format_params(best.params)}` |")
    lines.append(
        f"| Silhouette score | **{best.silhouette:.4f}** "
        f"({QUALITY_LABEL[best.quality_flag]}) |"
    )
    lines.append(f"| Cluster count | {best.n_clusters} |")
    lines.append(f"| Noise count | {best.n_noise} |")
    lines.append("")
    lines.append(
        "See [`parameter_search.html`](parameter_search.html) for the full "
        "search log (all candidates with selection / rejection reasoning)."
    )
    lines.append("")

    # Per-cluster detail.
    lines.append("## Cluster Detail")
    lines.append("")

    non_noise = [s for s in summaries if s.cluster_id != NOISE_LABEL]
    non_noise.sort(key=lambda s: s.size, reverse=True)

    for summary in non_noise:
        name = cluster_names.get(summary.cluster_id)
        heading = (
            f"### Cluster #{summary.cluster_id}: {name} ({summary.size} rows)"
            if name
            else f"### Cluster #{summary.cluster_id} ({summary.size} rows)"
        )
        lines.append(heading)
        lines.append("")

        if not summary.representative_texts:
            lines.append("_No representative text available._")
            lines.append("")
            continue

        # Representative text: summary if available, otherwise explanatory placeholder.
        annotation = annotations.get(summary.cluster_id)
        lines.append("**Representative text:**")
        lines.append("")
        if annotation and annotation.summary:
            lines.append(annotation.summary)
        else:
            lines.append(
                "_(No summary generated. Re-run with `--name-clusters` to "
                "enable LLM summarisation.)_"
            )
        lines.append("")

        # The raw near-centroid items stay visible as verification data.
        lines.append(
            f"**Raw data near centroid ({len(summary.representative_texts)} items):**"
        )
        lines.append("")
        for idx, text in enumerate(summary.representative_texts, start=1):
            display = text.replace("\n", " ")
            if len(display) > 200:
                display = display[:200] + "…"
            lines.append(f"{idx}. {display}")
        lines.append("")

    # Noise.
    noise_summary = next((s for s in summaries if s.cluster_id == NOISE_LABEL), None)
    if noise_summary and noise_summary.size > 0:
        lines.append(f"### Unassigned (noise · {noise_summary.size} rows)")
        lines.append("")
        lines.append(
            "Samples that fell in low-density regions and were not assigned "
            "to any cluster."
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parameter search report (Markdown body)
# ---------------------------------------------------------------------------


def _build_parameter_search_md(
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
) -> str:
    """Build the Markdown body for the parameter search report."""
    lines: list[str] = []
    lines.append("# Parameter Search Report")
    lines.append("")

    # Conditions.
    lines.append("## Search Conditions")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Input | `{input_path}` |")
    lines.append(f"| Text column | `{text_col}` |")
    lines.append(f"| Sweep sample size | {best.sweep_sample_size} |")
    pca_summary = (
        f"PCA {best.dim_before_pca} → {best.dim_after_pca} dims"
        if best.dim_before_pca != best.dim_after_pca
        else f"no reduction ({best.dim_before_pca} dims)"
    )
    lines.append(f"| Dimensionality reduction | {pca_summary} |")
    lines.append(
        f"| Noise-ratio filter | drop candidates with noise > "
        f"{best.max_noise_ratio * 100:.0f}% |"
    )
    lines.append("")

    # Selected configuration.
    lines.append("## Selected Configuration")
    lines.append("")
    lines.append(f"- **Algorithm**: {best.algorithm}")
    lines.append(f"- **Parameters**: `{_format_params(best.params)}`")
    lines.append(
        f"- **Final silhouette score**: {best.silhouette:.4f} "
        f"({QUALITY_LABEL[best.quality_flag]})"
    )
    lines.append(f"- **Cluster count**: {best.n_clusters}")
    lines.append(f"- **Noise count**: {best.n_noise}")
    lines.append("")
    lines.append("**Why this one:**")
    lines.append("")
    lines.append("1. Highest sample silhouette among candidates passing the noise filter.")
    lines.append("2. The configuration still holds up when re-run on the full data.")
    lines.append("")

    # Classify all candidates.
    sample_size = best.sweep_sample_size or 1
    accepted: list[dict[str, Any]] = []
    rejected_noise: list[dict[str, Any]] = []
    rejected_degenerate: list[dict[str, Any]] = []
    for trial in best.all_trials:
        noise_ratio = trial["n_noise"] / sample_size
        enriched = dict(trial)
        enriched["noise_ratio"] = noise_ratio
        if trial["silhouette"] is None or trial["n_clusters"] < 2:
            rejected_degenerate.append(enriched)
        elif noise_ratio > best.max_noise_ratio:
            rejected_noise.append(enriched)
        else:
            accepted.append(enriched)

    # Accepted candidate ranking.
    accepted.sort(key=lambda t: t["silhouette"], reverse=True)
    lines.append("## Accepted Candidates (Ranked)")
    lines.append("")
    if not accepted:
        lines.append("_No candidates passed the filters._")
        lines.append("")
    else:
        lines.append(
            "| Rank | Method | Parameters | Clusters | Noise (ratio) "
            "| Silhouette | Status |"
        )
        lines.append("|---:|---|---|---:|---|---:|:---:|")
        winner_key = (best.algorithm, _freeze_params(best.params))
        for rank, trial in enumerate(accepted, start=1):
            key = (trial["algorithm"], _freeze_params(trial["params"]))
            marker = "✓ Selected" if key == winner_key else "—"
            noise_str = f"{trial['n_noise']} ({trial['noise_ratio'] * 100:.1f}%)"
            lines.append(
                f"| {rank} | {trial['algorithm']} | "
                f"`{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {noise_str} "
                f"| {trial['silhouette']:.4f} | {marker} |"
            )
        lines.append("")

    # Rejected: noise ratio over threshold.
    if rejected_noise:
        rejected_noise.sort(key=lambda t: t["silhouette"], reverse=True)
        lines.append("## Rejected — Noise-Ratio Filter")
        lines.append("")
        lines.append(
            f"Candidates with sample-noise over "
            f"{best.max_noise_ratio * 100:.0f}% are dropped even if the score "
            "looks high; a handful of tiny clusters surrounded by noise has "
            "no operational value."
        )
        lines.append("")
        lines.append(
            "| Method | Parameters | Clusters | Noise (ratio) | Silhouette |"
        )
        lines.append("|---|---|---:|---|---:|")
        for trial in rejected_noise:
            noise_str = f"{trial['n_noise']} ({trial['noise_ratio'] * 100:.1f}%)"
            lines.append(
                f"| {trial['algorithm']} | "
                f"`{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {noise_str} "
                f"| {trial['silhouette']:.4f} |"
            )
        lines.append("")

    # Rejected: degenerate (fewer than 2 clusters or no silhouette).
    if rejected_degenerate:
        lines.append("## Rejected — No Cluster Structure")
        lines.append("")
        lines.append(
            "Candidates that produced fewer than two valid clusters or "
            "could not be scored."
        )
        lines.append("")
        lines.append("| Method | Parameters | Clusters | Noise (ratio) |")
        lines.append("|---|---|---:|---|")
        for trial in rejected_degenerate:
            noise_str = f"{trial['n_noise']} ({trial['noise_ratio'] * 100:.1f}%)"
            lines.append(
                f"| {trial['algorithm']} | "
                f"`{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {noise_str} |"
            )
        lines.append("")

    # Per-method digest.
    lines.append("## Score Progression by Method")
    lines.append("")
    for algo in ("kmeans", "hdbscan", "leiden"):
        algo_trials = [t for t in best.all_trials if t["algorithm"] == algo]
        if not algo_trials:
            continue
        lines.append(f"### {algo}")
        lines.append("")
        lines.append("| Parameters | Clusters | Noise | Silhouette |")
        lines.append("|---|---:|---:|---:|")
        for trial in algo_trials:
            score = trial["silhouette"]
            score_str = f"{score:.4f}" if score is not None else "—"
            lines.append(
                f"| `{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {trial['n_noise']} | {score_str} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _write_html(
    path: Path,
    md_content: str,
    title: str,
    body_prefix: str = "",
) -> None:
    """Convert Markdown to HTML and persist the file.

    Args:
        path: target file
        md_content: Markdown body
        title: <title> tag contents (used by browsers and as OG title)
        body_prefix: HTML inserted before the converted Markdown (e.g. a chart)
    """
    import markdown as _md

    body_html = _md.markdown(
        md_content,
        extensions=["tables", "fenced_code"],
    )
    html = _wrap_html(body_html, title, body_prefix=body_prefix)
    path.write_text(html, encoding="utf-8")


def _wrap_html(body_html: str, title: str, body_prefix: str = "") -> str:
    """Wrap the body in the HTML shell with inline CSS."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"UTF-8\">\n"
        f"<title>{_escape_html(title)}</title>\n"
        f"<style>{_HTML_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body_prefix}\n"
        f"{body_html}\n"
        "<div class=\"footer\">Generated by voice-classifier</div>\n"
        "</body>\n"
        "</html>\n"
    )


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for the <title> element."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Parameter search chart (inline SVG, zero-dependency)
# ---------------------------------------------------------------------------


def _build_parameter_search_chart_svg(best: BestConfig) -> str:
    """Render the sweep result as a dual-axis SVG chart.

    - X axis: trials ranked by descending silhouette.
    - Y1 (left, bars): silhouette score.
    - Y2 (right, line): number of valid clusters.
    - Rank number is printed above each bar.
    - Trials without a computable silhouette are skipped.
    - The selected candidate's bar is highlighted.
    """
    valid = [t for t in best.all_trials if t["silhouette"] is not None]
    if not valid:
        return ""

    # Rank by descending silhouette.
    valid.sort(key=lambda t: t["silhouette"], reverse=True)
    n = len(valid)

    # Canvas dimensions.
    width = 900
    height = 420
    margin_l, margin_r, margin_t, margin_b = 70, 70, 36, 90
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    # Silhouette scale: include 0 when negatives exist.
    max_sil = max(t["silhouette"] for t in valid)
    min_sil = min(0.0, min(t["silhouette"] for t in valid))
    sil_top = max(max_sil * 1.1, 0.05)
    sil_bottom = min_sil

    # Cluster-count scale.
    max_clusters = max(t["n_clusters"] for t in valid)
    cluster_top = max(max_clusters * 1.1, 1)

    # Bar layout.
    bar_slot = chart_w / n
    bar_w = bar_slot * 0.7

    # Identify the selected candidate for highlighting.
    winner_key = (best.algorithm, _freeze_params(best.params))

    # Colours.
    c_bar = "#4e79a7"
    c_bar_winner = "#f28e2c"
    c_line = "#e15759"
    c_grid = "#d0d7de"
    c_text = "#1f2328"
    c_muted = "#57606a"

    svg: list[str] = []
    svg.append(
        f'<svg width="100%" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:{width}px;display:block;margin:0 auto;'
        f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
    )

    def y_sil(score: float) -> float:
        """Map silhouette to a y-coordinate (screen-down)."""
        if sil_top == sil_bottom:
            return margin_t + chart_h
        ratio = (score - sil_bottom) / (sil_top - sil_bottom)
        return margin_t + chart_h * (1 - ratio)

    def y_clusters(count: float) -> float:
        """Map cluster count to a y-coordinate."""
        if cluster_top == 0:
            return margin_t + chart_h
        ratio = count / cluster_top
        return margin_t + chart_h * (1 - ratio)

    # Grid lines at 4 intervals based on the left axis.
    for i in range(5):
        frac = i / 4
        y = margin_t + chart_h * (1 - frac)
        sil_val = sil_bottom + (sil_top - sil_bottom) * frac
        clusters_val = cluster_top * frac
        # Horizontal grid line.
        svg.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" '
            f'y2="{y:.1f}" stroke="{c_grid}" stroke-width="1" '
            f'stroke-dasharray="2,2" />'
        )
        # Left axis label (silhouette).
        svg.append(
            f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{c_muted}">{sil_val:.2f}</text>'
        )
        # Right axis label (cluster count).
        svg.append(
            f'<text x="{width - margin_r + 8}" y="{y + 4:.1f}" '
            f'text-anchor="start" font-size="11" fill="{c_muted}">'
            f'{int(round(clusters_val))}</text>'
        )

    # Emphasised zero line when negatives exist.
    if sil_bottom < 0:
        zero_y = y_sil(0.0)
        svg.append(
            f'<line x1="{margin_l}" y1="{zero_y:.1f}" x2="{width - margin_r}" '
            f'y2="{zero_y:.1f}" stroke="{c_muted}" stroke-width="1" />'
        )

    # Axes.
    svg.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
        f'y2="{margin_t + chart_h}" stroke="{c_text}" stroke-width="1.5" />'
    )
    svg.append(
        f'<line x1="{width - margin_r}" y1="{margin_t}" '
        f'x2="{width - margin_r}" y2="{margin_t + chart_h}" '
        f'stroke="{c_text}" stroke-width="1.5" />'
    )
    svg.append(
        f'<line x1="{margin_l}" y1="{margin_t + chart_h}" '
        f'x2="{width - margin_r}" y2="{margin_t + chart_h}" '
        f'stroke="{c_text}" stroke-width="1.5" />'
    )

    # Axis titles.
    svg.append(
        f'<text x="{margin_l - 50}" y="{margin_t + chart_h / 2:.1f}" '
        f'font-size="12" fill="{c_bar}" '
        f'transform="rotate(-90 {margin_l - 50} {margin_t + chart_h / 2:.1f})" '
        f'text-anchor="middle">Silhouette score</text>'
    )
    svg.append(
        f'<text x="{width - margin_r + 50}" y="{margin_t + chart_h / 2:.1f}" '
        f'font-size="12" fill="{c_line}" '
        f'transform="rotate(90 {width - margin_r + 50} '
        f'{margin_t + chart_h / 2:.1f})" text-anchor="middle">Cluster count</text>'
    )

    # Bars.
    zero_y = y_sil(0.0)
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        x_left = x_center - bar_w / 2
        y_top = y_sil(trial["silhouette"])
        bar_y = min(y_top, zero_y)
        bar_h = abs(y_top - zero_y)

        is_winner = (
            trial["algorithm"],
            _freeze_params(trial["params"]),
        ) == winner_key
        fill = c_bar_winner if is_winner else c_bar

        svg.append(
            f'<rect x="{x_left:.1f}" y="{bar_y:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{fill}">'
            f'<title>rank {rank}: {trial["algorithm"]} '
            f'{_format_params(trial["params"])} '
            f'(score={trial["silhouette"]:.4f}, '
            f'clusters={trial["n_clusters"]}, '
            f'noise={trial["n_noise"]})</title>'
            f'</rect>'
        )
        # Rank label above the bar.
        label_y = bar_y - 4 if bar_h > 0 else zero_y - 4
        svg.append(
            f'<text x="{x_center:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle" font-size="10" fill="{c_text}">{rank}</text>'
        )

    # Line series for the cluster count.
    points = []
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        y = y_clusters(trial["n_clusters"])
        points.append(f"{x_center:.1f},{y:.1f}")
    path_d = "M " + " L ".join(points)
    svg.append(
        f'<path d="{path_d}" fill="none" stroke="{c_line}" stroke-width="2" />'
    )
    # Markers on each data point.
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        y = y_clusters(trial["n_clusters"])
        svg.append(
            f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="3" '
            f'fill="{c_line}" />'
        )

    # X-axis labels rotated 45° so they don't overlap.
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        algo_short = {"kmeans": "KM", "hdbscan": "HD", "leiden": "LD"}.get(
            trial["algorithm"], trial["algorithm"][:2]
        )
        param_label = _compact_param_label(trial["params"])
        svg.append(
            f'<text x="{x_center:.1f}" y="{margin_t + chart_h + 12:.1f}" '
            f'text-anchor="end" font-size="9" fill="{c_muted}" '
            f'transform="rotate(-45 {x_center:.1f} '
            f'{margin_t + chart_h + 12:.1f})">{algo_short}:{param_label}</text>'
        )

    # Legend.
    legend_y = height - 20
    legend_items = [
        (margin_l + 10, c_bar, "Silhouette (bars, left axis)"),
        (margin_l + 220, c_bar_winner, "Selected"),
        (margin_l + 310, c_line, "Cluster count (line, right axis)"),
    ]
    for x, color, label in legend_items:
        svg.append(
            f'<rect x="{x}" y="{legend_y}" width="14" height="10" fill="{color}" />'
        )
        svg.append(
            f'<text x="{x + 20}" y="{legend_y + 9}" font-size="11" '
            f'fill="{c_text}">{label}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _compact_param_label(params: dict[str, Any]) -> str:
    """Short parameter label for axis tick text."""
    if "k" in params:
        return f"k={params['k']}"
    if "min_cluster_size" in params:
        return f"mcs={params['min_cluster_size']}"
    if "resolution" in params:
        return f"res={params['resolution']:.1f}"
    first_key = next(iter(params))
    return f"{first_key}={params[first_key]}"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_params(params: dict[str, Any]) -> str:
    """Format params as a comma-separated `key=value` string with bounded precision."""
    parts: list[str] = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _freeze_params(params: dict[str, Any]) -> tuple:
    """Convert a dict into a hashable tuple (for winner identity comparison)."""
    return tuple(sorted(params.items()))


def _safe_float(value: Any) -> float | None:
    """Coerce to float, returning None on NaN/invalid values (safe for JSON)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f
