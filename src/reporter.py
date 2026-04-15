"""レポート出力 — 4つの成果物を生成する.

生成ファイル:
    - report.md              クラスタリング結果レポート（採用結果・クラスタ詳細）
    - parameter_search.md    パラメータ探索レポート（全試行・採用理由・除外理由）
    - clusters.csv           全レコードに cluster_id を付与したCSV
    - params.json            採用パラメータ・スコアの機械可読版
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .clusterer import ClusterSummary, NOISE_LABEL
from .tuner import BestConfig

logger = logging.getLogger(__name__)

QUALITY_LABEL: dict[str, str] = {
    "good": "良好",
    "warn": "許容（注意）",
    "poor": "要再検討",
}

OutputFormat = Literal["md", "html", "both"]

# HTML レポート用の埋め込み CSS. テーブル可読性と PII 配慮の配色
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
    output_format: OutputFormat = "md",
) -> None:
    """成果物を `output_dir` に書き出す.

    Args:
        output_dir: 出力先（存在しなければ作成）
        df: `loader.load_csv` の戻り値
        labels: サンプルごとのクラスタID
        summaries: `clusterer.summarize_clusters` の戻り値
        best: `tuner.find_best_clustering` の戻り値
        input_path: 入力CSVパス（レポート記載用）
        text_col: 対象テキスト列名（レポート記載用）
        cluster_names: LLM 等で生成したクラスタID→ラベルのマップ（任意）
        output_format: "md" / "html" / "both" — レポート形式
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = cluster_names or {}

    # 常に出力（機械可読）
    _write_clusters_csv(output_dir / "clusters.csv", df, labels, names)
    _write_params_json(output_dir / "params.json", best, input_path, text_col, names)

    # Markdown 文字列を生成（両形式で共通利用）
    clustering_md = _build_clustering_report_md(
        df=df, summaries=summaries, best=best,
        input_path=input_path, text_col=text_col, cluster_names=names,
    )
    parameter_search_md = _build_parameter_search_md(
        best=best, input_path=input_path, text_col=text_col,
    )

    want_md = output_format in ("md", "both")
    want_html = output_format in ("html", "both")

    if want_md:
        (output_dir / "report.md").write_text(clustering_md, encoding="utf-8")
        (output_dir / "parameter_search.md").write_text(
            parameter_search_md, encoding="utf-8"
        )
    if want_html:
        _write_html(output_dir / "report.html", clustering_md, "クラスタリング結果レポート")
        _write_html(
            output_dir / "parameter_search.html",
            parameter_search_md,
            "パラメータ探索レポート",
        )

    logger.info("レポート出力完了: %s (format=%s)", output_dir, output_format)


# ---------------------------------------------------------------------------
# clusters.csv / params.json
# ---------------------------------------------------------------------------


def _write_clusters_csv(
    path: Path,
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> None:
    """入力DataFrameに cluster_id / cluster_name を付けて保存."""
    out = df.copy()
    out["cluster_id"] = labels
    if cluster_names:
        out["cluster_name"] = [cluster_names.get(int(cid), "") for cid in labels]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.debug("clusters.csv 保存: %d 行", len(out))


def _write_params_json(
    path: Path,
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
    cluster_names: dict[int, str],
) -> None:
    """採用パラメータ・スコアをJSON化."""
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
# report.md — 採用結果のみを読みやすく
# ---------------------------------------------------------------------------


def _build_clustering_report_md(
    df: pd.DataFrame,
    summaries: list[ClusterSummary],
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
    cluster_names: dict[int, str],
) -> str:
    """クラスタリング結果レポートの Markdown 文字列を生成."""
    lines: list[str] = []
    lines.append("# クラスタリング結果レポート")
    lines.append("")

    # 品質警告（poor の場合は冒頭に目立つ形で）
    if best.quality_flag == "poor":
        lines.append("> ⚠️ **シルエットスコアが低めです（再検討推奨）**")
        lines.append(">")
        lines.append("> データの性質上、明瞭なクラスタ構造が得られていない可能性があります.")
        lines.append("> パラメータ探索の詳細は `parameter_search.md` を参照してください.")
        lines.append("")

    # サマリテーブル
    lines.append("## 概要")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|---|---|")
    lines.append(f"| 入力 | `{input_path}` |")
    lines.append(f"| テキスト列 | `{text_col}` |")
    lines.append(f"| 有効サンプル数 | {len(df)} |")
    lines.append(f"| 採用アルゴリズム | **{best.algorithm}** |")
    lines.append(f"| パラメータ | `{_format_params(best.params)}` |")
    lines.append(
        f"| シルエットスコア | **{best.silhouette:.4f}** ({QUALITY_LABEL[best.quality_flag]}) |"
    )
    lines.append(f"| クラスタ数 | {best.n_clusters} |")
    lines.append(f"| ノイズ件数 | {best.n_noise} |")
    lines.append("")
    lines.append(
        "探索過程の詳細（全候補・採用/除外理由）は "
        "[`parameter_search.md`](parameter_search.md) を参照."
    )
    lines.append("")

    # クラスタ別詳細
    lines.append("## クラスタ詳細")
    lines.append("")

    non_noise = [s for s in summaries if s.cluster_id != NOISE_LABEL]
    non_noise.sort(key=lambda s: s.size, reverse=True)

    for summary in non_noise:
        name = cluster_names.get(summary.cluster_id)
        heading = (
            f"### クラスタ #{summary.cluster_id}: {name}（{summary.size}件）"
            if name
            else f"### クラスタ #{summary.cluster_id}（{summary.size}件）"
        )
        lines.append(heading)
        lines.append("")
        if not summary.representative_texts:
            lines.append("_代表テキストなし_")
            lines.append("")
            continue
        lines.append("**代表テキスト（重心に近い順）:**")
        lines.append("")
        for idx, text in enumerate(summary.representative_texts, start=1):
            display = text.replace("\n", " ")
            if len(display) > 200:
                display = display[:200] + "…"
            lines.append(f"{idx}. {display}")
        lines.append("")

    # ノイズ
    noise_summary = next((s for s in summaries if s.cluster_id == NOISE_LABEL), None)
    if noise_summary and noise_summary.size > 0:
        lines.append(f"### 未分類（ノイズ・{noise_summary.size}件）")
        lines.append("")
        lines.append("密度の低い領域に位置し、いずれのクラスタにも属さなかったサンプルです.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parameter_search.md — 探索全貌と採用根拠
# ---------------------------------------------------------------------------


def _build_parameter_search_md(
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
) -> str:
    """パラメータ探索レポートの Markdown 文字列を生成."""
    lines: list[str] = []
    lines.append("# パラメータ探索レポート")
    lines.append("")

    # 探索条件
    lines.append("## 探索条件")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|---|---|")
    lines.append(f"| 入力 | `{input_path}` |")
    lines.append(f"| テキスト列 | `{text_col}` |")
    lines.append(f"| スイープサンプル数 | {best.sweep_sample_size} |")
    pca_summary = (
        f"PCA {best.dim_before_pca} → {best.dim_after_pca} 次元"
        if best.dim_before_pca != best.dim_after_pca
        else f"削減なし（{best.dim_before_pca} 次元）"
    )
    lines.append(f"| 次元削減 | {pca_summary} |")
    lines.append(f"| ノイズ率フィルタ | 候補のノイズ率 > {best.max_noise_ratio * 100:.0f}% は除外 |")
    lines.append("")

    # 採用結果と理由
    lines.append("## 採用結果")
    lines.append("")
    lines.append(f"- **アルゴリズム**: {best.algorithm}")
    lines.append(f"- **パラメータ**: `{_format_params(best.params)}`")
    lines.append(f"- **最終シルエットスコア**: {best.silhouette:.4f} ({QUALITY_LABEL[best.quality_flag]})")
    lines.append(f"- **クラスタ数**: {best.n_clusters}")
    lines.append(f"- **ノイズ件数**: {best.n_noise}")
    lines.append("")
    lines.append("**採用理由**:")
    lines.append("")
    lines.append("1. スイープで得たシルエットスコアが、ノイズ率フィルタを通過した候補中で最大")
    lines.append("2. フルデータでの再適用後も意味のあるクラスタ構造を保持")
    lines.append("")

    # 全候補を採用フィルタ適用結果別に分類
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

    # 採用候補ランキング
    accepted.sort(key=lambda t: t["silhouette"], reverse=True)
    lines.append("## 採用候補ランキング")
    lines.append("")
    if not accepted:
        lines.append("_フィルタ通過候補なし_")
        lines.append("")
    else:
        lines.append("| 順位 | 手法 | パラメータ | クラスタ | ノイズ (率) | シルエット | 判定 |")
        lines.append("|---:|---|---|---:|---|---:|:---:|")
        winner_key = (best.algorithm, _freeze_params(best.params))
        for rank, trial in enumerate(accepted, start=1):
            key = (trial["algorithm"], _freeze_params(trial["params"]))
            marker = "✓ 採用" if key == winner_key else "—"
            noise_str = f"{trial['n_noise']} ({trial['noise_ratio'] * 100:.1f}%)"
            lines.append(
                f"| {rank} | {trial['algorithm']} | `{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {noise_str} "
                f"| {trial['silhouette']:.4f} | {marker} |"
            )
        lines.append("")

    # フィルタ除外（ノイズ率超過）
    if rejected_noise:
        rejected_noise.sort(key=lambda t: t["silhouette"], reverse=True)
        lines.append("## 除外: ノイズ率フィルタ")
        lines.append("")
        lines.append(
            f"サンプル中のノイズ割合が {best.max_noise_ratio * 100:.0f}% を超えたため、"
            "高スコアでも実用性なしと判定."
        )
        lines.append("")
        lines.append("| 手法 | パラメータ | クラスタ | ノイズ (率) | シルエット |")
        lines.append("|---|---|---:|---|---:|")
        for trial in rejected_noise:
            noise_str = f"{trial['n_noise']} ({trial['noise_ratio'] * 100:.1f}%)"
            lines.append(
                f"| {trial['algorithm']} | `{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {noise_str} | {trial['silhouette']:.4f} |"
            )
        lines.append("")

    # 除外（退化: クラスタ不足やスコア計算不可）
    if rejected_degenerate:
        lines.append("## 除外: クラスタ構造が成立せず")
        lines.append("")
        lines.append(
            "有効クラスタが 2 未満、またはシルエット計算が成立しなかった候補."
        )
        lines.append("")
        lines.append("| 手法 | パラメータ | クラスタ | ノイズ (率) |")
        lines.append("|---|---|---:|---|")
        for trial in rejected_degenerate:
            noise_str = f"{trial['n_noise']} ({trial['noise_ratio'] * 100:.1f}%)"
            lines.append(
                f"| {trial['algorithm']} | `{_format_params(trial['params'])}` "
                f"| {trial['n_clusters']} | {noise_str} |"
            )
        lines.append("")

    # 手法別ダイジェスト
    lines.append("## 手法別のスコア推移")
    lines.append("")
    for algo in ("kmeans", "dbscan", "hdbscan"):
        algo_trials = [t for t in best.all_trials if t["algorithm"] == algo]
        if not algo_trials:
            continue
        lines.append(f"### {algo}")
        lines.append("")
        lines.append("| パラメータ | クラスタ | ノイズ | シルエット |")
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
# HTML 生成
# ---------------------------------------------------------------------------


def _write_html(path: Path, md_content: str, title: str) -> None:
    """Markdown を HTML に変換してファイル保存."""
    import markdown as _md

    body_html = _md.markdown(
        md_content,
        extensions=["tables", "fenced_code"],
    )
    html = _wrap_html(body_html, title)
    path.write_text(html, encoding="utf-8")


def _wrap_html(body_html: str, title: str) -> str:
    """`<html>` シェルで包む. CSS はインラインで埋め込む（配布容易性重視）."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"ja\">\n"
        "<head>\n"
        "<meta charset=\"UTF-8\">\n"
        f"<title>{_escape_html(title)}</title>\n"
        f"<style>{_HTML_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "<div class=\"footer\">Generated by voice-classifier</div>\n"
        "</body>\n"
        "</html>\n"
    )


def _escape_html(text: str) -> str:
    """最小限の HTML エスケープ（title 要素用）."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_params(params: dict[str, Any]) -> str:
    """`key=value` をカンマ区切りで. 数値は有効桁を抑える."""
    parts: list[str] = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _freeze_params(params: dict[str, Any]) -> tuple:
    """dict を比較可能な tuple に変換（採用候補特定用）."""
    return tuple(sorted(params.items()))


def _safe_float(value: Any) -> float | None:
    """NaN/None を JSON 安全な None に変換."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f
