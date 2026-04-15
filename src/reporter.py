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
    # 元ファイル名をプレフィックスにして <元ファイル名>_classified.csv を生成
    input_stem = Path(str(input_path)).stem
    _write_classified_rows_csv(
        output_dir / f"{input_stem}_classified.csv", df, labels, names
    )
    _write_cluster_list_csv(output_dir / "clusters.csv", summaries, names)
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

    # クラスタリング結果レポート: format 設定に従って md / html を生成
    if want_md:
        (output_dir / "report.md").write_text(clustering_md, encoding="utf-8")
    if want_html:
        _write_html(
            output_dir / "report.html",
            clustering_md,
            "クラスタリング結果レポート",
        )

    # パラメータ探索レポート: HTML のみ常時生成（グラフを埋め込むため）
    # md は廃止（テキストで試行詳細を追うよりも視覚的な比較が有用）
    chart_svg = _build_parameter_search_chart_svg(best)
    _write_html(
        output_dir / "parameter_search.html",
        parameter_search_md,
        "パラメータ探索レポート",
        body_prefix=chart_svg,
    )

    logger.info("レポート出力完了: %s (format=%s)", output_dir, output_format)


# ---------------------------------------------------------------------------
# clusters.csv / params.json
# ---------------------------------------------------------------------------


def _write_classified_rows_csv(
    path: Path,
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> None:
    """入力DataFrameに cluster_id / cluster_name を付けて保存.

    元ファイル名を手がかりにして `<元ファイル名>_classified.csv` として保存することで、
    入力ごとに判別しやすくする（`clusters.csv` と混同しにくい命名）.
    """
    out = df.copy()
    out["cluster_id"] = labels
    if cluster_names:
        out["cluster_name"] = [cluster_names.get(int(cid), "") for cid in labels]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.debug("%s 保存: %d 行", path.name, len(out))


def _write_cluster_list_csv(
    path: Path,
    summaries: list[ClusterSummary],
    cluster_names: dict[int, str],
) -> None:
    """クラスタ1件=1行のサマリ CSV を保存.

    列:
        cluster_id, cluster_name, size, rep_1, rep_2, ..., rep_N

    ソート規則:
        1. ノイズ（cluster_id == -1）は末尾
        2. それ以外はサイズ降順
    """
    # 代表テキスト列の幅を揃えるため、最大件数を求める
    max_reps = max(
        (len(s.representative_texts) for s in summaries),
        default=0,
    )

    # 並び替え: ノイズは末尾、それ以外はサイズ降順
    non_noise = sorted(
        [s for s in summaries if s.cluster_id != NOISE_LABEL],
        key=lambda s: s.size,
        reverse=True,
    )
    noise = [s for s in summaries if s.cluster_id == NOISE_LABEL]
    ordered = non_noise + noise

    columns = ["cluster_id", "cluster_name", "size"] + [
        f"rep_{i + 1}" for i in range(max_reps)
    ]

    rows: list[dict[str, object]] = []
    for summary in ordered:
        # ノイズクラスタは代表テキストなし、ラベルは常に「未分類」相当
        name = cluster_names.get(summary.cluster_id, "")
        if summary.cluster_id == NOISE_LABEL and not name:
            name = "未分類"

        row: dict[str, object] = {
            "cluster_id": int(summary.cluster_id),
            "cluster_name": name,
            "size": int(summary.size),
        }
        # 代表テキストを rep_1..rep_N に割り当て（足りない列は空文字）
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
    logger.debug("%s 保存: %d クラスタ", path.name, len(df))


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


def _write_html(
    path: Path,
    md_content: str,
    title: str,
    body_prefix: str = "",
) -> None:
    """Markdown を HTML に変換してファイル保存.

    Args:
        path: 出力ファイル
        md_content: Markdown 本文
        title: `<title>` 要素と既定の見出し
        body_prefix: Markdown 変換結果の手前に挿入する追加 HTML（グラフ等）
    """
    import markdown as _md

    body_html = _md.markdown(
        md_content,
        extensions=["tables", "fenced_code"],
    )
    html = _wrap_html(body_html, title, body_prefix=body_prefix)
    path.write_text(html, encoding="utf-8")


def _wrap_html(body_html: str, title: str, body_prefix: str = "") -> str:
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
        f"{body_prefix}\n"
        f"{body_html}\n"
        "<div class=\"footer\">Generated by voice-classifier</div>\n"
        "</body>\n"
        "</html>\n"
    )


# ---------------------------------------------------------------------------
# Parameter search chart (inline SVG, zero-dependency)
# ---------------------------------------------------------------------------


def _build_parameter_search_chart_svg(best: BestConfig) -> str:
    """パラメータ探索結果を SVG の 2軸グラフで描画.

    - X軸: 試行（シルエットスコア降順でランク順に並べる）
    - Y1軸（左・棒）: シルエットスコア
    - Y2軸（右・折れ線）: 有効クラスタ数
    - 棒グラフの上には順位番号（1, 2, ...）を表示
    - シルエットが算出できなかった試行は除外
    - 採用候補は棒を強調色で描画
    """
    valid = [t for t in best.all_trials if t["silhouette"] is not None]
    if not valid:
        return ""

    # スコア降順で順位付け
    valid.sort(key=lambda t: t["silhouette"], reverse=True)
    n = len(valid)

    # キャンバス寸法
    width = 900
    height = 420
    margin_l, margin_r, margin_t, margin_b = 70, 70, 36, 90
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    # スコアスケール: 負の値も扱えるよう 0 を原点に含める
    max_sil = max(t["silhouette"] for t in valid)
    min_sil = min(0.0, min(t["silhouette"] for t in valid))
    # 表示余白を上 10% 足す
    sil_top = max(max_sil * 1.1, 0.05)
    sil_bottom = min_sil

    # クラスタ数スケール
    max_clusters = max(t["n_clusters"] for t in valid)
    cluster_top = max(max_clusters * 1.1, 1)

    # 棒グラフ配置
    bar_slot = chart_w / n
    bar_w = bar_slot * 0.7

    # 採用候補の特定
    winner_key = (best.algorithm, _freeze_params(best.params))

    # 色
    c_bar = "#4e79a7"
    c_bar_winner = "#f28e2c"
    c_line = "#e15759"
    c_grid = "#d0d7de"
    c_text = "#1f2328"
    c_muted = "#57606a"

    # --- SVG ---
    svg: list[str] = []
    svg.append(
        f'<svg width="100%" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:{width}px;display:block;margin:0 auto;'
        f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
    )

    # 軸の原点をチャート内で計算
    def y_sil(score: float) -> float:
        """シルエットスコアを y 座標（上から）に変換."""
        if sil_top == sil_bottom:
            return margin_t + chart_h
        ratio = (score - sil_bottom) / (sil_top - sil_bottom)
        return margin_t + chart_h * (1 - ratio)

    def y_clusters(count: float) -> float:
        """クラスタ数を y 座標に変換."""
        if cluster_top == 0:
            return margin_t + chart_h
        ratio = count / cluster_top
        return margin_t + chart_h * (1 - ratio)

    # グリッド線（左軸基準で 4 段）
    for i in range(5):
        frac = i / 4
        y = margin_t + chart_h * (1 - frac)
        sil_val = sil_bottom + (sil_top - sil_bottom) * frac
        clusters_val = cluster_top * frac
        # 横グリッド
        svg.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" '
            f'y2="{y:.1f}" stroke="{c_grid}" stroke-width="1" '
            f'stroke-dasharray="2,2" />'
        )
        # 左軸ラベル（シルエット）
        svg.append(
            f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{c_muted}">{sil_val:.2f}</text>'
        )
        # 右軸ラベル（クラスタ数）
        svg.append(
            f'<text x="{width - margin_r + 8}" y="{y + 4:.1f}" '
            f'text-anchor="start" font-size="11" fill="{c_muted}">'
            f'{int(round(clusters_val))}</text>'
        )

    # ゼロ基準線（スコアが負に入る場合のみ強調）
    if sil_bottom < 0:
        zero_y = y_sil(0.0)
        svg.append(
            f'<line x1="{margin_l}" y1="{zero_y:.1f}" x2="{width - margin_r}" '
            f'y2="{zero_y:.1f}" stroke="{c_muted}" stroke-width="1" />'
        )

    # 軸線
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

    # 軸タイトル
    svg.append(
        f'<text x="{margin_l - 50}" y="{margin_t + chart_h / 2:.1f}" '
        f'font-size="12" fill="{c_bar}" '
        f'transform="rotate(-90 {margin_l - 50} {margin_t + chart_h / 2:.1f})" '
        f'text-anchor="middle">シルエットスコア</text>'
    )
    svg.append(
        f'<text x="{width - margin_r + 50}" y="{margin_t + chart_h / 2:.1f}" '
        f'font-size="12" fill="{c_line}" '
        f'transform="rotate(90 {width - margin_r + 50} '
        f'{margin_t + chart_h / 2:.1f})" text-anchor="middle">クラスタ数</text>'
    )

    # 棒グラフ
    zero_y = y_sil(0.0)
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        x_left = x_center - bar_w / 2
        y_top = y_sil(trial["silhouette"])
        # スコアが正か負かで向きを決める
        bar_y = min(y_top, zero_y)
        bar_h = abs(y_top - zero_y)

        # 採用候補は強調色
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
        # 順位番号ラベル（棒の上）
        label_y = bar_y - 4 if bar_h > 0 else zero_y - 4
        svg.append(
            f'<text x="{x_center:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle" font-size="10" fill="{c_text}">{rank}</text>'
        )

    # 折れ線（クラスタ数）
    points = []
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        y = y_clusters(trial["n_clusters"])
        points.append(f"{x_center:.1f},{y:.1f}")
    path_d = "M " + " L ".join(points)
    svg.append(
        f'<path d="{path_d}" fill="none" stroke="{c_line}" stroke-width="2" />'
    )
    # 折れ線の各ポイントにマーカー
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        y = y_clusters(trial["n_clusters"])
        svg.append(
            f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="3" '
            f'fill="{c_line}" />'
        )

    # X 軸ラベル: 手法名を90度回転で縦書き（候補が多いと重なるため）
    for rank, trial in enumerate(valid, start=1):
        x_center = margin_l + bar_slot * (rank - 0.5)
        algo_short = {"kmeans": "KM", "dbscan": "DB", "hdbscan": "HD"}.get(
            trial["algorithm"], trial["algorithm"][:2]
        )
        param_label = _compact_param_label(trial["params"])
        svg.append(
            f'<text x="{x_center:.1f}" y="{margin_t + chart_h + 12:.1f}" '
            f'text-anchor="end" font-size="9" fill="{c_muted}" '
            f'transform="rotate(-45 {x_center:.1f} '
            f'{margin_t + chart_h + 12:.1f})">{algo_short}:{param_label}</text>'
        )

    # 凡例
    legend_y = height - 20
    legend_items = [
        (margin_l + 10, c_bar, "シルエットスコア（棒・左軸）"),
        (margin_l + 220, c_bar_winner, "採用候補"),
        (margin_l + 310, c_line, "クラスタ数（折れ線・右軸）"),
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
    """軸ラベル用の短いパラメータ表示. 主要キーのみ."""
    if "k" in params:
        return f"k={params['k']}"
    if "eps" in params:
        return f"eps={params['eps']:.2f}"
    if "min_cluster_size" in params:
        return f"mcs={params['min_cluster_size']}"
    # 最初のキーだけ
    first_key = next(iter(params))
    return f"{first_key}={params[first_key]}"


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
