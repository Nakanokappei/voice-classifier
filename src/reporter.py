"""レポート出力 — Markdown / CSV / JSON を生成する.

責務:
    - 採用クラスタ設定・代表テキストを Markdown で要約
    - 元データに `cluster_id` を付与して CSV 保存
    - 採用パラメータ・スコアを JSON 保存
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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


def write_report(
    output_dir: Path | str,
    df: pd.DataFrame,
    labels: np.ndarray,
    summaries: list[ClusterSummary],
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
) -> None:
    """3種類の成果物を `output_dir` に書き出す.

    Args:
        output_dir: 出力先（存在しなければ作成）
        df: `loader.load_csv` の戻り値（`_normalized_text`, `_duplicate_count` を持つ）
        labels: サンプルごとのクラスタID
        summaries: `clusterer.summarize_clusters` の戻り値
        best: `tuner.find_best_clustering` の戻り値
        input_path: 入力CSVパス（レポート記載用）
        text_col: 対象テキスト列名（レポート記載用）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_clusters_csv(output_dir / "clusters.csv", df, labels)
    _write_params_json(output_dir / "params.json", best, input_path, text_col)
    _write_markdown_report(
        output_dir / "report.md",
        df=df,
        summaries=summaries,
        best=best,
        input_path=input_path,
        text_col=text_col,
    )

    logger.info("レポート出力完了: %s", output_dir)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_clusters_csv(path: Path, df: pd.DataFrame, labels: np.ndarray) -> None:
    """入力DataFrameに cluster_id を付けて保存."""
    out = df.copy()
    out["cluster_id"] = labels
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.debug("clusters.csv 保存: %d 行", len(out))


def _write_params_json(
    path: Path,
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_markdown_report(
    path: Path,
    df: pd.DataFrame,
    summaries: list[ClusterSummary],
    best: BestConfig,
    input_path: Path | str,
    text_col: str,
) -> None:
    """人間可読な Markdown レポートを書く."""
    lines: list[str] = []
    lines.append("# voice-classifier レポート")
    lines.append("")

    # 品質警告（poor の場合は冒頭に目立つ形で）
    if best.quality_flag == "poor":
        lines.append("> ⚠️ **シルエットスコアが低めです（再検討推奨）**")
        lines.append(">")
        lines.append("> データの性質上、明瞭なクラスタ構造が得られていない可能性があります.")
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
    lines.append(f"| シルエットスコア | **{best.silhouette:.4f}** ({QUALITY_LABEL[best.quality_flag]}) |")
    lines.append(f"| クラスタ数 | {best.n_clusters} |")
    lines.append(f"| ノイズ件数 | {best.n_noise} |")
    lines.append("")

    # クラスタ別詳細
    lines.append("## クラスタ詳細")
    lines.append("")

    non_noise = [s for s in summaries if s.cluster_id != NOISE_LABEL]
    # サイズ降順で表示
    non_noise.sort(key=lambda s: s.size, reverse=True)

    for summary in non_noise:
        lines.append(f"### クラスタ #{summary.cluster_id}（{summary.size}件）")
        lines.append("")
        if not summary.representative_texts:
            lines.append("_代表テキストなし_")
            lines.append("")
            continue
        lines.append("**代表テキスト（重心に近い順）:**")
        lines.append("")
        for idx, text in enumerate(summary.representative_texts, start=1):
            # Markdown 表示のため改行を除去し、長文は省略
            display = text.replace("\n", " ")
            if len(display) > 200:
                display = display[:200] + "…"
            lines.append(f"{idx}. {display}")
        lines.append("")

    # ノイズ
    noise_summary = next(
        (s for s in summaries if s.cluster_id == NOISE_LABEL), None
    )
    if noise_summary and noise_summary.size > 0:
        lines.append(f"### 未分類（ノイズ・{noise_summary.size}件）")
        lines.append("")
        lines.append("密度の低い領域に位置し、いずれのクラスタにも属さなかったサンプルです.")
        lines.append("")

    # 試行履歴（折りたたみ）
    lines.append("## 試行履歴")
    lines.append("")
    lines.append("<details><summary>評価した全候補（クリックで展開）</summary>")
    lines.append("")
    lines.append("| アルゴリズム | パラメータ | クラスタ数 | ノイズ | シルエット |")
    lines.append("|---|---|---:|---:|---:|")
    for trial in best.all_trials:
        score = trial["silhouette"]
        score_str = f"{score:.4f}" if score is not None else "—"
        lines.append(
            f"| {trial['algorithm']} | `{_format_params(trial['params'])}` "
            f"| {trial['n_clusters']} | {trial['n_noise']} | {score_str} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


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
