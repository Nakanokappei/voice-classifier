"""CLI エントリポイント — loader → embedder → tuner → clusterer → reporter.

使い方::

    python src/pipeline.py --input data/input/sample.csv --text-col "対応内容"
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _suppress_known_benign_warnings() -> None:
    """sklearn 内部で発生する既知良性 RuntimeWarning を抑制する.

    - sklearn.utils.extmath (safe_sparse_dot) の divide/overflow/invalid
      は randomized SVD や cosine_distances 正規化で発生する数値ノイズで、
      voice-classifier の演算結果には影響しない
    - ゼロノルム行は上流で e_0 に置換済みなので意味的な警告は残らない
    """
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*(divide by zero|overflow|invalid value).*matmul.*",
    )
    # PCA の randomized SVD 内部の類似警告も同メッセージなので上記で拾える


_suppress_known_benign_warnings()

# モジュールとして（`python -m src.pipeline`）、またはスクリプトとして（`python src/pipeline.py`）
# どちらでも動くように import 経路を両対応
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import clusterer, embedder, loader, namer, progress, reporter, tuner
else:
    from . import clusterer, embedder, loader, namer, progress, reporter, tuner


logger = logging.getLogger("voice_classifier")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を解析."""
    parser = argparse.ArgumentParser(
        description="顧客の声 自動分類・洞察レポートシステム",
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="入力CSVパス"
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="分類対象テキストの列名. 省略時は対話的に候補から選択",
    )
    parser.add_argument(
        "--text-cols",
        default=None,
        help=(
            "複数列を結合して埋め込み対象にする場合のカンマ区切り列名. "
            "例: --text-cols 'Ticket Subject,Ticket Description'. "
            "--text-col と排他."
        ),
    )
    parser.add_argument(
        "--column-labels",
        default=None,
        help=(
            "複数列モード時の列名→ラベル変換. 例: "
            '--column-labels \'Ticket Subject=subject,Ticket Description=body\''
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="出力ディレクトリルート（ここにタイムスタンプ付きで作成）",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="埋め込みキャッシュ保存先",
    )
    parser.add_argument(
        "--model",
        default=embedder.DEFAULT_MODEL,
        help="OpenAI 埋め込みモデル名",
    )
    parser.add_argument("--top-k", type=int, default=5, help="代表テキスト抽出件数")
    parser.add_argument("--min-clusters", type=int, default=2, help="KMeans 下限K")
    parser.add_argument("--max-clusters", type=int, default=20, help="KMeans 上限K")
    parser.add_argument(
        "--name-clusters",
        action="store_true",
        help="LLM を使って各クラスタに短いラベルを自動生成（Chat API 呼び出し）",
    )
    parser.add_argument(
        "--name-model",
        default=namer.DEFAULT_MODEL,
        help="クラスタ名生成に使う OpenAI Chat モデル",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="md",
        choices=["md", "html", "both"],
        help="レポート形式. md (既定), html, both",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    """パイプライン本体. 出力先ディレクトリを返す."""
    # 出力先はタイムスタンプ付きで一意化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    _configure_logging(args.log_level, run_dir / "run.log")

    # 列指定モードの解決
    text_cols, column_labels = _parse_column_specs(args)
    if text_cols:
        args.text_col = None  # multi-col モード
        display_cols = ", ".join(text_cols)
        logger.info("複数列モード: %s", display_cols)
    else:
        # text-col が未指定なら候補を提示して対話的に決定
        text_col = args.text_col or _resolve_text_col_interactively(args.input)
        args.text_col = text_col
        display_cols = text_col

    logger.info("=== voice-classifier 開始 ===")
    logger.info("input=%s cols=%s output=%s", args.input, display_cols, run_dir)

    # ネーミングステップを実行するかでフェーズ数が変わる
    total_steps = 6 if args.name_clusters else 5
    reporter_ui = progress.ProgressReporter(total_steps=total_steps)
    reporter_ui.banner("voice-classifier")

    # Step 1: CSV 読み込み + 正規化
    with reporter_ui.step("CSVを読み込み・正規化") as step:
        step.detail(f"入力: {args.input}")
        if text_cols:
            step.detail(f"複数列結合: {', '.join(text_cols)}")
            df = loader.load_csv(
                args.input, text_cols=text_cols, column_labels=column_labels
            )
        else:
            step.detail(f"テキスト列: {args.text_col}")
            df = loader.load_csv(args.input, text_col=args.text_col)
        step.set_summary(f"→ {len(df):,}件（重複集約後）")

    # Step 2: 埋め込み取得（キャッシュ優先）
    with reporter_ui.step("埋め込みベクトルを取得") as step:
        texts = df["_normalized_text"].tolist()
        step.detail(f"モデル: {args.model}")
        embeddings = embedder.get_embeddings(
            texts=texts,
            cache_dir=args.cache_dir,
            model=args.model,
        )
        step.set_summary(
            f"→ shape={embeddings.shape[0]:,}×{embeddings.shape[1]}"
        )

    # Step 3: 自動チューニング
    with reporter_ui.step("クラスタリング候補を探索") as step:
        step.detail(
            f"KMeans/DBSCAN/HDBSCAN を {tuner.SWEEP_SAMPLE_SIZE}件のサンプルで走査"
        )
        best = tuner.find_best_clustering(
            embeddings,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
        )
        step.set_summary(
            f"→ 採用 {best.algorithm} "
            f"{_format_params_short(best.params)} "
            f"(score={best.silhouette:.4f}, "
            f"clusters={best.n_clusters}, noise={best.n_noise})"
        )

    # Step 4: 代表テキスト抽出
    with reporter_ui.step("代表テキストを抽出") as step:
        summaries = clusterer.summarize_clusters(
            df=df,
            embeddings=embeddings,
            labels=best.labels,
            top_k=args.top_k,
        )
        step.set_summary(f"→ 各クラスタから上位{args.top_k}件ずつ")

    # Step 5 (任意): クラスタ名の LLM 生成
    cluster_names: dict[int, str] = {}
    if args.name_clusters:
        with reporter_ui.step("クラスタ名を生成") as step:
            step.detail(f"モデル: {args.name_model}")
            cluster_names = namer.generate_cluster_names(
                summaries=summaries,
                cache_dir=args.cache_dir,
                model=args.name_model,
            )
            # 生成ラベルを逐次表示（最大 3 件）
            preview = [
                f"#{cid}:{name}"
                for cid, name in list(cluster_names.items())[:3]
            ]
            step.set_summary(f"→ {len(cluster_names)}件 例: {', '.join(preview)}")

    # Final step: レポート出力
    with reporter_ui.step("レポートを生成") as step:
        step.detail(f"出力先: {run_dir}")
        # レポート表示用のテキスト列ラベル（複数列時は結合表示）
        report_text_col = args.text_col if args.text_col else ", ".join(text_cols)
        reporter.write_report(
            output_dir=run_dir,
            df=df,
            labels=best.labels,
            summaries=summaries,
            best=best,
            input_path=args.input,
            text_col=report_text_col,
            cluster_names=cluster_names,
            output_format=args.output_format,
        )
        # 出力ファイル名を format から決定して表示
        if args.output_format == "html":
            artifacts = "report.html / parameter_search.html / clusters.csv / params.json"
        elif args.output_format == "both":
            artifacts = (
                "report.md+html / parameter_search.md+html / "
                "clusters.csv / params.json"
            )
        else:
            artifacts = "report.md / parameter_search.md / clusters.csv / params.json"
        step.set_summary(f"→ {artifacts}")

    reporter_ui.footer(str(run_dir))
    logger.info("=== 完了: %s ===", run_dir)
    return run_dir


def _format_params_short(params: dict) -> str:
    """短縮パラメータ表示（CLI進捗用）."""
    parts: list[str] = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    """スクリプト実行のエントリ."""
    load_dotenv()
    args = parse_args(argv)
    try:
        run(args)
    except Exception:
        logger.exception("パイプラインが失敗しました")
        return 1
    return 0


def _parse_column_specs(
    args: argparse.Namespace,
) -> tuple[list[str] | None, dict[str, str]]:
    """--text-cols と --column-labels を解析してタプルで返す.

    Returns:
        (列リスト or None, ラベル辞書). --text-cols 未指定なら (None, {}).
    """
    if not args.text_cols:
        return None, {}

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    if not text_cols:
        raise ValueError("--text-cols が空です")

    labels: dict[str, str] = {}
    if args.column_labels:
        for pair in args.column_labels.split(","):
            if "=" not in pair:
                raise ValueError(
                    f"--column-labels の形式が不正 ('key=value' 形式): {pair}"
                )
            key, value = pair.split("=", 1)
            labels[key.strip()] = value.strip()

    if args.text_col:
        raise ValueError("--text-col と --text-cols は同時に指定できません")

    return text_cols, labels


def _resolve_text_col_interactively(input_path: Path) -> str:
    """`--text-col` 未指定時、CSVを解析して候補を提示しユーザに選ばせる.

    - 1 件しか候補がなければ確認のみで即採用
    - 複数あれば番号で選択、Enter で先頭候補
    """
    candidates = loader.suggest_text_columns(input_path)
    if not candidates:
        raise ValueError(
            f"{input_path} からテキスト列候補が見つかりません. "
            "--text-col で明示的に指定してください"
        )

    print("\n対象テキスト列の候補:", file=sys.stderr)
    for idx, cand in enumerate(candidates, start=1):
        sample_preview = " / ".join(
            s[:40] + ("…" if len(s) > 40 else "") for s in cand.sample_values
        )
        print(
            f"  [{idx}] {cand.name}  "
            f"(平均 {cand.avg_length:.0f}字, 非空 {cand.non_empty_ratio * 100:.0f}%, "
            f"ユニーク率 {cand.unique_ratio * 100:.0f}%)\n"
            f"       例: {sample_preview}",
            file=sys.stderr,
        )

    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"\n単一候補のため採用: {chosen.name}\n", file=sys.stderr)
        return chosen.name

    # 対話入力（非対話環境では先頭候補を採用）
    default = candidates[0]
    if not sys.stdin.isatty():
        print(
            f"\n非対話モード: 先頭候補 {default.name} を採用\n",
            file=sys.stderr,
        )
        return default.name

    while True:
        raw = input(f"\n番号を選択 [1-{len(candidates)}], Enter で [{default.name}]: ")
        choice = raw.strip()
        if not choice:
            return default.name
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1].name
        print("無効な入力です", file=sys.stderr)


def _configure_logging(level: str, log_path: Path) -> None:
    """stderr と run.log の両方にログを出す."""
    root = logging.getLogger()
    root.setLevel(level)

    # ハンドラが既に設定されているなら一旦リセット（再実行時の重複防止）
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
