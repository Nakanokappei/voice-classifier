"""CLI エントリポイント — loader → embedder → tuner → clusterer → reporter.

使い方::

    python src/pipeline.py --input data/input/sample.csv --text-col "対応内容"
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# モジュールとして（`python -m src.pipeline`）、またはスクリプトとして（`python src/pipeline.py`）
# どちらでも動くように import 経路を両対応
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import clusterer, embedder, loader, reporter, tuner
else:
    from . import clusterer, embedder, loader, reporter, tuner


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

    # text-col が未指定なら候補を提示して対話的に決定
    text_col = args.text_col or _resolve_text_col_interactively(args.input)
    args.text_col = text_col

    logger.info("=== voice-classifier 開始 ===")
    logger.info("input=%s text_col=%s output=%s", args.input, text_col, run_dir)

    # Step 1: CSV 読み込み + 正規化
    df = loader.load_csv(args.input, text_col)

    # Step 2: 埋め込み取得（キャッシュ優先）
    texts = df["_normalized_text"].tolist()
    embeddings = embedder.get_embeddings(
        texts=texts,
        cache_dir=args.cache_dir,
        model=args.model,
    )
    logger.info("embedding shape=%s", embeddings.shape)

    # Step 3: 自動チューニング
    best = tuner.find_best_clustering(
        embeddings,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )

    # Step 4: 代表テキスト抽出
    summaries = clusterer.summarize_clusters(
        df=df,
        embeddings=embeddings,
        labels=best.labels,
        top_k=args.top_k,
    )

    # Step 5: レポート出力
    reporter.write_report(
        output_dir=run_dir,
        df=df,
        labels=best.labels,
        summaries=summaries,
        best=best,
        input_path=args.input,
        text_col=args.text_col,
    )

    logger.info("=== 完了: %s ===", run_dir)
    return run_dir


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
