"""CLI entrypoint — runs loader → embedder → tuner → clusterer → (namer) → reporter.

Usage::

    python src/pipeline.py --input data/input/sample.csv --text-col "response_body"
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
    """Silence known-benign RuntimeWarnings that originate inside sklearn.

    - sklearn.utils.extmath (safe_sparse_dot) occasionally emits divide/overflow/
      invalid-value warnings from randomised SVD or cosine_distances. They are
      numerical noise and do not affect results.
    - With zero-norm rows already replaced by `e_0` upstream, no semantically
      meaningful warning remains to suppress.
    """
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*(divide by zero|overflow|invalid value).*matmul.*",
    )


_suppress_known_benign_warnings()


# Module imports: run both as a script (`python src/pipeline.py`) and as a
# module (`python -m src.pipeline`).
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import clusterer, embedder, loader, namer, progress, reporter, tuner
else:
    from . import clusterer, embedder, loader, namer, progress, reporter, tuner


logger = logging.getLogger("voice_classifier")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Customer-voice auto-classification and insight report system",
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input CSV path"
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help=(
            "Single column to embed. When omitted and --text-cols is also "
            "omitted, candidates are offered interactively."
        ),
    )
    parser.add_argument(
        "--text-cols",
        default=None,
        help=(
            "Comma-separated list of columns to concatenate for embedding. "
            "Example: --text-cols 'Ticket Subject,Ticket Description'. "
            "Mutually exclusive with --text-col."
        ),
    )
    parser.add_argument(
        "--column-labels",
        default=None,
        help=(
            "Optional `column=label` pairs (comma-separated) for multi-column "
            "mode. Example: --column-labels 'Ticket Subject=subject,"
            "Ticket Description=body'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Root output directory (per-run subdir added with a timestamp)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Embedding cache directory",
    )
    parser.add_argument(
        "--model",
        "--embedding-model",
        dest="model",
        default=embedder.DEFAULT_MODEL,
        help=(
            f"OpenAI embedding model. Default: {embedder.DEFAULT_MODEL}. "
            "Alternatives: text-embedding-3-large, text-embedding-ada-002. "
            "Cache is segregated per model so switching is safe."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Representative texts per cluster")
    parser.add_argument("--min-clusters", type=int, default=2, help="Lower bound for K (KMeans)")
    parser.add_argument("--max-clusters", type=int, default=20, help="Upper bound for K (KMeans)")
    parser.add_argument(
        "--name-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate an LLM label and summary for each cluster (default: on). "
            "The summary is shown as the main representative text; the raw "
            "near-centroid items remain visible below as verification data. "
            "Disable with --no-name-clusters to skip the extra LLM calls."
        ),
    )
    parser.add_argument(
        "--name-model",
        "--llm-model",
        dest="name_model",
        default=namer.DEFAULT_MODEL,
        help=(
            f"OpenAI chat model used for cluster labelling. Default: "
            f"{namer.DEFAULT_MODEL}. API-spec differences across GPT-5 / "
            "o-series / GPT-4o / GPT-3.5 (e.g. max_completion_tokens vs "
            "max_tokens, temperature restrictions) are absorbed automatically."
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="md",
        choices=["md", "html", "both"],
        help="Report format for report.md/html. Default md.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (stderr). run.log always captures INFO+.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    """Run the pipeline and return the output directory."""
    # Per-run output directory with a millisecond-unique timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    _configure_logging(args.log_level, run_dir / "run.log")

    # Resolve column selection: CLI flags first, otherwise interactive.
    text_cols, column_labels = _parse_column_specs(args)
    if text_cols:
        args.text_col = None  # multi-column mode from CLI
        display_cols = ", ".join(text_cols)
        logger.info("Multi-column mode: %s", display_cols)
    elif args.text_col:
        display_cols = args.text_col
    else:
        # No CLI flag given — prompt interactively; allow multi-selection.
        picked_single, picked_multi = _resolve_text_cols_interactively(args.input)
        if picked_multi:
            text_cols = picked_multi
            args.text_col = None
            display_cols = ", ".join(text_cols)
            logger.info("Multi-column mode (interactive): %s", display_cols)
        else:
            args.text_col = picked_single
            display_cols = picked_single

    logger.info("=== voice-classifier start ===")
    logger.info("input=%s cols=%s output=%s", args.input, display_cols, run_dir)

    # Step counts depend on whether LLM annotation runs.
    # `--name-clusters`: +3 steps (dataset context, generation, dedup).
    total_steps = 8 if args.name_clusters else 5
    reporter_ui = progress.ProgressReporter(total_steps=total_steps)
    reporter_ui.banner("voice-classifier")

    # Step 1: Load and normalise CSV.
    with reporter_ui.step("Load and normalise CSV") as step:
        step.detail(f"input: {args.input}")
        if text_cols:
            step.detail(f"multi-column: {', '.join(text_cols)}")
            df = loader.load_csv(
                args.input, text_cols=text_cols, column_labels=column_labels
            )
        else:
            step.detail(f"text column: {args.text_col}")
            df = loader.load_csv(args.input, text_col=args.text_col)
        step.set_summary(f"→ {len(df):,} rows (after dedup)")

    # Step 2: Fetch embeddings (cache first).
    with reporter_ui.step("Fetch embeddings") as step:
        texts = df["_normalized_text"].tolist()
        step.detail(f"model: {args.model}")
        embeddings = embedder.get_embeddings(
            texts=texts,
            cache_dir=args.cache_dir,
            model=args.model,
        )
        step.set_summary(
            f"→ shape={embeddings.shape[0]:,}×{embeddings.shape[1]}"
        )

    # Step 3: Clustering sweep and selection.
    with reporter_ui.step("Search clustering candidates") as step:
        step.detail(
            f"KMeans/HDBSCAN/Leiden sweep on {tuner.SWEEP_SAMPLE_SIZE} samples"
        )
        best = tuner.find_best_clustering(
            embeddings,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
        )
        step.set_summary(
            f"→ selected {best.algorithm} "
            f"{_format_params_short(best.params)} "
            f"(score={best.silhouette:.4f}, "
            f"clusters={best.n_clusters}, noise={best.n_noise})"
        )

    # Step 4: Extract representative texts.
    with reporter_ui.step("Extract representative texts") as step:
        summaries = clusterer.summarize_clusters(
            df=df,
            embeddings=embeddings,
            labels=best.labels,
            top_k=args.top_k,
        )
        step.set_summary(f"→ top-{args.top_k} per cluster")

    # Step 5 (optional): LLM annotation flow.
    cluster_annotations: dict[int, namer.ClusterAnnotation] = {}
    cluster_names: dict[int, str] = {}
    if args.name_clusters:
        # 5a. Infer the dataset meaning (grounding context).
        with reporter_ui.step("Infer dataset context") as step:
            step.detail(f"model: {args.name_model}")
            dataset_context = namer.infer_dataset_context(
                texts=df["_normalized_text"].tolist(),
                model=args.name_model,
            )
            step.set_summary(
                f"→ domain={dataset_context.domain} / "
                f"{dataset_context.granularity_hint}"
            )

        # 5b. Generate label + summary in parallel with grounding.
        with reporter_ui.step("Generate cluster label and summary") as step:
            step.detail(f"concurrency: {namer.MAX_CONCURRENCY}")
            cluster_annotations = namer.generate_cluster_annotations(
                summaries=summaries,
                cache_dir=args.cache_dir,
                model=args.name_model,
                dataset_context=dataset_context,
            )
            preview = [
                f"#{cid}:{ann.label}"
                for cid, ann in list(cluster_annotations.items())[:3]
            ]
            step.set_summary(
                f"→ {len(cluster_annotations)} clusters, "
                f"e.g. {', '.join(preview)}"
            )

        # 5c. Resolve label duplicates.
        with reporter_ui.step("Resolve label duplicates") as step:
            before = namer._label_frequencies(cluster_annotations)
            duplicates_before = sum(
                count for count in before.values() if count >= 2
            )
            cluster_annotations = namer.resolve_label_duplicates(
                summaries=summaries,
                annotations=cluster_annotations,
                model=args.name_model,
                dataset_context=dataset_context,
            )
            after = namer._label_frequencies(cluster_annotations)
            duplicates_after = sum(
                count for count in after.values() if count >= 2
            )
            step.set_summary(
                f"→ duplicates: {duplicates_before} → {duplicates_after}"
            )

        cluster_names = {
            cid: ann.label for cid, ann in cluster_annotations.items()
        }

    # Final step: write reports.
    with reporter_ui.step("Write reports") as step:
        step.detail(f"output dir: {run_dir}")
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
            cluster_annotations=cluster_annotations,
            output_format=args.output_format,
        )
        input_stem = Path(str(args.input)).stem
        classified_name = f"{input_stem}_classified.csv"
        if args.output_format == "html":
            report_files = "report.html"
        elif args.output_format == "both":
            report_files = "report.md+html"
        else:
            report_files = "report.md"
        step.set_summary(
            f"→ {report_files} / parameter_search.html / "
            f"clusters.csv / {classified_name} / params.json"
        )

    reporter_ui.footer(str(run_dir))
    logger.info("=== done: %s ===", run_dir)
    return run_dir


def _format_params_short(params: dict) -> str:
    """Short parameter string for CLI output."""
    parts: list[str] = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _parse_column_specs(
    args: argparse.Namespace,
) -> tuple[list[str] | None, dict[str, str]]:
    """Parse `--text-cols` and `--column-labels` into a list and dict.

    Returns:
        (column_list_or_None, labels_dict). (None, {}) when --text-cols is not set.
    """
    if not args.text_cols:
        return None, {}

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    if not text_cols:
        raise ValueError("--text-cols is empty")

    labels: dict[str, str] = {}
    if args.column_labels:
        for pair in args.column_labels.split(","):
            if "=" not in pair:
                raise ValueError(
                    f"--column-labels expects `key=value` pairs, got: {pair}"
                )
            key, value = pair.split("=", 1)
            labels[key.strip()] = value.strip()

    if args.text_col:
        raise ValueError("--text-col and --text-cols are mutually exclusive")

    return text_cols, labels


def _resolve_text_cols_interactively(
    input_path: Path,
) -> tuple[str | None, list[str] | None]:
    """Analyse the CSV and prompt the user to pick one or more text columns.

    Accepts a variety of selection syntaxes at the prompt:

        1            single column
        1,3,5        multiple (comma-separated)
        1;3;5        multiple (semicolon-separated)
        1-3          range (inclusive)
        1-3,5        mixed — expands to 1, 2, 3, 5

    Multi-column selections preserve the order the user typed (not sorted),
    matching the multi-column embedding contract.

    Returns:
        (text_col, None) for a single-column choice, or
        (None, text_cols) for a multi-column choice.
        Both None when no candidate was available (caller raises).
    """
    candidates = loader.suggest_text_columns(input_path)
    if not candidates:
        raise ValueError(
            f"No text column candidates found in {input_path}. "
            "Specify one explicitly with --text-col or --text-cols."
        )

    print("\nText column candidates:", file=sys.stderr)
    for idx, cand in enumerate(candidates, start=1):
        sample_preview = " / ".join(
            s[:40] + ("…" if len(s) > 40 else "") for s in cand.sample_values
        )
        print(
            f"  [{idx}] {cand.name}  "
            f"(avg {cand.avg_length:.0f} chars, non-empty {cand.non_empty_ratio * 100:.0f}%, "
            f"unique {cand.unique_ratio * 100:.0f}%)\n"
            f"       e.g. {sample_preview}",
            file=sys.stderr,
        )

    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"\nOnly one candidate — picking: {chosen.name}\n", file=sys.stderr)
        return chosen.name, None

    default = candidates[0]
    if not sys.stdin.isatty():
        print(
            f"\nNon-interactive mode: picking the top candidate {default.name}\n",
            file=sys.stderr,
        )
        return default.name, None

    prompt = (
        f"\nSelect columns from [1-{len(candidates)}]. "
        "Examples: '1' / '1,3' / '1;3' / '1-3' / '1-2,4'. "
        f"Enter for default [{default.name}]: "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default.name, None

        try:
            indices = _parse_column_selection(raw, len(candidates))
        except ValueError as exc:
            print(f"Invalid input: {exc}", file=sys.stderr)
            continue

        if len(indices) == 1:
            return candidates[indices[0] - 1].name, None
        names = [candidates[i - 1].name for i in indices]
        print(
            f"Selected {len(names)} columns (in order): {', '.join(names)}\n",
            file=sys.stderr,
        )
        return None, names


def _parse_column_selection(raw: str, max_index: int) -> list[int]:
    """Parse a selection string into a list of 1-based indices.

    Grammar:
        selection := token (sep token)*
        sep       := ',' | ';'
        token     := integer | range
        range     := integer '-' integer    (inclusive, ascending)

    Returns indices in the order they were specified, deduplicated while
    preserving the first occurrence. Raises ValueError on any token that
    can't be parsed or falls outside ``[1, max_index]``.
    """
    # Treat ';' as an alias for ','; then split and strip.
    tokens = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        raise ValueError("no numbers provided")

    result: list[int] = []
    seen: set[int] = set()

    def _accept(index: int) -> None:
        if not (1 <= index <= max_index):
            raise ValueError(f"{index} is out of range [1-{max_index}]")
        if index not in seen:
            seen.add(index)
            result.append(index)

    for tok in tokens:
        if "-" in tok:
            parts = tok.split("-", 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"malformed range '{tok}'")
            start, end = int(parts[0]), int(parts[1])
            if end < start:
                raise ValueError(
                    f"range '{tok}' must be ascending (start ≤ end)"
                )
            for idx in range(start, end + 1):
                _accept(idx)
        else:
            if not tok.isdigit():
                raise ValueError(f"'{tok}' is not a number")
            _accept(int(tok))

    return result


def _configure_logging(level: str, log_path: Path) -> None:
    """Set up logging to both stderr and `run.log`.

    - stderr uses the user-specified `--log-level` (so `ERROR` stays quiet).
    - run.log always records INFO and above, even when stderr is set to ERROR,
      so that cache hits and phase outcomes can be reviewed after the fact.
    """
    root = logging.getLogger()
    # Minimum level processed by the handlers below.
    root.setLevel("DEBUG" if level == "DEBUG" else "INFO")

    # Reset handlers so repeated runs in the same process don't duplicate lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # stderr honours the user-requested level (ERROR silences INFO/WARNING).
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    # run.log always captures INFO+ so cache hits and phase decisions are kept.
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel("DEBUG" if level == "DEBUG" else "INFO")
    root.addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint.

    Returns:
        0 on success, 1 on unhandled exception, 130 on Ctrl-C.
        130 = 128 + SIGINT(2), the conventional shell exit code for
        interactive cancellation.
    """
    load_dotenv()
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        # The in-flight step's context manager has already printed the
        # "cancelled by user" marker. Print a summary footer and bail out.
        logger.info("Pipeline cancelled by user")
        print(file=sys.stderr)
        print(
            "\033[1;33m"  # bold yellow
            "Run cancelled by user — partial output may remain under data/output/"
            "\033[0m",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except Exception:
        logger.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
