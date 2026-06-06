# Detailed Design

Module-level internals. For the high-level overview see
[`architecture.md`](architecture.md); for algorithmic rationale see
[`algorithm.md`](algorithm.md).

---

## 1. Module map

```
src/
├── pipeline.py     CLI entrypoint; orchestrates the whole run.
├── loader.py       Input CSV parsing, text normalisation, dedup, column pick.
├── embedder.py     Azure OpenAI Embeddings with on-disk cache + parallel batches.
├── tuner.py        Clustering sweep (KMeans/HDBSCAN/Leiden) + winner refit.
├── clusterer.py    Representative row extraction (top-K near centroid).
├── namer.py        LLM cluster labelling + summarisation + dedup.
├── advisor.py      LLM advisory note summarising the whole run (stronger model).
├── reporter.py     Markdown / HTML / CSV / JSON output writers.
├── progress.py     CLI phase/progress display.
└── utils.py        Cross-module helpers (normalise, hash, cache, retry).
```

Only `pipeline.py` performs external I/O directly (CSV paths, stdin/stderr).
Every other module accepts in-memory inputs and returns in-memory outputs,
making each unit testable in isolation.

---

## 2. Data flow

```
pipeline.run(args)
    │
    ├─ loader.load_csv(path, text_col=..., text_cols=...)
    │     ├─ _detect_encoding(path)
    │     ├─ _validate_raw_structure(path, encoding)      # header/row count checks
    │     ├─ _read_csv_rfc4180(path, encoding)            # pandas read
    │     ├─ _validate_csv_structure(df)
    │     └─ build `_normalized_text`, dedup, truncate
    │           -> DataFrame (all original columns + _normalized_text, _duplicate_count)
    │
    ├─ embedder.get_embeddings(texts, cache_dir, model)
    │     ├─ cache hits filtered out
    │     ├─ _embed_texts_in_parallel()                    # 8 worker threads
    │     │     └─ _embed_batch() via utils.call_with_exponential_backoff
    │     └─ utils.save_pickle_cache(cache_path, cache)
    │           -> ndarray (N, D) float32
    │
    ├─ tuner.find_best_clustering(embeddings, min_k, max_k)
    │     ├─ utils.l2_normalize(embeddings)
    │     ├─ _reduce_dimensions()                          # PCA to 30 dims if >50
    │     ├─ sample 1,500 rows
    │     ├─ _run_all_sweeps(sample, ...)
    │     │     ├─ _sweep_kmeans / _sweep_hdbscan / _sweep_leiden
    │     │     └─ each delegates to _run_sweep (shared loop)
    │     ├─ _select_winner() with noise-ratio filter
    │     ├─ _fit_winner_on_full_data()                    # refit on all N rows
    │     └─ _evaluate_silhouette_on_subsample()
    │           -> BestConfig
    │
    ├─ clusterer.summarize_clusters(df, embeddings, labels, top_k)
    │     └─ _pick_centroid_nearest() per cluster
    │           -> list[ClusterSummary]
    │
    ├─ if args.name_clusters:
    │     ├─ namer.infer_dataset_context(texts)            # 5-sample inference
    │     ├─ namer.generate_cluster_annotations(summaries, ...)
    │     │     └─ _annotate_clusters_in_parallel() via _invoke_annotation_llm
    │     └─ namer.resolve_label_duplicates(summaries, annotations, ...)
    │           └─ _differentiate_labels_in_parallel() × up to 3 iterations
    │
    └─ reporter.write_report(output_dir, df, labels, summaries, best,
                             text_col, cluster_names, cluster_annotations, format)
          ├─ _write_classified_rows_csv     -> <input>_classified.csv
          ├─ _write_cluster_list_csv        -> clusters.csv
          ├─ _write_params_json             -> params.json
          ├─ _build_clustering_report_md    -> report.md / report.html
          └─ _build_parameter_search_md + _build_parameter_search_chart_svg
                                            -> parameter_search.html
```

---

## 3. Module contracts

### 3.1 `loader`

**Public**

```python
def load_csv(
    path: Path | str,
    text_col: str | None = None,
    text_cols: list[str] | None = None,
    column_labels: dict[str, str] | None = None,
) -> pd.DataFrame

def suggest_text_columns(path: Path | str, top_k: int = 5) -> list[ColumnCandidate]
```

**Invariants**

- Returned DataFrame is row-aligned with the caller's model of the data:
  every remaining row has a non-empty `_normalized_text`.
- `_duplicate_count` ≥ 1; the sum of counts equals the original non-empty row
  count, minus rows that normalised to empty.
- Every column of the source CSV is preserved.

**Failure modes**

- `FileNotFoundError`: path doesn't exist.
- `ValueError`: empty file, empty header, duplicate header, row/column
  mismatch, missing text column, both `text_col` and `text_cols` set.

### 3.2 `embedder`

**Public**

```python
def get_embeddings(
    texts: list[str],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> np.ndarray
```

**Invariants**

- Output order matches input order exactly, even when batches complete out
  of order.
- Cache never grows unbounded in a single run — only misses are added.
- An atomic `tmp → replace` pattern protects the cache from partial writes.

**Failure modes**

- `RuntimeError` (API key missing, retries exhausted).
- `ValueError` (empty `texts`).

### 3.3 `tuner`

**Public**

```python
def find_best_clustering(
    embeddings: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 20,
) -> BestConfig
```

**Invariants**

- `BestConfig.labels.shape == (N,)` where `N == embeddings.shape[0]`.
- `BestConfig.all_trials` contains a standardised dict per evaluated
  candidate, including those filtered out by the noise-ratio rule.
- The same (input, seed) yields the same `BestConfig.labels`.

**Extension**

Adding a new algorithm requires:

1. An optional import block in the module preamble.
2. A `_sweep_<name>(sample)` function that delegates to `_run_sweep`.
3. Appending it to the `Algorithm` Literal.
4. Adding a branch in `_fit_winner_on_full_data`.
5. Adding it to the reporter's score-progression table.

### 3.4 `clusterer`

**Public**

```python
def summarize_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: int = 5,
) -> list[ClusterSummary]
```

**Invariants**

- `ClusterSummary.size` matches the number of rows assigned to the cluster.
- Non-noise cluster list is sorted by ascending cluster id; noise, when
  present, is appended last.
- Representative picks live in `df`'s index space; callers can use
  `df.iloc[rep_index]` directly.

### 3.5 `namer`

**Public**

```python
def infer_dataset_context(texts: list[str], ...) -> DatasetContext

def generate_cluster_annotations(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    dataset_context: DatasetContext | None = None,
) -> dict[int, ClusterAnnotation]

def resolve_label_duplicates(
    summaries: list[ClusterSummary],
    annotations: dict[int, ClusterAnnotation],
    ...
) -> dict[int, ClusterAnnotation]
```

**Invariants**

- `ClusterAnnotation.label` is 1-30 chars, stripped of quotation marks and
  common decoration.
- `ClusterAnnotation.summary` is at most 400 chars, single-line (newlines
  collapsed to spaces).
- Cache keys include a grounding-hint salt so different datasets don't
  collide in a shared cache.

**Model-family adaptation**

`_build_chat_kwargs(model, ..., max_tokens, temperature, json_mode)` centralises
the parameter differences:

- `max_completion_tokens` for `gpt-5*`, `o1*`, `o3*`, `o4*`.
- `max_tokens` for everything older.
- Omit `temperature` for o-series reasoning models.

To onboard a new family, update these two helpers (`_uses_max_completion_tokens`
and `_supports_custom_temperature`).

### 3.6 `advisor`

**Public**

```python
@dataclass
class RunDigest:
    target: str
    algorithm: str
    params_text: str
    silhouette: float
    n_clusters: int
    n_noise: int
    total_rows: int
    noise_ratio_pct: float
    max_share_pct: float
    coverage_top_n: list[tuple[int, float]]      # (N, cum% of total rows)
    coverage_top_n_ex_noise: list[tuple[int, float]]
    top_cluster_labels: list[tuple[str, int]]    # top 15 by size
    dataset_domain: str
    dataset_hint: str
    dedup_converged: bool

def build_run_digest(
    best: BestConfig,
    summaries: list[ClusterSummary],
    cluster_annotations: dict[int, ClusterAnnotation] | None,
    total_rows: int,
    dataset_context: DatasetContext | None,
    dedup_converged: bool,
) -> RunDigest

def generate_run_advice(
    digest: RunDigest,
    model: str | None = None,   # resolves AZURE_OPENAI_ADVISOR_DEPLOYMENT
    api_key: str | None = None,
) -> str  # Markdown; empty string on failure
```

**Rationale**

- The advisor is a separate module from `namer` because it uses a stronger
  deployment (`AZURE_OPENAI_ADVISOR_DEPLOYMENT` vs.
  `AZURE_OPENAI_NAMER_DEPLOYMENT`) and reasons over the whole run rather than
  a single cluster.
- `build_run_digest` is pure data reduction. It can be tested without any
  network access, and `generate_run_advice` is the only function that
  performs API I/O.
- Failure is non-fatal: `generate_run_advice` returns `""` on retry
  exhaustion, and the caller simply omits the advisory section.
- The Markdown sections are prescribed in the system prompt (`## Advisory`,
  then `### Verdict`, `### How to use these clusters`, `### Caveats and
  things to watch`, `### Recommended next steps`) so `reporter.py` can
  inject a single `<section class="advisory">` wrapper in HTML.

**Invariants**

- The digest reports coverage at ranks `[1, 5, 10, 20, 50, 100]` (clamped
  to the number of non-noise clusters actually available).
- `top_cluster_labels` is at most 15 items; beyond that the prompt grows
  without adding signal.
- An accidental ``` fence around the response is stripped by
  `_sanitise_markdown` before return.

### 3.7 `reporter`

**Public**

```python
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
    advice_md: str = "",
) -> None
```

When `advice_md` is non-empty it is rendered to HTML and injected as a
`<section class="advisory">` block immediately below the sweep chart in
`parameter_search.html`. The `.advisory` CSS rule is part of the embedded
stylesheet so the report remains a single self-contained file.

**Artefact decisions**

| Artefact | Always? | Format control |
|---|---|---|
| `clusters.csv` | yes | fixed CSV |
| `<input>_classified.csv` | yes | fixed CSV |
| `params.json` | yes | fixed JSON |
| `report.md` | `--format` in `md` / `both` | Markdown |
| `report.html` | `--format` in `html` / `both` | HTML |
| `parameter_search.html` | yes | HTML (with embedded SVG) |

`parameter_search.md` **was** part of an earlier design and has been
retired; the chart-enabled HTML is the single source of truth.

### 3.8 `progress`

Phase-based reporter with ANSI-aware line overwriting. Contract is
intentionally small — if another module ever needs progress reporting, it
should use this class rather than rolling its own.

### 3.9 `utils`

Shared primitives. Any duplication of these helpers across modules is a
code smell — add a new helper here instead.

```python
def l2_normalize(x: np.ndarray) -> np.ndarray
def content_hash(payload: str) -> str
def load_pickle_cache(path: Path) -> dict
def save_pickle_cache(path: Path, cache: dict, *, also_json: bool = False) -> None
def call_with_exponential_backoff(
    fn: Callable[[], T],
    *,
    log_prefix: str,
    max_retries: int = 4,
    base_sec: float = 2.0,
    retriable: tuple = OPENAI_RETRIABLE_ERRORS,
) -> T
```

---

## 4. Key data structures

```python
@dataclass
class BestConfig:
    algorithm: Literal["kmeans", "hdbscan", "leiden"]
    params: dict[str, Any]
    silhouette: float
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    all_trials: list[dict[str, Any]]  # one per candidate
    sweep_sample_size: int
    dim_before_pca: int
    dim_after_pca: int

@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    representative_indices: list[int]
    representative_texts: list[str]

@dataclass
class ClusterAnnotation:
    label: str
    summary: str

@dataclass
class DatasetContext:
    domain: str
    granularity_hint: str
    sample_texts: list[str]

@dataclass
class ColumnCandidate:
    name: str
    avg_length: float
    non_empty_ratio: float
    unique_ratio: float
    sample_values: list[str]
```

---

## 5. Logging

- Logger names follow the module path: `src.loader`, `src.embedder`, etc.
- `pipeline.py` attaches two handlers on the root logger:
    - `StreamHandler(stderr)` at the user's `--log-level`.
    - `FileHandler(run.log)` at INFO (or DEBUG when `--log-level DEBUG`).

Only `pipeline.py` touches the handlers; modules just call `logger.info` /
`logger.warning` / `logger.debug`.

---

## 6. Error handling policy

1. **Caller-facing errors** (bad arguments, missing files, malformed CSV)
   surface as `ValueError` / `FileNotFoundError` with explicit messages and
   propagate to `main()`, which logs them and returns exit code 1.
2. **Transient API errors** (rate limit, 5xx, timeouts) go through
   `utils.call_with_exponential_backoff` and are retried. Only when the
   retry budget is exhausted does a `RuntimeError` propagate.
3. **Per-cluster LLM failures** are logged and replaced with a fallback
   label (`Cluster #N`) so the pipeline still completes.
4. **Silhouette computation failures** (single cluster, etc.) return
   `None`, which the selector handles explicitly.

---

## 7. Performance considerations

- **PCA before clustering**: collapses 1,536d embeddings to 30d in-memory
  once per run. This is ~10× faster than clustering raw high-dim vectors.
- **Silhouette on a 2k subsample** instead of the full data keeps O(n²)
  computations bounded.
- **Sweep on a 1.5k subsample** (then refit on full data) caps the sweep at
  a constant cost irrespective of N.
- **Parallel batched embeddings** (8 workers × 100 texts/batch) saturate
  typical Azure OpenAI Standard-tier rate limits without hitting them.
- **LLM annotation parallelism** matches the same 8-worker budget.

---

## 8. Extension points

- **New clustering algorithm**: see §3.3.
- **New embedding provider**: replace `embedder._make_azure_client` and
  `_embed_batch`. Cache schema unchanged.
- **New LLM provider**: replace `namer._make_azure_client` and
  `_build_chat_kwargs`. Cache schema unchanged when the grounding-salt
  contract is preserved.
- **New output format**: add a helper under `reporter` and wire it through
  `write_report`'s `OutputFormat` literal.
- **Alternative dimensionality reduction**: swap `_reduce_dimensions` (e.g.
  UMAP). Keep the post-reduction L2-normalisation so everything downstream
  stays cosine-consistent.
