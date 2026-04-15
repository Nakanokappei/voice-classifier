# Architecture

## Modules and Data Flow

```
┌──────────┐   DataFrame      ┌──────────┐   ndarray(N,D)       ┌─────────┐
│ loader   │ ───────────────▶ │ embedder │ ───────────────────▶ │ tuner   │
└──────────┘                  └──────────┘                       └────┬────┘
                                                                      │ BestConfig
                                                                      ▼
                        ┌──────────┐     labels / reps      ┌───────────┐
                        │ reporter │ ◀────────────────────── │ clusterer │
                        └────┬─────┘                         └───────────┘
                             │              (optional)             ▲
                             │              ┌──────────────────────┤
                             ▼              │ annotations          │
                 data/output/YYYYMMDD_HHMMSS/├─ annotations ┌──────┴───┐
                 ├─ report.md / report.html │              │ advisor  │
                 ├─ parameter_search.html   ┌┴───────┐      └────┬─────┘
                 ├─ clusters.csv            │ namer  │        (LLM verdict
                 ├─ <input>_classified.csv  └────────┘         injected into
                 └─ params.json           (LLM labels &        parameter_
                                          summaries)          search.html)
```

`pipeline.py` is the only entrypoint; it calls each module in order.

## Module Responsibilities

### `loader.py`
- Read the CSV and extract the chosen text column.
- Apply character normalisation (NFKC), whitespace collapse, and empty-row removal.
- Return a `pandas.DataFrame` (original columns + `_normalized_text`).

### `embedder.py`
- Default to `text-embedding-3-small`.
- **Cache required**: keyed by SHA-256 of the text, persisted to
  `cache/embeddings_<model>.pkl`.
- Batch fetching (up to `BATCH_SIZE`) with parallel requests.
- Returns `np.ndarray` with shape `(N, D)`.

### `tuner.py`
- Enumerate candidate methods and parameters:
    - KMeans: `k ∈ [min_k, max_k]`
    - HDBSCAN: sweep over `min_cluster_size`
    - Leiden: HNSW k-NN graph + community detection over a `resolution` grid
- Score each candidate with cosine silhouette.
- Return a `BestConfig(algorithm, params, score, labels)`.
- See [`algorithm.md`](algorithm.md).

### `clusterer.py`
- Take a `BestConfig` and extract representative texts.
- For each cluster, return the top-N rows closest to the centroid
  (cosine distance).
- Noise (label `-1`) is grouped but carries no representatives.

### `namer.py` (optional)
- Infer the dataset's business context from a small sample.
- Generate label + summary per cluster in parallel, grounded on the context.
- Resolve duplicate labels by differentiating the smaller clusters.

### `advisor.py` (optional)
- Builds a compact `RunDigest` from the selected configuration, coverage
  stats (top-N cumulative share), top cluster labels, dataset domain, and
  whether dedup converged.
- Calls a stronger chat model (default `gpt-5.4`) to produce a Markdown
  advisory with four sections: Verdict, How to use these clusters, Caveats,
  Recommended next steps.
- Reasons over the entire run, not a single cluster — which is why this
  module is separate from `namer.py` and uses a larger model.
- Failure is non-fatal; when the API call fails, an empty string is
  returned and the rest of the report is still written.

### `reporter.py`
- `report.md` / `report.html`: clustering outcome with the LLM summary (or a
  placeholder) plus the raw near-centroid items.
- `parameter_search.html`: full search report with a dual-axis SVG chart
  and a Pareto coverage curve. When the advisor ran, its Markdown verdict
  is inserted as an `.advisory` section immediately below the chart.
- `clusters.csv`: one row per cluster — `cluster_id, cluster_name, size,
  summary, rep_1..N`.
- `<input>_classified.csv`: original data plus `cluster_id` / `cluster_name`.
- `params.json`: machine-readable metadata.

### `pipeline.py`
- Parse CLI arguments.
- Create a timestamped output directory.
- Run `loader → embedder → tuner → clusterer → (namer) → (advisor) → reporter`.
- Log to stderr and `output_dir/run.log` (INFO+ always captured).

## Types and Interfaces

```python
# tuner.py
@dataclass
class BestConfig:
    algorithm: Literal["kmeans", "hdbscan", "leiden"]
    params: dict[str, Any]
    silhouette: float
    labels: np.ndarray  # shape (N,)
    n_clusters: int
    n_noise: int

# clusterer.py
@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    representative_indices: list[int]
    representative_texts: list[str]

# namer.py
@dataclass
class ClusterAnnotation:
    label: str
    summary: str

# advisor.py
@dataclass
class RunDigest:
    target: str                                  # faq / chatbot / insight
    algorithm: str
    params_text: str
    silhouette: float
    n_clusters: int
    n_noise: int
    total_rows: int
    noise_ratio_pct: float
    max_share_pct: float
    coverage_top_n: list[tuple[int, float]]      # (N, cum% of total)
    coverage_top_n_ex_noise: list[tuple[int, float]]
    top_cluster_labels: list[tuple[str, int]]
    dataset_domain: str
    dataset_hint: str
    dedup_converged: bool
```

## I/O Contract

- **Only `pipeline.py` performs I/O directly.** `loader.py` takes a path
  passed in by the pipeline; everything else operates on in-memory data.
- Other modules stay close to pure functions so they remain easy to test.

## Failure Modes

| Situation | Handling |
|---|---|
| `OPENAI_API_KEY` unset | `RuntimeError` at startup |
| OpenAI API failure (rate limit, etc.) | Exponential backoff with N retries; then `raise` |
| Effective sample count < `min_clusters` | `ValueError` before tuning |
| All noise | `clusterer` emits a noise-only report; score recorded as `nan` |
