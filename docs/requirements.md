# Requirements Specification

Status: **v1.0**, 2026-04-16
Target: internal development reference for voice-classifier.

---

## 1. Purpose

Classify free-text customer-voice records (support tickets, repair intake,
feedback forms, etc.) into semantically coherent clusters, surface a
human-readable label and summary per cluster, and deliver reports suitable
for both analysts (Markdown/HTML) and downstream automation (CSV/JSON).

The tool is operated by business analysts and data engineers on a single
workstation; it is **not** a server-side service.

---

## 2. Users & Usage Scenarios

| Persona | Usage scenario |
|---|---|
| Customer Support Manager | Feeds last month's tickets into the tool and reads `report.md` to understand top pain points and volume share. |
| QA / Repair Analyst | Uses `clusters.csv` and `<input>_classified.csv` to slice failure modes and trend them over time. |
| Data Engineer | Integrates `params.json` + `<input>_classified.csv` into a downstream dashboard or data warehouse. |
| Pipeline Developer | Extends modules (new algorithm, different embedding provider, bespoke reporter) via the existing module contract. |

---

## 3. Scope

### 3.1 In scope

- CSV ingestion (RFC 4180, BOM, CRLF/LF/CR, CP932/UTF-8/EUC-JP/ISO-2022-JP).
- Text normalisation (NFKC, whitespace collapse, deduplication).
- Embedding retrieval via OpenAI API with on-disk cache.
- Automatic clustering algorithm selection across KMeans / HDBSCAN / Leiden.
- Silhouette-based scoring with a noise-ratio usability filter.
- Dimensionality reduction (PCA) for high-dimensional embeddings.
- LLM-based cluster labelling and summarisation, grounded on an inferred
  dataset context, with duplicate-label resolution.
- Markdown, HTML (with inline SVG chart), and CSV / JSON outputs.
- Idempotent per-run output directories (timestamped).

### 3.2 Out of scope

- Multi-tenant service / REST API delivery (an internal CLI only).
- Cross-dataset comparison, time-series tracking across runs.
- Real-time / streaming ingestion.
- Authentication beyond API key material in `.env`.
- Any cloud provider other than OpenAI.
- UI / GUI (Markdown / HTML reports are the interface).

---

## 4. Functional Requirements

### 4.1 Input handling (FR-IN-*)

- **FR-IN-01**: Accept a local CSV file via `--input`.
- **FR-IN-02**: Detect encoding by BOM first, then try UTF-8, CP932, EUC-JP,
  ISO-2022-JP in order.
- **FR-IN-03**: Parse CSV per RFC 4180, accepting quoted newlines and escaped
  double quotes.
- **FR-IN-04**: Detect duplicate header names and surface an error listing the
  offending names.
- **FR-IN-05**: Detect column-count mismatches and report affected line numbers
  (capped at 10).
- **FR-IN-06**: Support single-column (`--text-col`) and multi-column
  (`--text-cols` + `--column-labels`) modes.
- **FR-IN-07**: When neither text-col flag is given, present an interactive
  ranked list of candidate columns and let the user pick.
- **FR-IN-08**: Normalise text with NFKC, whitespace collapse, and line-ending
  unification; drop empty rows; deduplicate identical normalised texts and
  preserve original row counts.
- **FR-IN-09**: Truncate any single row to ≤ 4,000 characters before embedding.

### 4.2 Embedding (FR-EMB-*)

- **FR-EMB-01**: Use OpenAI embedding models (default `text-embedding-3-small`).
- **FR-EMB-02**: Cache results by SHA-256 of the normalised text, keyed per
  model, in `cache/embeddings_<model>.pkl`.
- **FR-EMB-03**: Serve cache hits without any API call; log the hit ratio.
- **FR-EMB-04**: Issue API requests in parallel batches (default 8 workers ×
  100 texts/batch).
- **FR-EMB-05**: Retry RateLimitError / APIError with exponential backoff up to
  5 times before failing.

### 4.3 Clustering (FR-CL-*)

- **FR-CL-01**: L2-normalise embeddings; substitute `e_0` for any zero-norm row.
- **FR-CL-02**: Apply PCA to 30 components when input dimensionality > 50.
- **FR-CL-03**: Sweep the following candidate set on a 1,500-row subsample:
    - KMeans: `k ∈ {2, 3, 5, 7, 10, 15, 20, 30, 50, 80}` clipped to `[min,max]`.
    - HDBSCAN: `min_cluster_size ∈ {5, 10, 15, 20, 30, 50, 80, 100}`, `min_samples=5`.
    - Leiden (optional): `resolution ∈ {0.3, 0.5, 0.7, 1.0, 1.3, 1.7}`, n_neighbors=15.
- **FR-CL-04**: Score each trial with cosine silhouette (noise excluded).
- **FR-CL-05**: Reject candidates whose noise ratio on the sample exceeds 50%.
  Fall back to scoring-only if every candidate is rejected.
- **FR-CL-06**: Refit the winning configuration on the full data.
- **FR-CL-07**: Extract the top-K rows closest to each cluster's L2-normalised
  centroid (K defaults to 5; configurable via `--top-k`).

### 4.4 LLM annotation (FR-LLM-*)

- **FR-LLM-01**: When enabled (default `--name-clusters`), first sample 5
  unique records and query the LLM for a dataset context
  (domain + granularity hint).
- **FR-LLM-02**: Generate label + summary per cluster in parallel (max 8
  concurrent requests) with the dataset context injected into the system
  prompt.
- **FR-LLM-03**: Detect labels that collide across clusters and regenerate the
  smaller cluster's label with a differentiation prompt until all labels are
  unique (max 3 iterations; log a warning when unconverged).
- **FR-LLM-04**: Cache results by SHA-256(rep_texts + grounding_salt), keyed
  per chat model.
- **FR-LLM-05**: Absorb API-spec differences across model families (GPT-5,
  o-series, GPT-4o, GPT-3.5) for `max_tokens` vs `max_completion_tokens`,
  `temperature`, and `response_format`.
- **FR-LLM-06**: Request output in the **same language as the input records**
  via prompt instruction.

### 4.5 Reporting (FR-RPT-*)

- **FR-RPT-01**: Per run, create a timestamped directory
  `data/output/YYYYMMDD_HHMMSS/`.
- **FR-RPT-02**: Always emit `clusters.csv` (one row per cluster),
  `<input>_classified.csv` (original rows + labels), `params.json`, and
  `run.log`.
- **FR-RPT-03**: Emit `report.md` and/or `report.html` based on `--format`.
- **FR-RPT-04**: Always emit `parameter_search.html` with an inline SVG
  chart (silhouette bars + cluster-count line) plus tabulated rankings and
  exclusion reasons.
- **FR-RPT-05**: `report.md` per-cluster section must contain:
    - Heading with cluster id, LLM label (if available), and size.
    - "Representative text" = LLM summary (or placeholder when disabled).
    - "Raw data near centroid" — visible list of the top-K raw rows.
- **FR-RPT-06**: `run.log` captures INFO+ events irrespective of the stderr
  `--log-level` setting.

### 4.6 CLI (FR-CLI-*)

- **FR-CLI-01**: Provide a single entrypoint `src/pipeline.py` / `python -m src.pipeline`.
- **FR-CLI-02**: Print a step-based progress display; one persistent line per
  completed step on a TTY.
- **FR-CLI-03**: Exit code 0 on success; 1 on unhandled exception.
- **FR-CLI-04**: Support `--log-level DEBUG|INFO|WARNING|ERROR` for stderr
  verbosity.

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **NFR-PERF-01**: End-to-end execution for 10,000 unique texts must complete
  within 2 minutes on a modern laptop, given a warm embedding cache.
- **NFR-PERF-02**: LLM-based labelling must add at most 2 minutes for ≤ 300
  clusters in typical network conditions (8-wide concurrency).
- **NFR-PERF-03**: Silhouette scoring is bounded to a 2,000-row subsample to
  keep O(n²) operations bounded.

### 5.2 Reliability

- **NFR-REL-01**: A failing API call within retry budget must not abort the
  run.
- **NFR-REL-02**: A failing per-cluster LLM annotation must fall back to
  `Cluster #N` without aborting other clusters.
- **NFR-REL-03**: Corrupt cache files must be silently discarded and rebuilt.

### 5.3 Portability

- **NFR-PORT-01**: Python 3.10+ on Windows, macOS, and Linux.
- **NFR-PORT-02**: Every third-party clustering backend (hdbscan, hnswlib,
  igraph, leidenalg) is optional at runtime; the pipeline adapts the
  candidate set to what's installed.

### 5.4 Observability

- **NFR-OBS-01**: `run.log` records phase boundaries, cache hit rates, PCA
  before/after dimensions, chosen parameters, and silhouette scores.
- **NFR-OBS-02**: `params.json` records enough metadata to reproduce a run
  (algorithm, params, sample size, PCA dims, filter thresholds).

### 5.5 Privacy & Security

- **NFR-SEC-01**: `.env`, `data/input/`, `data/output/`, `cache/` are
  git-ignored; no secrets or PII may be committed.
- **NFR-SEC-02**: The only permitted outbound network call is to the OpenAI
  API (embeddings + chat completions).
- **NFR-SEC-03**: API keys are sourced from `OPENAI_API_KEY` env or `.env`,
  never hard-coded, never echoed to logs or reports.

### 5.6 Maintainability

- **NFR-MAIN-01**: Every source file has an English docstring and English
  inline comments.
- **NFR-MAIN-02**: Only `pipeline.py` performs direct I/O on behalf of the
  modules; other modules stay close to pure functions for testability.
- **NFR-MAIN-03**: Shared helpers (L2 normalisation, hashing, cache I/O,
  retry) live in `src/utils.py`.

### 5.7 Testability

- **NFR-TEST-01**: All OpenAI API calls are mocked in unit tests.
- **NFR-TEST-02**: At least 80% line coverage across `src/` (enforced
  informally via PR review).

---

## 6. Assumptions & Constraints

- **A-01**: The operator has an OpenAI API key with embeddings and chat
  completions access.
- **A-02**: Input CSV fits comfortably in RAM on the operator's workstation
  (typical target ≤ 200,000 rows, ≤ 200 MB).
- **A-03**: Network to the OpenAI API is available during runs that require
  new embeddings / LLM calls.
- **C-01**: The internal Windows workstation must be able to install every
  dependency via pip (no admin rights assumed).
- **C-02**: Runtime must not depend on any local server (no database, no
  message broker).

---

## 7. Acceptance Criteria

The pipeline is considered functionally complete when:

1. Every functional requirement in §4 is exercised by at least one test (unit
   or integration).
2. `pytest` + `ruff` both pass in CI on Python 3.10 and 3.12.
3. Running
   `python src/pipeline.py --input data/input/customer_support_tickets.csv
   --text-col "Ticket Description"`
   produces all expected artefacts in a new timestamped directory within
   2 minutes on a modern laptop (warm cache), with silhouette ≥ 0.30 on the
   provided sample dataset.
