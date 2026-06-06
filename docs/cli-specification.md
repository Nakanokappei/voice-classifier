# CLI Specification

Authoritative reference for the command-line surface of `voice-classifier`.

---

## 1. Synopsis

```
python src/pipeline.py
    --input PATH
    [--text-col NAME | --text-cols A,B[,C...]]
    [--column-labels A=x,B=y[,...]]
    [--output-dir PATH]
    [--cache-dir PATH]
    [--model NAME | --embedding-model NAME]
    [--top-k N]
    [--min-clusters N]
    [--max-clusters N]
    [--name-clusters | --no-name-clusters]
    [--name-model NAME | --llm-model NAME]
    [--advise | --no-advise]
    [--advisor-model NAME]
    [--format md|html|both]
    [--log-level DEBUG|INFO|WARNING|ERROR]
```

Equivalent to `python -m src.pipeline` when run from the project root.

---

## 2. Arguments

### 2.1 Required

| Flag | Value | Description |
|---|---|---|
| `--input` | file path | Input CSV to classify. Must be readable. |

### 2.2 Column selection (mutually exclusive group — at least one required)

| Flag | Value | Description |
|---|---|---|
| `--text-col` | column name | Single-column mode — embed this column only. |
| `--text-cols` | comma-separated list | Multi-column mode — concatenate these columns as `label: value` lines for each row. |

When both are omitted and stdin is a TTY, the tool prints scored column
candidates and prompts for a selection. In non-interactive environments the
first candidate is auto-selected.

### 2.3 Optional

| Flag | Default | Description |
|---|---|---|
| `--column-labels` | — | `col=label,...` map used as prefixes in multi-column mode. Labels default to the column name. |
| `--output-dir` | `data/output` | Root output directory. A timestamped subdirectory is created inside it. |
| `--cache-dir` | `cache` | Embedding + annotation cache directory. |
| `--model` / `--embedding-model` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure OpenAI embedding deployment name. Cache is segregated per deployment. |
| `--top-k` | `5` | Number of near-centroid representative rows per cluster. |
| `--min-clusters` | `2` | Lower bound for K (KMeans sweep). |
| `--max-clusters` | `20` | Upper bound for K. |
| `--target` | `faq` | Which downstream use case the clustering should optimise for. `faq` prefers 30-80 clusters with max share ≤ 10%. `chatbot` targets 50-150 finer intents with share ≤ 7%. `insight` maximises silhouette with no cluster-count bias. |
| `--name-clusters` / `--no-name-clusters` | on | Toggle LLM labelling + summarisation. |
| `--name-model` / `--llm-model` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | Azure OpenAI chat deployment for labelling. Model-family differences (max_completion_tokens vs max_tokens, temperature restrictions) are absorbed automatically based on the deployment name. |
| `--advise` / `--no-advise` | on | Toggle the LLM advisory note inserted at the top of `parameter_search.html`. No-op without `--name-clusters`. |
| `--advisor-model` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Azure OpenAI chat deployment for the advisory note. Use a stronger model than `--name-model` because it reasons over the whole run. |
| `--format` | `md` | Report format for `report.*`. `parameter_search.html` is always emitted regardless. |
| `--log-level` | `INFO` | stderr verbosity. `run.log` always captures INFO+. |

### 2.4 Argument constraints

- `--text-col` and `--text-cols` cannot both be specified.
- `--column-labels` has no effect in single-column mode.
- `--min-clusters` must be ≥ 2.
- `--max-clusters` must be ≥ `--min-clusters`.
- `--top-k` must be ≥ 1.

---

## 3. Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AZURE_OPENAI_API_KEY` | yes | — | Authenticates every Azure OpenAI API call. Loaded from `.env` via `python-dotenv`. |
| `AZURE_OPENAI_ENDPOINT` | yes | — | Resource endpoint (e.g. `https://<resource>.openai.azure.com`). |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | yes | — | Deployment name used by the embedder. |
| `AZURE_OPENAI_NAMER_DEPLOYMENT` | yes | — | Deployment name used by the namer (cluster labelling). |
| `AZURE_OPENAI_ADVISOR_DEPLOYMENT` | yes (when `--advise`) | — | Deployment name used by the advisor; advisor is skipped when unset. |
| `AZURE_OPENAI_API_VERSION` | no | `2024-10-21` | API version sent on every request. |
| `AZURE_OPENAI_REQUEST_TIMEOUT` | no | `60` | Request timeout in seconds. |
| `AZURE_OPENAI_NAMER_MODEL_FAMILY` | no | inferred from deployment name | Override the namer model family detection (one of `gpt-5`, `o1`, `o3`, `o4`). Affects `max_completion_tokens` / `temperature` handling. |

Any other `AZURE_OPENAI_*` variables passed through the environment are
ignored by this tool.

---

## 4. Output files

All outputs land under `{output_dir}/YYYYMMDD_HHMMSS/`.

| File | Condition | Description |
|---|---|---|
| `report.md` | `--format md` or `both` | Human-readable clustering result. |
| `report.html` | `--format html` or `both` | Same content as `report.md`, rendered to HTML with embedded CSS. |
| `parameter_search.html` | always | Parameter-search report with a dual-axis SVG chart on top. |
| `clusters.csv` | always | One row per cluster (`cluster_id`, `cluster_name`, `size`, `summary`, `rep_1..N`). |
| `<input>_classified.csv` | always | Original CSV rows plus `cluster_id` and (when `--name-clusters`) `cluster_name`. |
| `params.json` | always | Machine-readable metadata: algorithm, params, silhouette, quality flag, sweep sample size, PCA dims, filter threshold, per-trial log. |
| `run.log` | always | INFO+ execution log (includes cache hits and LLM duplicate-resolution iterations). |

---

## 5. Exit codes

| Code | Meaning |
|---|---|
| `0` | Pipeline completed. All output files present. |
| `1` | Unhandled exception reached `main()`. `run.log` contains the traceback. |

The tool does **not** differentiate between error classes beyond this; for
scripting, parse `run.log` or `params.json` to decide downstream actions.

---

## 6. stderr output

The CLI writes progress markers and log events to stderr. Format:

```
============================================================
 voice-classifier
============================================================
[1/5] Load and normalise CSV...                <-- ephemeral line
[1/5] ✓ Load and normalise CSV (0.1s) → 8,067 rows (after dedup)
[2/5] Fetch embeddings...
[2/5] ✓ Fetch embeddings (0.1s) → shape=8,067×1536
[3/5] Search clustering candidates...
  <tqdm progress bars appear here during the sweep>
[3/5] ✓ Search clustering candidates (2.0s) → selected hdbscan ...
[4/5] ✓ Extract representative texts (0.0s) → top-5 per cluster
[5/5] ✓ Write reports (0.1s) → report.md / parameter_search.html / ...

All steps complete: data/output/20260416_012345 (total 2.3s)
```

On a TTY, the ephemeral line is overwritten with the completion line via
ANSI cursor controls. On non-TTY (pipes, CI logs), both lines are kept so
the timeline is reconstructible from the log.

Total steps are `5` without LLM labelling, `8` with it (one step each for
dataset context inference, label generation, and duplicate resolution).

---

## 7. stdout output

Currently empty. All human-facing output is on stderr; machine-readable data
is in the output files. This separation is intentional so
`pipeline.py ... > stdout.txt` stays a zero-byte file, and callers shell-pipe
on stderr safely.

---

## 8. Examples

### Minimal run

```bash
python src/pipeline.py --input data/input/tickets.csv --text-col "body"
```

### Multi-column embedding

```bash
python src/pipeline.py --input data/input/tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Skip LLM (embeddings only)

```bash
python src/pipeline.py --input data/input/tickets.csv --text-col "body" \
    --no-name-clusters
```

### HTML output only

```bash
python src/pipeline.py --input data/input/tickets.csv --text-col "body" \
    --format html
```

### Alternative chat model

```bash
python src/pipeline.py --input data/input/tickets.csv --text-col "body" \
    --name-model gpt-4o-mini
```

---

## 9. Non-stability guarantees

- Argument names and their semantics are stable across patch releases.
- Output file names, columns, and the JSON schema of `params.json` are
  **stable across minor releases**; breaking additions require a minor bump.
- Internal module APIs (function signatures inside `src/`) are **not**
  stability-guaranteed and may change in any release.
