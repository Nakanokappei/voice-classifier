# voice-classifier

[![CI](https://github.com/Nakanokappei/voice-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Nakanokappei/voice-classifier/actions/workflows/ci.yml)

An analysis pipeline that auto-classifies customer-voice CSVs (support
interactions, repair tickets, etc.) and emits per-cluster representative text
and an insight report.

## Features

- **Embeddings**: OpenAI `text-embedding-3-small` by default, with a local cache.
- **Auto-tuning**: sweeps KMeans / HDBSCAN / Leiden and picks the best config
  by cosine silhouette.
- **Representative text**: LLM summary of each cluster plus the raw
  near-centroid rows for verification.
- **Reports**: Markdown / HTML report, cluster-list CSV, annotated rows CSV,
  and machine-readable params JSON.

## Requirements

- Python 3.10+
- An OpenAI API key
- Windows / macOS / Linux

## Setup

```bash
# 1. (Optional) virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the API key
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY
```

> **Note (Windows)**: `hdbscan` sometimes needs a C compiler to build.
> If the install fails, try `pip install hdbscan --only-binary=:all:`.

## Usage

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

### Key options

| Option | Default | Description |
|---|---|---|
| `--input` | required | Input CSV path |
| `--text-col` | optional | Column to embed in single-column mode. Prompts interactively when omitted. |
| `--text-cols` | optional | Comma-separated column list. Joined as `label: value` per column and embedded. |
| `--column-labels` | optional | Comma-separated `column=label` pairs used as prefixes in multi-column mode. |
| `--output-dir` | `data/output` | Root output directory |
| `--cache-dir` | `cache` | Embedding cache directory |
| `--model` / `--embedding-model` | `text-embedding-3-small` | Embedding model. Can be swapped for `text-embedding-3-large` etc. Cache is per-model. |
| `--top-k` | `5` | Representative rows per cluster |
| `--min-clusters` | `2` | Lower bound for K (KMeans) |
| `--max-clusters` | `20` | Upper bound for K (KMeans) |
| `--target` | `faq` | Downstream use case to optimise for. `faq` prefers 30-80 clusters (good for FAQ pages), `chatbot` targets 50-150 finer intents, `insight` maximises silhouette for exploratory analysis. |
| `--name-clusters` / `--no-name-clusters` | **on** | LLM label + summary per cluster (default on). Flow: (1) infer dataset context from 5 samples → (2) grounded parallel label generation → (3) duplicate-label resolution (up to 3 passes). The summary is shown as "Representative text" in the report; raw near-centroid rows remain visible below as verification. Use `--no-name-clusters` to skip LLM calls. |
| `--name-model` / `--llm-model` | `gpt-5.4-nano` | Chat model for cluster labelling. API differences across GPT-5 / o-series / GPT-4o / GPT-3.5 (e.g. `max_completion_tokens` vs `max_tokens`) are handled automatically. |
| `--advise` / `--no-advise` | **on** | Generate an LLM advisory note summarising what the chosen configuration means for downstream use (FAQ / chatbot / insight). Inserted at the top of `parameter_search.html`. Requires `--name-clusters` so the advisor can cite real cluster labels. |
| `--advisor-model` | `gpt-5.4` | Chat model used for the advisory note. Deliberately a stronger model than `--name-model` because the advisor reasons over the whole run, not a single cluster. |
| `--format` | `md` | Report format: `md` / `html` / `both` |

### Examples

```bash
# Single column
python src/pipeline.py --input tickets.csv --text-col "response_body"

# Multi-column (CKPS-compatible)
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"

# LLM labelling + HTML report
python src/pipeline.py --input tickets.csv --text-col "response_body" \
    --name-clusters --format both
```

## Output

Per run, a directory under `data/output/YYYYMMDD_HHMMSS/` is created:

- `report.md` / `report.html` — clustering result (chosen config, per-cluster
  representative text)
- `parameter_search.html` — full search report with a dual-axis chart and
  Pareto coverage curve on top, followed (when `--advise` is enabled) by an
  LLM advisory section summarising the run in plain language
- `clusters.csv` — **cluster list** (one row per cluster): `id, name, size, summary, rep_1..N`
- `<input>_classified.csv` — original data with `cluster_id` / `cluster_name`
  appended
- `params.json` — machine-readable algorithm, parameters, score, and metadata
- `run.log` — execution log (INFO+ always captured, including cache hits)

`report` honours `--format md` (default) / `html` / `both`.
`parameter_search` is always HTML (it embeds the chart).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module design and dataflow
- [`docs/algorithm.md`](docs/algorithm.md) — tuning strategy and scoring
- [`docs/data-format.md`](docs/data-format.md) — input CSV specification

## Testing

```bash
pytest
```
