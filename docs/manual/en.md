# voice-classifier — User Manual (English)

A command-line tool that classifies customer-voice CSVs (support tickets,
repair records, etc.) into clusters, labels each cluster with an LLM, and
produces both human-readable and machine-readable reports.

---

## 1. What it does

Given a CSV whose rows contain free-text customer utterances:

1. Embed each unique row with an OpenAI embedding model.
2. Sweep candidate clustering configurations (KMeans / HDBSCAN / Leiden) and
   pick the best one by cosine silhouette.
3. Pull the rows closest to each cluster's centroid as raw representatives.
4. Ask the LLM to produce a short label and a descriptive summary for each
   cluster, grounded on an inferred dataset context.
5. Deduplicate label collisions by asking the LLM to differentiate smaller
   clusters from the keeper.
6. Write Markdown, HTML, and CSV reports.

---

## 2. Installation

```bash
# Python 3.10 or later is required.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then edit .env to fill in OPENAI_API_KEY
```

Optional clustering backends (all tuner sweeps fall back gracefully when they
are missing):

- `hdbscan` — fast density-based clustering.
- `hnswlib` + `python-igraph` + `leidenalg` — graph-based Leiden clustering.

---

## 3. First run

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

The tool will:

1. Read `sample.csv`, normalising text (NFKC, whitespace collapse, dedup).
2. Fetch / cache embeddings for every unique row.
3. Search clustering configurations.
4. Extract the 5 rows closest to each centroid.
5. Unless `--no-name-clusters` is passed, infer a dataset context, run LLM
   labelling in parallel, and resolve duplicate labels.
6. Write outputs under `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Column selection

### Single column

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Multiple columns (merged as `label: value` lines)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Interactive picker

Omit both flags and the CLI prints scored candidates and prompts you to choose:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Output layout

Every run creates a timestamped directory:

```
data/output/20260416_012345/
├── report.md                           Clustering result for humans
├── report.html                          Same, rendered HTML (with --format html/both)
├── parameter_search.html                Full sweep report with chart on top
├── clusters.csv                         One row per cluster: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               Original rows + cluster_id (+ cluster_name)
├── params.json                          Machine-readable metadata
└── run.log                              INFO-level execution log
```

### clusters.csv columns

- `cluster_id` — integer, `-1` for noise.
- `cluster_name` — short LLM label (only when `--name-clusters` is on).
- `size` — number of rows.
- `summary` — LLM summary (same condition as above).
- `rep_1` ... `rep_N` — raw rows nearest the centroid.

### `<input>_classified.csv`

Your original data, plus a `cluster_id` column and, when available, a
`cluster_name` column.

---

## 6. Key options

| Option | Default | Purpose |
|---|---|---|
| `--input PATH` | required | Input CSV path |
| `--text-col NAME` | — | Single column to embed |
| `--text-cols A,B` | — | Multiple columns to concatenate |
| `--column-labels A=x,B=y` | — | Labels for multi-column prefixes |
| `--output-dir PATH` | `data/output` | Root output directory |
| `--cache-dir PATH` | `cache` | Embedding cache directory |
| `--model NAME` | `text-embedding-3-small` | OpenAI embedding model |
| `--top-k N` | `5` | Rows per cluster in `rep_*` |
| `--min-clusters N` | `2` | Lower bound for K |
| `--max-clusters N` | `20` | Upper bound for K |
| `--name-clusters` / `--no-name-clusters` | on | Toggle LLM labelling |
| `--name-model NAME` | `gpt-5.4-nano` | Chat model for labelling |
| `--format md|html|both` | `md` | Format for `report.*` |
| `--log-level LEVEL` | `INFO` | stderr log verbosity |

---

## 7. Configuration

### Environment variables (`.env`)

- `OPENAI_API_KEY` (required)
- `OPENAI_EMBEDDING_MODEL` (optional override)
- `OPENAI_REQUEST_TIMEOUT` (seconds, default 60)

### Cache

`cache/` stores embedding vectors and LLM annotations, keyed by content hash.
Switching models keeps the caches separate. To force a fresh run, delete the
relevant `cache/embeddings_*.pkl` or `cache/cluster_annotations_*.pkl` file.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Fill `.env` or export the env variable. |
| `Column '...' not found` | The CLI prints the available columns — copy one of them. |
| `Column count mismatch on lines: ...` | Your CSV has unclosed quotes or stray commas on those lines. |
| All candidates filtered by noise ratio | Dataset may lack clear clusters. A warning will appear and the filter is auto-relaxed. |
| Score is `poor` (< 0.20) | Dataset may need richer text, or try `--no-name-clusters` and review raw representatives manually. |
| `hdbscan` install fails on Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden is being skipped | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Privacy notes

- Input CSVs may contain PII. `data/input/` and `data/output/` are git-ignored.
- The pipeline sends text to OpenAI Embeddings and (optionally) Chat
  Completions. Redact/mask locally if your data is sensitive.
- The `cache/` directory stores raw embeddings plus LLM-generated labels and
  summaries. Treat it with the same care as the source CSV.
