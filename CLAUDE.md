# voice-classifier
**Customer Voice Auto-Classification & Insight Report System**

An analysis pipeline that vectorizes customer support interaction CSVs with
Azure OpenAI Embeddings, automatically picks the optimal clustering algorithm
and parameters, and outputs per-cluster representative text and a summary
report.

---

## Architecture

```
voice-classifier/
├── CLAUDE.md               ← this file (project rules for Claude)
├── README.md
├── requirements.txt
├── .env.example            ← template for AZURE_OPENAI_* env vars (.env is gitignored)
│
├── data/
│   ├── input/              ← place customer-supplied CSVs here (gitignored)
│   └── output/             ← reports and clustering results go here (gitignored)
│
├── src/
│   ├── pipeline.py         ← entrypoint; runs every step in order
│   ├── loader.py           ← CSV reading, preprocessing, text column extraction
│   ├── embedder.py         ← Azure OpenAI Embeddings API calls and cache management
│   ├── tuner.py            ← parameter sweep and best-method auto-selection
│   ├── clusterer.py        ← run clustering, extract representative texts
│   ├── namer.py            ← LLM-based cluster labelling / summarisation
│   ├── advisor.py          ← LLM advisory note over the whole run (stronger model)
│   ├── reporter.py         ← generate reports (Markdown / HTML / CSV / JSON)
│   ├── progress.py         ← CLI progress display utilities
│   └── diagnose.py         ← support report for a customer whose run fails
│                             (stdlib only — must work before deps install)
│
├── cache/                  ← local cache for embedding vectors (gitignored)
│
├── tests/
│   └── test_*.py
│
└── docs/
    ├── architecture.md     ← detailed design and inter-module dataflow
    ├── algorithm.md        ← tuning strategy and score evaluation criteria
    └── data-format.md      ← input CSV format specification
```

---

## Tech Stack

| Purpose | Library |
|---|---|
| Embedding retrieval | `openai` SDK targeting Azure OpenAI Service (deployment-name driven) |
| Clustering | `scikit-learn` (KMeans, DBSCAN), `hdbscan` |
| Scoring | `scikit-learn` (silhouette_score, davies_bouldin_score) |
| Data processing | `pandas`, `numpy` |
| Progress display | `tqdm` |
| Environment variables | `python-dotenv` |
| Markdown → HTML | `markdown` |

Python version: **3.10+**

---

## Key Constraints

- **Runtime environment is the client's Windows workstation.** Only Python
  standard library and pip-installable packages are allowed.
- **API keys live in `.env`.** Never embed secrets in source code or output files.
- **Input CSVs may contain PII.** `data/input/` and `data/output/` are in
  `.gitignore`.
- **Embeddings must be cached.** Hit the Azure OpenAI API at most once per
  unique text; persist results under `cache/` (`.pkl` or `.npy`).
- **The only outbound network call is to Azure OpenAI.** Do not send data to
  any other cloud service.

---

## Pipeline Flow

```
Load CSV → text preprocessing → get embeddings (cache first)
    → sample parameter candidates → select the best with silhouette score
    → run clustering with the chosen config → extract representative texts
    → optional: generate LLM labels/summaries + resolve duplicates
    → emit Markdown report, HTML report, cluster list CSV, annotated rows CSV
```

See `@docs/architecture.md` for details.

---

## Coding Standards

- Function and variable names: **English snake_case**.
- Type hints are required: `def func(x: list[str]) -> pd.DataFrame:`.
- Each module has a **single responsibility**; only `pipeline.py` performs I/O
  directly.
- Never silently swallow errors — `raise` or `logging.error` them.
- Comments and docstrings: **English**. Log messages and user-facing strings
  are also English by default; LLM output language follows the input data.

---

## Output Format

Outputs are written under `data/output/YYYYMMDD_HHMMSS/`:

| File | Content |
|---|---|
| `report.md` / `report.html` | Clustering result (selected config, per-cluster reps) |
| `parameter_search.html` | Parameter-search report; top: dual-axis chart + Pareto curve + (optional) LLM advisory note explaining the result |
| `clusters.csv` | One row per cluster: `cluster_id, cluster_name, size, summary, rep_1..N` |
| `<input>_classified.csv` | Every input row annotated with `cluster_id` / `cluster_name` |
| `params.json` | Machine-readable metadata: algorithm, params, score, search meta |
| `run.log` | Execution log (INFO level, including cache hits) |

---

## Quick Start

```bash
cp .env.example .env              # fill in AZURE_OPENAI_* credentials and deployment names
pip install -r requirements.txt
python src/pipeline.py --input data/input/sample.csv --text-col "response_body"
```

---

## Docs Index

Reference these on demand with `@` — they are not auto-loaded every session:

- `@docs/architecture.md` — module design and dataflow
- `@docs/algorithm.md` — tuning strategy and algorithm selection criteria
- `@docs/data-format.md` — input CSV spec and preprocessing rules
