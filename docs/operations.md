# Operations Guide

Deploying, running, and troubleshooting voice-classifier in practice. The
tool is a local CLI — this guide covers the workstation-operations side of
that, not server / service operations.

---

## 1. Deployment targets

| Platform | Supported | Notes |
|---|---|---|
| Windows 10/11 (work PC) | ✓ | Primary operator target. `hdbscan` may need `--only-binary=:all:`. |
| macOS (Apple Silicon & Intel) | ✓ | Everything installs from pip. |
| Linux (ubuntu-latest on GitHub Actions) | ✓ in CI | Leiden deps skipped on CI CPU; see `test-design.md`. |
| Linux (local workstation, recent CPU) | ✓ | All backends work. |

There is no "server deployment" — the tool runs on demand and exits.

---

## 2. First-time setup

```bash
# 1. Clone and install
git clone https://github.com/Nakanokappei/voice-classifier.git
cd voice-classifier
python -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure the Azure OpenAI credentials and deployment names
cp .env.example .env
# Edit .env and fill in:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_EMBEDDING_DEPLOYMENT
#   AZURE_OPENAI_NAMER_DEPLOYMENT
#   AZURE_OPENAI_ADVISOR_DEPLOYMENT

# 3. Smoke test (no API call — mocked in tests)
pytest -q

# 4. First real run
python src/pipeline.py --input data/input/sample.csv --text-col "body"
```

### 2.1 Optional backend installation

When `hdbscan` or the Leiden stack fails to install:

```bash
# HDBSCAN: use prebuilt wheel only.
pip install hdbscan --only-binary=:all:

# Leiden (all three must succeed):
pip install hnswlib python-igraph leidenalg
```

If any fails, the pipeline still runs — the missing backend is silently
skipped. You'll see a log line at startup (`hdbscan is not installed;
HDBSCAN candidates will be skipped`).

---

## 3. Routine operation

### 3.1 Typical analyst workflow

1. Place the monthly CSV in `data/input/tickets_YYYY-MM.csv`.
2. Run:
   ```bash
   python src/pipeline.py --input data/input/tickets_2026-04.csv \
       --text-cols "Subject,Body" \
       --column-labels "Subject=subject,Body=body" \
       --format both
   ```
3. Open `data/output/<timestamp>/report.html` for the human-readable view.
4. Share `clusters.csv` downstream if needed.

### 3.2 Expected runtime

With a warm embedding cache (re-running on the same data):

- 10k unique rows: **~2 seconds** for clustering + reporting.
- With `--name-clusters`: add ~1 minute for LLM calls (≤ 300 clusters, 8-wide
  concurrency).

Cold run (no cache) adds embedding-retrieval time; the embedder logs the
hit ratio at the start of the run.

---

## 4. Runbook — common issues

### 4.1 `AZURE_OPENAI_API_KEY is not set` / `AZURE_OPENAI_ENDPOINT is not set`

Confirm `.env` contains both `AZURE_OPENAI_API_KEY=...` and
`AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com`. If the tool is
launched from a different directory, `python-dotenv` looks for `.env` from
the current working directory upwards — run from the project root.

### 4.1.1 `Azure OpenAI <stage> deployment name is not set`

The pipeline expects three deployment names — embedding, namer, advisor —
each as its own `AZURE_OPENAI_*_DEPLOYMENT` env var. Either populate all
three in `.env`, or pass `--model` / `--name-model` / `--advisor-model` on
the command line. Without `AZURE_OPENAI_ADVISOR_DEPLOYMENT` the advisor
step is skipped silently.

### 4.2 `Column count mismatch on lines: 42, 57, 91`

The input CSV has rows with the wrong number of cells at those line
numbers. Usually caused by unclosed quotes or unescaped commas. Options:

1. Fix the offending lines in the source system.
2. Manually patch the CSV and retry.
3. Pre-process with a more forgiving tool (e.g. `csvkit`) and write out a
   clean copy.

### 4.3 `All candidates have noise ratio > 50%`

Every clustering candidate classified more than half of the sample as noise.
Possible causes:

- Input texts are too short or too uniform (not enough structure to cluster).
- The chosen text column is not meaningful (e.g. numeric ID column picked
  by accident).

The tool falls back to scoring-only and picks the highest silhouette
anyway. Open `parameter_search.html` to see which candidates were close and
consider adjusting your column selection.

### 4.4 `Silhouette score is low — review recommended` (poor quality flag)

Silhouette below 0.20 on the final score. The cluster structure is weak.
Typical remedies:

- Check `parameter_search.html` — if every candidate has low scores, the
  data likely lacks clear topics.
- Try a multi-column input for richer context (`--text-cols "Subject,Body"`).
- Verify the text language matches the embedding model's sweet spot.
- Inspect the raw representatives in `report.md` manually.

### 4.5 `Duplicate resolution did not converge within 3 iterations`

LLM keeps generating the same labels across several clusters. Options:

- Check `clusters.csv` for the remaining duplicates; they are still useful
  but require human review.
- Retry with a stronger / different model (`--name-model gpt-4o-mini` or
  `gpt-5.4-mini`).
- Disable LLM labelling (`--no-name-clusters`) and annotate manually.

### 4.6 `hnswlib Illegal instruction` / Leiden skipped silently

The prebuilt `hnswlib` wheel uses AVX instructions that some CPUs lack.
Reinstall from source:

```bash
pip install hnswlib --no-binary hnswlib
```

If you're on CI / a locked-down machine, accept the automatic skip — the
pipeline still runs with KMeans + HDBSCAN.

### 4.7 `RuntimeError: embeddings: exhausted 5 retries`

Azure OpenAI returned rate limits or server errors through every retry.
Options:

- Wait a few minutes and rerun — the embedding cache will preserve partial
  progress so only the failed batches are retried.
- Lower concurrency by temporarily editing `embedder.MAX_CONCURRENCY` (not a
  CLI flag today; consider exposing it if this recurs).
- Check the Azure status page and the resource quota in the Azure portal.

### 4.8 Unexpected output / report looks wrong

1. Check `run.log` (always in the output directory). It lists the actual
   algorithm chosen and the per-trial scores.
2. Inspect `parameter_search.html` — it visualises the sweep.
3. Look at `params.json` — it has the full reproducibility metadata.

---

## 5. Observability

The pipeline has no external metrics / tracing integrations. What it does
emit:

- **stderr**: phase-based progress (see `docs/cli-specification.md` §6).
- **run.log**: INFO-level events (cache hits, PCA dims, winning algorithm,
  duplicate-resolution iterations, final score).
- **params.json**: the canonical record of a run's outcome — reach for this
  when diffing two runs.

There is no remote telemetry; every signal stays on the operator's machine.

---

## 6. Capacity & cost

### 6.1 Compute

- Memory footprint scales with (N × D) for embeddings. 10k rows × 1,536 dims
  × float32 ≈ 60 MB.
- CPU: PCA + KMeans + HDBSCAN on a PCA-reduced representation is cheap
  (seconds). Leiden adds ~O(N × k) for the k-NN graph, still sub-minute.

### 6.2 Azure OpenAI spend

Ballpark (prices at time of writing; verify in the vendor console):

| Workload | Tokens | Approx cost |
|---|---|---|
| Embeddings, 10k rows of ~50 tokens each | 500k | **$0.01** at `text-embedding-3-small` |
| Cluster labelling + dedup, 200 clusters | ~400k | **$0.05** at `gpt-5.4-nano` |

A warm cache reduces every subsequent run's embedding cost to ~$0.

---

## 7. Data retention

| Artefact | Retention policy |
|---|---|
| `data/input/` | Operator's responsibility. Delete once the dataset is no longer needed. |
| `data/output/<run>/` | Keep as long as the analysis is referenced; each run is self-contained. |
| `cache/embeddings_*.pkl` | Keep for cross-run speedups. Delete when you retire a dataset or switch models. |
| `cache/cluster_annotations_*.{pkl,json}` | Same as above. |
| `run.log` | Stays alongside the output; purge with the containing run directory. |

---

## 8. Upgrading voice-classifier

```bash
git pull
pip install -r requirements.txt --upgrade
pytest -q                           # sanity check
```

Breaking changes (if any) are noted in the commit message of the release
commit. When in doubt, keep an old checkout around until the new one has
produced a result you can spot-check.

---

## 9. Escalation

For issues beyond this runbook:

1. Search `run.log` for the stack trace.
2. Gather `run.log`, `params.json`, and the redacted `report.md` (strip
   customer text before sharing).
3. Open a GitHub issue in the project repository with the above attachments
   and reproduction steps.
