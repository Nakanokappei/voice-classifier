# Algorithm — Tuning Strategy

## Goal

Given an embedding matrix with **no prior knowledge** of the right number of
clusters, produce a reasonable partition. The user should not need to pick the
algorithm or tune parameters by hand.

Design roots:
- CKPS (`Chatbot Knowledge Preparation System`) `parameter_search.py` — sweep
  strategy on a small sample.
- LLM Topic Modeler `ClusteringService.swift` — PCA to combat the curse of
  dimensionality.

---

## Input Preprocessing

1. **L2 normalisation** — turn every row into a unit vector.
   - Cosine distance and L2 distance become equivalent, enabling the fast
     `euclidean` metric downstream.
   - `||a-b||² = 2 × (1 - cos_sim(a, b))`.

2. **PCA dimensionality reduction** (only for high-dim inputs) — counter the
   curse of dimensionality.
   - Active when the input has more than 50 dimensions; target 30.
   - Measurements on raw 1,536-d `text-embedding-3-small` show KMeans
     silhouette stuck at 0.08–0.11.
   - After PCA-30 + re-normalisation, scores recover to 0.17–0.22.
   - Constants: `PCA_TARGET_DIM = 30`, `PCA_DIM_THRESHOLD = 50`.

> **Why?** In high-dimensional spaces, most pairwise distances compress into a
> narrow band (distance concentration), so density-based methods can no longer
> distinguish dense from sparse regions. Projecting onto the top 30 principal
> components keeps the semantic variance while spreading distances back out.

---

## Two-Phase Flow

### Phase 1: Sweep on a 1,500-point subsample

Following CKPS, run every candidate on a **1,500-point subsample** and score
each with cosine silhouette. 1,500 is the ceiling where O(n²) silhouette stays
practical.

### Phase 2: Re-run the winner on the full data

Re-fit the winning `(algorithm, params)` over every row. Nearest-neighbour
label propagation from a small sample tends to mark 80%+ of the data as noise
when the sample doesn't span the full manifold, so we refit instead.

---

## Candidate Methods

| Method | Traits | Key parameters |
|---|---|---|
| **MiniBatchKMeans** | Partitions into K clusters; no noise; fast batches | `k` |
| **HDBSCAN** | Hierarchical density-based; automatic K; handles variable-density clusters | `min_cluster_size`, `min_samples=5` |
| **Leiden** | Graph-based community detection on an HNSW k-NN graph; no noise; excels on high-dim text embeddings (BERTopic-style) | `resolution`, `n_neighbors=15` |

DBSCAN is intentionally omitted: its single `eps` assumption performs poorly
on text embeddings where cluster densities vary widely (frequent topics are
dense, niche ones are sparse). HDBSCAN dominates on every realistic
text-clustering benchmark we've run, so it replaces DBSCAN entirely.

HDBSCAN requires `pip install hdbscan`; Leiden needs `hnswlib + python-igraph
+ leidenalg`. When any of these is missing, that algorithm is skipped
automatically without failing the rest of the pipeline.

### Parameter grids

```
K (KMeans):                 [2, 3, 5, 7, 10, 15, 20, 30, 50, 80]
min_cluster_size (HDBSCAN): [5, 10, 15, 20, 30, 50, 80, 100]
resolution (Leiden):        [0.3, 0.5, 0.7, 1.0, 1.3, 1.7]   # higher → more communities
```

Same discrete human-readable grids as CKPS. A log-scaled sweep adds little
signal for these metrics. The Leiden sweep builds its HNSW graph once per
invocation and reuses it across resolutions, so the full grid is cheap.

---

## Scoring

### Primary: cosine silhouette

```python
silhouette_score(sample, labels, metric="cosine")
```

- Range `[-1, 1]`; higher is better.
- Noise (`label == -1`) is excluded from the calculation.
- Fewer than 2 valid clusters → `None`, candidate dropped.

### Usability filter

A high silhouette doesn't imply usefulness: "6 tiny clusters amid 1,489 noise
points" can still score 0.89. We drop such candidates:

```
noise_ratio = n_noise / sample_size
candidates with noise_ratio > MAX_NOISE_RATIO_FOR_SELECTION (=0.5) are dropped
```

If nothing passes the filter, relax it and emit a warning.

---

## Quality Thresholds

| final silhouette | Quality flag | Report label |
|---|---|---|
| `≥ 0.40` | good | Good |
| `0.20 – 0.40` | warn | Acceptable (with caveats) |
| `< 0.20` | poor | Needs review (prominent warning banner) |

For text embeddings, scores above 0.4 on raw data are rare. 0.2-ish is already
a practically useful clustering (confirmed on real datasets).

---

## Reproducibility

- `random_state = 42` for every RNG.
- The same seed controls subsampling, PCA, and MiniBatchKMeans initialisation.
- Same input + same model ⇒ deterministic output.

---

## Real-World Example (24,693 repair intake rows → 14,084 unique)

```
Selected:      KMeans k=10
Score:         0.2272 (warn)
Clusters:      10, Noise: 0
Runtime:       2.6 s with warm embedding cache

(Rejected candidates:)
  DBSCAN eps=0.35 → sample_sil=0.865 but 97% noise → filtered out
  HDBSCAN mcs=30  → sample_sil=0.370 but 52% noise → borderline
```
