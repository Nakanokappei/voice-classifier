# Test Design

Test strategy, coverage targets, and test-category conventions for
voice-classifier.

---

## 1. Goals

- Every public module contract (`loader.load_csv`, `embedder.get_embeddings`,
  `tuner.find_best_clustering`, `clusterer.summarize_clusters`,
  `namer.generate_cluster_annotations`, `reporter.write_report`) is exercised
  by unit tests.
- No test makes a real network call. All OpenAI API interactions are mocked.
- Tests run in under 5 seconds locally on a modern laptop so the development
  loop stays tight.
- CI on GitHub Actions (Python 3.10 + 3.12) is the authoritative gate.

---

## 2. Test layout

```
tests/
├── conftest.py              sys.path wiring so `from src import ...` works
├── test_loader.py           CSV parsing, normalisation, multi-column, bom/crlf/rfc4180
├── test_embedder.py         OpenAI API mocked; cache hit/miss/partial, parallel ordering, retry
├── test_tuner.py            Synthetic blobs; algorithm selection; Leiden skip-guard
├── test_clusterer.py        Centroid-nearest picks; noise handling; length mismatch
├── test_namer.py            Mocked Chat API; label+summary shape, dedup, API-spec adapter
├── test_advisor.py          Mocked Chat API; digest construction, fence stripping, retry fallback
└── test_reporter.py         File shape checks; summary injection; HTML/MD/both toggle
```

Each test file is self-contained and uses only the fixtures from pytest's
standard library (`tmp_path`, `monkeypatch`, `MagicMock`).

---

## 3. Test categories

### 3.1 Unit tests

Direct function / method invocations with controlled inputs.

**Examples**

- `test_load_csv_normalizes_and_deduplicates` verifies NFKC folding and
  duplicate collapse.
- `test_build_chat_kwargs_uses_max_completion_tokens_for_gpt5` pins the
  model-family adapter logic.

**Scope**: the vast majority of the suite.

### 3.2 Integration-ish tests

Multiple modules cooperating, all external dependencies stubbed:

- `test_finds_three_clusters_on_clean_blobs` runs `tuner.find_best_clustering`
  end-to-end on a synthetic dataset.
- `test_write_report_produces_expected_files` exercises `reporter.write_report`
  with real DataFrames and ClusterSummary objects, verifying all five output
  files are present.

### 3.3 Contract / guard tests

Encode invariants that future refactors must not break:

- `test_result_order_preserved_across_parallel_batches` — parallel
  `ThreadPoolExecutor` must not reorder embeddings.
- `test_clusters_csv_is_cluster_list` — `clusters.csv` must remain
  one-row-per-cluster (not one-row-per-input-row).
- `test_noise_cluster_is_appended_at_end_without_reps` — noise handling in
  `clusterer` is load-bearing for the reporter.

---

## 4. Test data strategy

| Layer | Data |
|---|---|
| loader | Hand-written CSV fragments (UTF-8 BOM, CRLF mix, quoted newlines, CP932, duplicates, mismatches). |
| embedder | Deterministic fake client (`_CountingFakeClient`) that vectorises each input from character codes. |
| tuner | Synthetic `_synthetic_blobs()` around fixed centroids with numpy RNG seed 42. |
| clusterer | Hand-crafted ndarrays chosen so the centroid direction is unambiguous. |
| namer | Fake chat client that returns JSON derived from the first representative character. |
| reporter | Real `BestConfig`, `ClusterSummary`, and DataFrames; assertions walk the file system. |

No real customer data ever lands in `tests/`.

---

## 5. Mocking OpenAI

Every test that exercises `embedder` or `namer` patches the client factory so
no socket is ever opened:

```python
with patch.object(embedder, "_make_openai_client", return_value=fake_client):
    result = embedder.get_embeddings(...)
```

Fake clients implement only the attributes the code touches (`.embeddings.
create`, `.chat.completions.create`). `MagicMock` is used where behaviour is
throwaway; hand-written classes are used where we need thread-safe call
counting or response-shaping logic.

---

## 6. Coverage targets

- **80% line coverage** across `src/` at minimum.
- Pytest already covers every module; the `test_` file count (7) matches the
  module count (9; `utils` is exercised indirectly, `progress` has limited
  testable surface).
- Coverage is **not** enforced by CI today; it is reviewed on PR.

---

## 7. Regression / non-regression rules

- **Never delete a test to make CI green.** If behaviour legitimately changes,
  update the test and explain the reason in the commit message.
- **Every bug fix comes with a regression test** that fails on the unpatched
  code and passes with the fix.
- **Performance sensitivities** — tuner sweep, silhouette computation — are
  checked informally via `time` in manual E2E runs; we don't maintain a
  performance test suite because synthetic benchmarks poorly represent real
  embedding workloads.

---

## 8. Known gaps (accepted)

- **No end-to-end OpenAI integration test.** Running against the real API on
  every CI would cost money and be flaky; manual E2E with the sample data is
  done before each notable release.
- **No property-based tests.** Input space for a text pipeline is too large
  to get cheap value out of Hypothesis; targeted unit tests give better ROI.
- **No mutation testing.** Out of scope for a CLI tool of this size.
- **No UI / visual regression on SVG chart.** The chart rendering is
  intentionally simple (pure-Python string building) and covered by
  `test_parameter_search_separates_accepted_and_rejected`, which asserts key
  elements exist in the generated HTML.

---

## 9. Conventions

- Test files follow `test_<module>.py`; one test file per source module.
- Tests within a file are grouped by section comments; no heavy fixture
  hierarchy.
- Assertion messages describe **what** is expected, not what broke
  (`assert len(df) == 2, "BOM row should be parsed"`).
- Japanese test input strings are allowed — they are *data*, not source
  comments, and they exercise the tool's multi-language support.

---

## 10. CI workflow

See `.github/workflows/ci.yml`. The pipeline on every push / PR:

1. Checkout (actions/checkout@v6).
2. Set up Python (matrix: 3.10, 3.12).
3. `pip install -r requirements.txt` with Leiden deps filtered out (CI CPU
   does not support the AVX wheels reliably).
4. `ruff check --select E9,F63,F7,F82 src tests`.
5. `pytest -v --tb=short`.

The CI matrix intentionally excludes Leiden because the Python-igraph /
hnswlib prebuilt wheels crash on GitHub Actions' default runner CPU. Leiden
is gated behind `_LEIDEN_AVAILABLE`, so its tests auto-skip when the stack
isn't present. Full Leiden coverage happens locally.

---

## 11. Adding a new test

1. Identify the module under test; use `tests/test_<module>.py`.
2. Prefer the smallest possible input that demonstrates the invariant.
3. Use `tmp_path` for any file system interaction.
4. Mock network calls via `patch.object(<module>, "_make_openai_client", ...)`.
5. Run `pytest path/to/test_file.py::test_name -v` to verify locally.
6. Run the whole suite (`pytest`) before committing.
