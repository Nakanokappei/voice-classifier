# Development Guide

How to set up a local development environment, make changes, and get them
merged.

---

## 1. Environment setup

```bash
git clone https://github.com/Nakanokappei/voice-classifier.git
cd voice-classifier

python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install pytest ruff                       # dev extras

cp .env.example .env                          # fill AZURE_OPENAI_* values for E2E runs
```

Python 3.10+ is required. 3.12 is the preferred dev version (matches the
upper bound of the CI matrix).

### 1.1 Optional backends

Local dev should include every optional backend so you can exercise the
whole sweep:

```bash
pip install hdbscan
pip install hnswlib python-igraph leidenalg
```

If any of these fails to build, see `docs/operations.md` §4.6.

---

## 2. Project layout (dev perspective)

```
voice-classifier/
├── src/                  Feature modules
├── tests/                Unit tests (7 files, 67 tests as of writing)
├── docs/                 Design documents + user manuals
├── data/                 Per-machine data; gitignored
├── cache/                Embedding / annotation caches; gitignored
├── .github/workflows/    CI pipeline
├── requirements.txt      Runtime deps (dev adds pytest + ruff)
├── pytest.ini            pytest config (warning filters, testpaths)
├── CLAUDE.md             Project rules Claude should follow
└── README.md             Getting-started entry point
```

The rest of this guide assumes `pwd` is the repository root.

---

## 3. Daily workflow

### 3.1 Running the tool against sample data

```bash
python src/pipeline.py \
    --input data/input/customer_support_tickets.csv \
    --text-col "Ticket Description"
```

A first run populates `cache/`. Subsequent runs on the same data finish in
seconds.

### 3.2 Running tests

```bash
pytest -q                                   # quiet
pytest -v                                   # verbose
pytest tests/test_tuner.py::test_leiden_sweep_runs_when_available -v
pytest -k "dedup"                           # keyword-match subset
```

### 3.3 Linting

```bash
ruff check --select E9,F63,F7,F82 src tests
```

Extending the ruff ruleset is welcome but do it in a dedicated PR so the
review can focus on the mechanical changes.

### 3.4 Verifying progress output looks right

```bash
python src/pipeline.py --input data/input/... --text-col "body"
```

Inspect the stderr output visually; overwriting only works on a TTY, so
piping to `tee` or `tail` will show both in-progress and completion lines
— that is the expected fallback behaviour.

---

## 4. Coding conventions

- **English only** for comments, docstrings, identifiers, log messages.
- **Function and variable names** use snake_case. Module names too.
- **Dataclasses** for structured return types (`BestConfig`,
  `ClusterSummary`, `ClusterAnnotation`, `DatasetContext`,
  `ColumnCandidate`).
- **Type hints on every public function**; `from __future__ import annotations`
  is already imported at the top of each module.
- **Single-responsibility modules**: if a new feature splits cleanly across
  existing modules, split it. If you find yourself importing 4 modules
  inside a new one, reconsider the design.
- **Shared helpers** live in `src/utils.py`. If the same pattern appears
  twice, hoist it.

---

## 5. Making changes

### 5.1 Branching

The main branch is `main`. For anything non-trivial:

```bash
git checkout -b feat/<short-name>          # or fix/, chore/, docs/
```

Small refactors and docs changes can land directly on `main` if you are
comfortable that tests pass locally.

### 5.2 Commit style

- Imperative mood in the subject (`Add Leiden sweep`, not `Added Leiden sweep`).
- One topic per commit. Unrelated fixes go in separate commits.
- Use the body to explain **why**, not what — the diff shows what.
- Include `Co-Authored-By:` for AI-assisted contributions.

### 5.3 Before pushing

```bash
pytest -q                  # green
ruff check --select E9,F63,F7,F82 src tests
```

If you added a new CLI flag:

- Update `docs/cli-specification.md`.
- Update `docs/manual/*.md` options table.
- Update README if it appears there.

If you added a new source module:

- Import it from `pipeline.py` (or wire through another appropriate module).
- Add `tests/test_<module>.py` with at least contract-level coverage.
- Update `docs/architecture.md` module map + `docs/detailed-design.md`.

---

## 6. Pull request review expectations

Checklist the reviewer will walk through:

- [ ] Tests cover the new behaviour and pass locally + CI.
- [ ] `ruff` check stays green.
- [ ] No unrelated file churn in the diff.
- [ ] Public API changes are intentional and documented.
- [ ] Error paths surface helpful messages (`ValueError`, not bare exceptions).
- [ ] No secrets, PII, or real customer data leaked into tests or diffs.
- [ ] Commit messages describe motivation, not just code delta.

Large PRs (> ~500 lines diff) should have a short description of the
high-level shape in the PR body.

---

## 7. Adding a new clustering algorithm

1. Add the optional import at the top of `src/tuner.py` with a
   `_<NAME>_AVAILABLE` flag.
2. Extend the `Algorithm` Literal.
3. Add parameter constants near the existing `HDBSCAN_*` / `LEIDEN_*` blocks.
4. Implement `_sweep_<name>(sample)` by delegating to `_run_sweep`.
5. Add a branch in `_fit_winner_on_full_data`.
6. Extend the reporter's per-method digest table to iterate over the new
   algorithm name.
7. Add tests to `tests/test_tuner.py`, skipping when the backend isn't
   installed (see `test_leiden_sweep_runs_when_available` for the pattern).
8. Document it in `docs/algorithm.md` and `docs/detailed-design.md`.

---

## 8. Adding a new CLI flag

1. Add to `parse_args()` in `src/pipeline.py` with a help string that
   mentions default and purpose.
2. If it changes a module signature, update that module's public contract
   (and unit tests).
3. Document it in:
   - `docs/cli-specification.md` §2.3 table
   - `docs/manual/*.md` options table (all 5 languages)
   - `README.md` options table (if the flag is prominent)

---

## 9. Release procedure

There is no formal release tag yet; the project is single-repo and always
runs from `main`. For a marked release:

1. Update `src/__init__.py` `__version__`.
2. Tag the commit (`git tag v0.2.0`) and push the tag.
3. Add release notes to `CHANGELOG.md` (create if missing) summarising
   user-visible changes, new options, migration notes.
4. Announce in whatever channel the users watch.

---

## 10. AI-assisted development

This project is comfortable with AI-assisted changes. A few guidelines:

- Keep generated code **reviewable**. If the AI produces 500 lines, a human
  reads 500 lines.
- The AI must not add new network calls or secret handling without explicit
  reviewer confirmation.
- Claude Code's checks apply (see `CLAUDE.md`): coding standards, English
  comments, security rules.
- Commits with significant AI help should include a `Co-Authored-By` trailer
  so provenance is visible.

---

## 11. Dependencies & tooling

Runtime deps (pinned loosely in `requirements.txt`):

- `openai` — embeddings + chat completions.
- `python-dotenv` — `.env` loading.
- `pandas`, `numpy` — tabular + vector math.
- `scikit-learn` — KMeans / PCA / silhouette.
- `hdbscan` — optional, density clustering.
- `hnswlib`, `python-igraph`, `leidenalg` — optional, Leiden stack.
- `tqdm` — progress bars.
- `markdown` — Markdown → HTML conversion.

Dev-only:

- `pytest` — test runner.
- `ruff` — linter.

Add a dep via `requirements.txt` and justify it in the commit message.

---

## 12. Troubleshooting development issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root or use `python -m pytest`. |
| `ImportError` for optional backend | The code path you're testing requires that backend; install it or skip that test with a guard. |
| Stale cache causing confusing results | `rm cache/embeddings_*.pkl cache/cluster_annotations_*.*` |
| Tests hang | Likely a real network call leaked through. Check that `_make_openai_client` is being patched. |

---

## 13. Project principles

These are the guardrails that keep the codebase coherent over time:

1. **Single responsibility per module** — reporter doesn't know about the
   API, embedder doesn't know about clusters.
2. **I/O at the edge** — only `pipeline.py` does file-system work directly.
3. **No silent failures** — log, fall back, or raise; never swallow.
4. **Reproducibility over cleverness** — fixed seeds, stable cache keys.
5. **Minimal CLI surface** — every flag earns its place and is documented
   in three places (CLI spec, manuals, README).
