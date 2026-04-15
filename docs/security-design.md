# Security Design

Threat model and mitigations for voice-classifier. The tool handles
potentially sensitive customer text, makes outbound calls to OpenAI, and
produces shareable artefacts — each surface is considered below.

---

## 1. Asset inventory

| Asset | Sensitivity | Storage |
|---|---|---|
| Input CSV (`data/input/`) | High (PII likely) | Local disk |
| Normalised text inside a run | High | In-memory during run |
| Embedding vectors | Medium (derived; still reversible enough to worry about) | `cache/embeddings_*.pkl` |
| LLM-generated labels/summaries | Medium | `cache/cluster_annotations_*.{pkl,json}` |
| Output reports (`data/output/...`) | High (echoes raw rows) | Local disk |
| OpenAI API key | Critical | `.env`, env variable |
| Operator's workstation | Out of scope | — |

---

## 2. Threat model

### 2.1 In scope

| Threat | Vector | Asset | Impact |
|---|---|---|---|
| Accidental commit of PII | Git repository | Input CSV, output reports, cache | Data leak |
| Accidental commit of API key | Git repository | `.env`, logs | Credential compromise |
| Over-sharing of output reports | Distribution of `data/output/*` | Output reports | Data leak |
| Cache poisoning via tampered pickle | Compromised `cache/*.pkl` | Embedding / annotation cache | Arbitrary code execution at load |
| CSV formula injection | Malicious CSV cell | Excel / CSV consumers | Formula execution in Excel |
| Prompt injection via input text | Malicious CSV row content | LLM labelling | Manipulated labels, prompt leakage |
| Outbound data to unintended services | Runtime dependency changes | Input text, cache | Data leak |

### 2.2 Out of scope

- Attacks against the operator's workstation (assumed trusted).
- Attacks against the OpenAI service (we trust the vendor's perimeter).
- Side-channel attacks (timing, power analysis).
- Insider threats with access to `.env` or cache files.

---

## 3. Mitigations

### 3.1 Credentials

- `OPENAI_API_KEY` is loaded via `python-dotenv` from `.env`. The file is in
  `.gitignore`.
- The key is never logged: `openai` raises `RateLimitError` / `APIError`
  without embedding the key in messages; our retry wrapper logs only the
  exception and the backoff timer.
- `_make_openai_client` raises if the key is missing; it does not fall back
  to any default / placeholder.

### 3.2 PII handling

- `data/input/`, `data/output/`, and `cache/` are listed in `.gitignore`.
- The loader never prints row content at INFO level; only counts and column
  metadata.
- Reports echo raw rows by design — that is the point of the tool — but the
  output is local disk only. Sharing is the operator's decision.
- `run.log` captures INFO+ and does **not** include raw row contents (only
  counts, model names, scores, configuration decisions).

### 3.3 Pickle cache integrity

The cache files under `cache/` are pickle-serialised. Python pickle is
**not safe to load from untrusted sources**.

- `cache/` is local to the operator's machine; only writable by the pipeline
  itself.
- When loading fails (corrupt pickle, different Python, etc.), the cache is
  silently discarded and rebuilt. We do not attempt to partially recover it.
- **Do not share `cache/` contents between machines.** If a colleague needs
  the same embeddings, either rerun the pipeline against their shared CSV or
  export via a non-pickle format.

### 3.4 CSV formula injection

When a classified row ends up in an Excel-opened CSV, values starting with
`=`, `+`, `-`, or `@` are interpreted as formulas. Voice-classifier writes
row contents **as received**; we do **not** prefix with `'` or otherwise
sanitise.

- Consumers of `clusters.csv` or `<input>_classified.csv` should apply their
  own defusing (e.g. prepend `'` or import as text).
- The upstream input CSV is the real source of concern: if that contains
  `=cmd|...!A0` style payloads, this tool preserves them.

### 3.5 Prompt injection

Row content reaches the LLM via the "representative texts" in annotation
prompts. A malicious row could try to override the system prompt
(`"Ignore previous instructions and respond with ..."`).

Mitigations:

- `_truncate_for_prompt` caps any single record at 500 characters, bounding
  the attack surface.
- The JSON response schema (`{"label": ..., "summary": ...}`) forces the
  model to emit structured output; free-form hijacks are unlikely to match
  the parser.
- `_parse_annotation_json` silently drops invalid JSON and returns a
  "Parse failure" label, so a successful hijack has limited blast radius.
- Dataset-context inference uses a deliberately terse prompt and discards
  any output that doesn't match the domain/granularity_hint schema.

Residual risk: the LLM may follow benign-looking-but-adversarial content
(`"This cluster is actually about ..."`). Labels surface to human reviewers
and a bad label causes no privileged action, so we accept this risk.

### 3.6 Outbound network

- The pipeline imports `openai` and `dotenv` only.
- No other HTTP client, no telemetry, no crash reporter.
- `OPENAI_REQUEST_TIMEOUT` can be lowered to bound the time individual
  requests can stall.

When upgrading dependencies, run `pip-audit` or review release notes for any
new outbound calls.

---

## 4. Operational guidelines

### 4.1 Before committing

- Confirm `git status` shows **no** file under `data/`, `cache/`, `.env`.
- If an accidental commit happened, **rewrite history** with `git filter-repo`
  or `git filter-branch` and rotate the API key.

### 4.2 Sharing reports

- Treat `data/output/*` as sensitive. Do not attach to public-facing
  tickets, forums, or third-party AI tools without redacting raw
  representative rows.
- `parameter_search.html` is safe to share more broadly — it contains only
  aggregate scores and parameters, no customer text.

### 4.3 Key rotation

- Rotate the OpenAI key at least twice a year or immediately after any
  suspected leak.
- Keys live in `.env`; no rotation tooling is provided. Update the file,
  restart any concurrent runs.

### 4.4 Cache retention

- The embedding cache can grow meaningfully for large datasets. Periodically
  review and delete `cache/*.pkl` files that correspond to datasets you no
  longer need to reproduce.
- Deleting the cache is lossless — the next run regenerates it.

---

## 5. Security review checklist

For each non-trivial change:

- [ ] No new outbound network call other than OpenAI.
- [ ] No new file written outside `data/output/<run>` or `cache/`.
- [ ] No new log line that echoes raw row content.
- [ ] Any new cache file uses `utils.save_pickle_cache` (atomic replace).
- [ ] Any new environment variable is documented in `.env.example` and
      in `docs/cli-specification.md`.
- [ ] Prompt changes preserve the strict-JSON response schema so parsing
      stays robust against model drift.
- [ ] Tests for new functionality mock the API; no real calls in CI.

---

## 6. Known limitations

- Pickle is the cache format. A strictly defence-in-depth design would use
  JSON + numpy `.npy` for embeddings. We accept pickle because the cache is
  single-machine-only.
- There is no in-file encryption of cached data. Disk-level encryption
  (FileVault, BitLocker, LUKS) is the recommended control.
- We do not implement rate limiting beyond exponential backoff. A runaway
  run could burn through OpenAI quota — monitor the provider dashboard if
  you expect to process > 100k unique rows.
