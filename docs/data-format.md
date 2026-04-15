# Data Format — Input CSV Specification

## Encoding

- Preferred: **UTF-8 (with or without BOM)**.
- Windows-Excel exports in **CP932 (Shift_JIS)** are auto-detected.
- Also tried: EUC-JP, ISO-2022-JP as fallbacks.

## Requirements

- **The header row is required.**
- Pick the target text column via `--text-col` (or combine several with
  `--text-cols`).
- Text columns are treated as strings; numeric/date-looking values are
  stringified.

## Preprocessing Rules

`loader.py` applies the following and stores the result in an internal
`_normalized_text` column:

| Step | Operation |
|---|---|
| 1. Trim | Remove leading and trailing whitespace |
| 2. Unicode normalisation | NFKC (fullwidth → halfwidth, etc.) |
| 3. Line-ending unification | `\r\n`, `\r` → `\n` |
| 4. Whitespace collapse | Runs of spaces/tabs/newlines → single space |
| 5. Empty-row drop | Strings of length 0 are removed |
| 6. Deduplication | Identical texts collapse to one row; count goes to `_duplicate_count` |

> **PII assumption.** Input data can contain personally identifiable
> information. Redaction and masking are the caller's responsibility; this
> pipeline sends the text through to the OpenAI Embeddings endpoint as-is.

## Column Handling

- Every column of the input CSV is preserved in the output CSV.
- Internal columns added by the pipeline:
    - `_normalized_text` — post-normalisation text (internal use)
    - `_duplicate_count` — number of duplicates collapsed into this row
    - `cluster_id` — cluster label assigned by the pipeline (noise = `-1`)
    - `cluster_name` — LLM-generated label (present when `--name-clusters` is on)

## Example

```csv
received_at,category,response_body
2026-01-10,return,The product arrived but the colour was wrong. I'd like to return it.
2026-01-11,shipping,Past the scheduled delivery date and still nothing has arrived.
2026-01-11,return,The size didn't fit so I'd like to return it.
```

Invocation:

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

## Length Guidance

- `text-embedding-3-small` accepts up to 8,191 tokens per input.
- Japanese typically uses 1-2 tokens per character; English about 0.3-0.5.
- Keep single records under ~4,000 characters for safety.

If a record exceeds the limit, `loader.py` truncates it to the first 4,000
characters and logs a WARNING.
