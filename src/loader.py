"""CSV loader — robustly reads the input CSV and normalises the chosen text column.

Responsibilities:
    - Detect BOM (UTF-8 / UTF-16 LE / UTF-16 BE) before trying encodings.
    - Fall back through a candidate encoding list (UTF-8 → Shift_JIS → EUC-JP).
    - RFC 4180 compliant parsing (supports quoted newlines and escaped quotes).
    - Tolerate CRLF / LF / CR line-ending mixes.
    - Detect duplicate column names and column-count mismatches up front.
    - Support both single-column (`text_col`) and multi-column (`text_cols`)
      embedding modes.
"""

from __future__ import annotations

import csv
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Truncate texts just below what the embedding API can accept (safe for Japanese too).
MAX_TEXT_LENGTH: int = 4000

# Byte-order marks. Checked first so we pick the right encoding without guessing.
BOM_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# Candidate encodings to try when there is no BOM. Order matters.
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "cp932", "euc-jp", "iso-2022-jp")


def load_csv(
    path: Path | str,
    text_col: str | None = None,
    text_cols: list[str] | None = None,
    column_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a CSV and return a DataFrame with a normalised text column.

    Args:
        path: input CSV path
        text_col: column name for the single-column mode
        text_cols: list of column names for the multi-column mode
            (mutually exclusive with `text_col`)
        column_labels: optional `column name -> label` mapping used by the
            multi-column mode; defaults to the column name itself

    Returns:
        The original DataFrame plus `_normalized_text` and `_duplicate_count`
        columns, deduplicated on the normalised text.

    Raises:
        FileNotFoundError: `path` does not exist
        ValueError: invalid column spec, empty file, or structural inconsistency
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input CSV is empty: {path}")

    # Validate the column selection mode.
    if text_col is None and not text_cols:
        raise ValueError(
            "Specify either `text_col` or `text_cols`"
        )
    if text_col is not None and text_cols:
        raise ValueError("`text_col` and `text_cols` are mutually exclusive")

    # Decide encoding up front (BOM wins over fallback guesses).
    encoding = _detect_encoding(path)
    logger.info("Loading %s with encoding=%s", path.name, encoding)

    # Raw-parse the header + rows for structural checks that pandas would hide
    # (duplicate column names, column-count mismatches).
    _validate_raw_structure(path, encoding)

    df = _read_csv_rfc4180(path, encoding)

    # Final DataFrame-level sanity checks.
    _validate_csv_structure(df, path)

    # Confirm every requested column actually exists.
    requested_cols = [text_col] if text_col is not None else list(text_cols or [])
    for col in requested_cols:
        if col not in df.columns:
            available = ", ".join(df.columns.astype(str))
            raise ValueError(
                f"Column '{col}' not found in CSV. Available columns: {available}"
            )

    # Build the normalised text column.
    if text_col is not None:
        df["_normalized_text"] = df[text_col].map(_normalize_text)
    else:
        df["_normalized_text"] = _build_multi_column_text(
            df, requested_cols, column_labels or {}
        )

    # Drop rows that became empty after normalisation.
    before = len(df)
    df = df[df["_normalized_text"].str.len() > 0].copy()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d empty text rows", dropped)

    if df.empty:
        raise ValueError("No valid text rows remain after normalisation")

    # Deduplicate identical texts (count is preserved in `_duplicate_count`).
    df = _collapse_duplicates(df)

    # Truncate overly long texts to the safe length.
    df["_normalized_text"] = df["_normalized_text"].map(_truncate)

    logger.info(
        "Load complete: %d rows (after normalisation and deduplication)", len(df)
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column candidate inference (suggest_text_columns)
# ---------------------------------------------------------------------------


@dataclass
class ColumnCandidate:
    """Metadata for a "possible text column" candidate.

    Attributes:
        name: column name
        avg_length: average character length over non-empty values
        non_empty_ratio: fraction of rows that are non-empty in this column
        unique_ratio: unique-value ratio among the non-empty rows; categorical
            columns with few distinct values score low
        sample_values: up to 3 examples for human inspection
    """

    name: str
    avg_length: float
    non_empty_ratio: float
    unique_ratio: float
    sample_values: list[str]

    @property
    def score(self) -> float:
        """Likelihood of being a useful text column for clustering.

        Scoring factors:

        - Length: longer content generally carries more signal.
        - Non-empty ratio: mostly-populated columns beat sparse ones.
        - Diversity: very short tail of distinct values (unique_ratio near 0)
          is mildly penalised — but we do *not* punish categorical columns
          hard, because they're often the most informative for clustering
          (e.g. "Ticket Subject" with a handful of values per ticket type).
        - ID-ness: columns where nearly every row has a unique value
          (unique_ratio ≥ 0.9) look like identifiers / emails / names.
          They rarely help clustering and often leak PII, so we apply a
          strong penalty.
        """
        diversity = 0.5 + 0.5 * self.unique_ratio
        id_like_penalty = _id_likeness_penalty(self.unique_ratio)
        return (
            self.avg_length
            * self.non_empty_ratio
            * diversity
            * id_like_penalty
        )


def _id_likeness_penalty(unique_ratio: float) -> float:
    """Multiplier that pushes "ID-like" columns down the ranking.

    Columns where nearly every row has a different value are almost always
    identifiers, timestamps, or PII (email, full name). They are rarely the
    right target for clustering.
    """
    if unique_ratio >= 0.90:
        return 0.30
    if unique_ratio >= 0.80:
        return 0.60
    return 1.0


def suggest_text_columns(path: Path | str, top_k: int = 10) -> list[ColumnCandidate]:
    """Score every non-numeric column and return the best candidates in order.

    Filter rule:
        Columns whose every non-empty value is numeric are excluded (they
        embed poorly and users never want them). Everything else — short
        codes, dates, ids that contain letters, etc. — is eligible and
        ranked by the heuristic score (length × non-empty × uniqueness).

    If you need to embed a genuinely numeric column (a rare case), bypass
    the suggestion flow by passing ``--text-col NAME`` directly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    encoding = _detect_encoding(path)
    df = _read_csv_rfc4180(path, encoding)

    candidates: list[ColumnCandidate] = []
    for col in df.columns:
        if not str(col).strip():
            continue
        series = df[col].astype(str).map(
            lambda v: v.strip() if v != "nan" else ""
        )
        non_empty = series[series.str.len() > 0]
        if non_empty.empty:
            continue

        # Skip purely numeric columns — not text-like.
        if _is_numeric_column(non_empty):
            continue

        avg_length = float(non_empty.str.len().mean())
        non_empty_ratio = float(len(non_empty) / len(df))
        unique_ratio = float(non_empty.nunique() / len(non_empty))
        samples = non_empty.head(3).tolist()

        candidates.append(
            ColumnCandidate(
                name=str(col),
                avg_length=avg_length,
                non_empty_ratio=non_empty_ratio,
                unique_ratio=unique_ratio,
                sample_values=samples,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def _is_numeric_column(non_empty_values: pd.Series) -> bool:
    """Return True if every non-empty value parses as a number.

    We're deliberately strict: a column with a mix of numbers and strings
    (e.g. some rows say "N/A") is *not* considered numeric, because the
    user probably wants to embed the non-numeric signal.
    """
    try:
        pd.to_numeric(non_empty_values, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Encoding / BOM detection
# ---------------------------------------------------------------------------


def _detect_encoding(path: Path) -> str:
    """Pick an encoding: BOM wins; otherwise try each candidate in order."""
    with path.open("rb") as f:
        head = f.read(4)

    # BOM-based detection.
    for signature, encoding in BOM_SIGNATURES:
        if head.startswith(signature):
            return encoding

    # No BOM: try each candidate and keep the first that decodes.
    sample_bytes = _peek(path, n_bytes=65536)
    for encoding in ENCODING_CANDIDATES:
        try:
            sample_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    # Everything failed; default to UTF-8 so the caller gets a proper error later.
    logger.warning("Encoding auto-detection failed; assuming UTF-8")
    return "utf-8"


def _peek(path: Path, n_bytes: int = 65536) -> bytes:
    """Read the first `n_bytes` of the file."""
    with path.open("rb") as f:
        return f.read(n_bytes)


# ---------------------------------------------------------------------------
# RFC 4180 CSV reading
# ---------------------------------------------------------------------------


def _read_csv_rfc4180(path: Path, encoding: str) -> pd.DataFrame:
    """Read the CSV in an RFC 4180 compliant way.

    - pandas' C engine handles quoted newlines and escaped quotes by default.
    - ``keep_default_na=False`` so strings like "nan"/"null" stay as strings.
    - ``dtype=str`` keeps numeric-looking text columns from being coerced.
    """
    try:
        return pd.read_csv(
            path,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            # QUOTE_MINIMAL is the default; CRLF/LF/CR mixes are handled natively.
        )
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV syntax is broken ({path}): {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV has no header or rows ({path}): {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to decode with encoding={encoding} ({path}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def _validate_raw_structure(path: Path, encoding: str) -> None:
    """Raw-parse the header and rows to catch issues pandas would paper over.

    pandas 2.0+ silently renames duplicate columns (`col`, `col.1`), so we need
    the raw header to detect duplicates. We also verify that every data row has
    the same number of columns as the header.

    Raises:
        ValueError: on duplicates or column-count mismatches.
    """
    py_encoding = _to_python_codec(encoding)

    try:
        with path.open("r", encoding=py_encoding, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return

            # Strip any leftover BOM character from the first header cell.
            header = [_strip_bom(col) for col in header]

            # Detect duplicate column names.
            stripped = [c.strip() for c in header]
            seen: dict[str, int] = {}
            duplicates: list[str] = []
            for col in stripped:
                if not col:
                    continue
                seen[col] = seen.get(col, 0) + 1
                if seen[col] == 2:
                    duplicates.append(col)
            if duplicates:
                raise ValueError(
                    f"CSV has duplicate column names: "
                    f"{', '.join(repr(c) for c in duplicates)}. "
                    "With duplicates, the target column cannot be disambiguated."
                )

            # Detect rows with a mismatched column count.
            expected = len(header)
            bad_lines: list[int] = []
            line_no = 2
            for row in reader:
                # Fully blank rows are skipped silently by pandas — do the same here.
                if not row or all(not cell for cell in row):
                    line_no += 1
                    continue
                if len(row) != expected:
                    bad_lines.append(line_no)
                    if len(bad_lines) >= 10:
                        break
                line_no += 1

            if bad_lines:
                lines_repr = ", ".join(str(n) for n in bad_lines)
                suffix = " (and more)" if len(bad_lines) == 10 else ""
                raise ValueError(
                    f"Column count mismatch on lines: {lines_repr}{suffix}. "
                    f"Header has {expected} columns. "
                    "Likely causes: unclosed quotes, or commas inside unquoted values."
                )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to read with encoding={encoding} ({path}): {exc}"
        ) from exc


def _validate_csv_structure(df: pd.DataFrame, path: Path) -> None:
    """Final checks on the DataFrame."""
    if df.empty:
        raise ValueError(f"CSV has no data rows: {path}")

    columns = df.columns.astype(str).tolist()
    if all(not col.strip() for col in columns):
        raise ValueError(
            f"CSV has no header row ({path}). "
            "The first line could not be interpreted as column names."
        )

    empty_names = [i for i, col in enumerate(columns) if not col.strip()]
    if empty_names:
        logger.warning(
            "%d column(s) have empty names (indices=%s)",
            len(empty_names),
            empty_names,
        )


# ---------------------------------------------------------------------------
# Text construction / normalization
# ---------------------------------------------------------------------------


def _build_multi_column_text(
    df: pd.DataFrame,
    cols: list[str],
    labels: dict[str, str],
) -> pd.Series:
    """Build a `_normalized_text` series by joining multiple columns.

    For each row:
      - take each column's value; skip when empty
      - format as ``label: value``
      - join with newlines
      - apply NFKC + whitespace collapse via `_normalize_text`
    """
    def _row_to_text(row: pd.Series) -> str:
        parts: list[str] = []
        for col in cols:
            raw = row.get(col, "")
            value = "" if raw is None else str(raw).strip()
            if not value or value.lower() == "nan":
                continue
            label = labels.get(col, col)
            parts.append(f"{label}: {value}")
        combined = "\n".join(parts)
        return _normalize_text(combined)

    return df.apply(_row_to_text, axis=1)


def _normalize_text(value: object) -> str:
    """Normalise a single text value."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.split())
    return text


def _collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate identical `_normalized_text` and keep counts in `_duplicate_count`."""
    counts = df.groupby("_normalized_text").size().rename("_duplicate_count")
    first_occurrence = df.drop_duplicates(subset="_normalized_text", keep="first")
    merged = first_occurrence.merge(
        counts, left_on="_normalized_text", right_index=True
    )
    return merged


def _truncate(text: str) -> str:
    """Truncate to `MAX_TEXT_LENGTH` so the embedding API never rejects the input."""
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    logger.warning(
        "Truncating %d-character text to %d characters",
        len(text),
        MAX_TEXT_LENGTH,
    )
    return text[:MAX_TEXT_LENGTH]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _to_python_codec(encoding: str) -> str:
    """Normalise pandas-style encoding aliases to Python's built-in codec names."""
    # pandas aliases like `utf-8-sig`, `utf-16-le`, `utf-16-be` are also valid in Python.
    return encoding


def _strip_bom(text: str) -> str:
    """Remove a stray UTF-8 BOM character if it's still on the front of a string."""
    if text.startswith("\ufeff"):
        return text[1:]
    return text
