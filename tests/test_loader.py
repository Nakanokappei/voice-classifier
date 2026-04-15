"""Tests for the loader module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import loader


def _write_csv(path: Path, content: str, encoding: str = "utf-8-sig") -> None:
    path.write_text(content, encoding=encoding)


def test_load_csv_basic(tmp_path: Path) -> None:
    """Read a UTF-8 BOM CSV; normalised column is added."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "id,対応内容\n"
        "1,返品 したい\n"
        "2, サイズ違い\n",
    )

    df = loader.load_csv(csv_path, text_col="対応内容")

    assert "_normalized_text" in df.columns
    assert "_duplicate_count" in df.columns
    assert len(df) == 2
    assert df["_normalized_text"].tolist() == ["返品 したい", "サイズ違い"]


def test_load_csv_normalizes_and_deduplicates(tmp_path: Path) -> None:
    """NFKC folds fullwidth to halfwidth; duplicates collapse."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "対応内容\n"
        "ＡＢＣ 商品\n"  # fullwidth
        "ABC 商品\n"      # halfwidth — identical after NFKC
        "配達遅延\n",
    )

    df = loader.load_csv(csv_path, text_col="対応内容")

    # "ABC 商品" collapses via fullwidth->halfwidth into 2 dupes
    row = df[df["_normalized_text"] == "ABC 商品"].iloc[0]
    assert int(row["_duplicate_count"]) == 2

    # "配達遅延" stays a single row
    assert int(df[df["_normalized_text"] == "配達遅延"]["_duplicate_count"].iloc[0]) == 1


def test_load_csv_drops_empty_rows(tmp_path: Path) -> None:
    """Empty and whitespace-only rows are dropped."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "対応内容\n"
        "\n"
        "   \n"
        "有効なテキスト\n",
    )

    df = loader.load_csv(csv_path, text_col="対応内容")
    assert len(df) == 1
    assert df["_normalized_text"].iloc[0] == "有効なテキスト"


def test_load_csv_missing_column_raises(tmp_path: Path) -> None:
    """Missing column raises ValueError."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, "other\nabc\n")

    with pytest.raises(ValueError, match="対応内容|not found"):
        loader.load_csv(csv_path, text_col="対応内容")


def test_load_csv_file_not_found(tmp_path: Path) -> None:
    """Nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load_csv(tmp_path / "missing.csv", text_col="対応内容")


def test_load_csv_truncates_long_text(tmp_path: Path) -> None:
    """Texts beyond MAX_TEXT_LENGTH are truncated."""
    csv_path = tmp_path / "input.csv"
    long_text = "あ" * (loader.MAX_TEXT_LENGTH + 500)
    _write_csv(csv_path, f"対応内容\n{long_text}\n")

    df = loader.load_csv(csv_path, text_col="対応内容")
    assert len(df["_normalized_text"].iloc[0]) == loader.MAX_TEXT_LENGTH


def test_all_empty_raises_value_error(tmp_path: Path) -> None:
    """Zero effective rows raises ValueError."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, "対応内容\n\n\n")

    with pytest.raises(ValueError):
        loader.load_csv(csv_path, text_col="対応内容")


# ---------------------------------------------------------------------------
# suggest_text_columns
# ---------------------------------------------------------------------------


def test_suggest_text_columns_prefers_long_unique_text(tmp_path: Path) -> None:
    """A long, unique column ranks first."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "id,カテゴリ,対応内容\n"
        "1,返品,商品が届きましたが色が思っていたものと違いました. 返品を希望します.\n"
        "2,配送,予定日を過ぎても届かないため状況を教えてほしいです.\n"
        "3,返品,サイズが合わなかったので返品したいです. よろしくお願いします.\n"
        "4,配送,商品が破損した状態で届きました. 交換を希望します.\n",
    )

    candidates = loader.suggest_text_columns(csv_path)

    assert candidates, "候補が見つかるべき"
    # The long column comes first
    assert candidates[0].name == "対応内容"
    # id (numeric) and category (short and repeated) should be excluded
    names = [c.name for c in candidates]
    assert "id" not in names
    # category column is dropped: avg length <10 chars and low uniqueness
    # If included anyway, it must rank below the long column
    if "カテゴリ" in names:
        assert names.index("対応内容") < names.index("カテゴリ")


def test_suggest_text_columns_includes_short_non_numeric(tmp_path: Path) -> None:
    """Short non-numeric columns (status codes, dates) are still eligible."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "status,note\n"
        "open,long note describing the ticket issue in detail\n"
        "closed,another sufficiently long note to outrank status\n"
        "open,yet another descriptive note for ranking purposes\n",
    )

    candidates = loader.suggest_text_columns(csv_path)
    names = [c.name for c in candidates]
    # Both columns are non-numeric, so both should be eligible.
    assert "status" in names
    assert "note" in names
    # The longer column still wins on score.
    assert names.index("note") < names.index("status")


def test_suggest_text_columns_excludes_purely_numeric(tmp_path: Path) -> None:
    """Columns whose values are all numeric are excluded from suggestions."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "id,amount,description\n"
        "1,100.50,A descriptive ticket body of reasonable length\n"
        "2,200,Another well-written message that is long enough\n"
        "3,300.25,Yet another text for the candidate list ranking\n",
    )

    candidates = loader.suggest_text_columns(csv_path)
    names = [c.name for c in candidates]
    assert "id" not in names
    assert "amount" not in names
    assert "description" in names


def test_mixed_numeric_column_treated_as_text(tmp_path: Path) -> None:
    """A column with any non-numeric value is treated as text-eligible."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "value\n"
        "100\n"
        "200\n"
        "N/A\n"
        "300\n",
    )

    candidates = loader.suggest_text_columns(csv_path)
    names = [c.name for c in candidates]
    # Presence of "N/A" makes the column non-numeric and therefore eligible.
    assert "value" in names


def test_suggest_text_columns_empty_csv_returns_empty(tmp_path: Path) -> None:
    """An empty CSV returns an empty candidate list."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, "a,b\n,\n,\n")
    candidates = loader.suggest_text_columns(csv_path)
    assert candidates == []


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def test_empty_file_raises_value_error(tmp_path: Path) -> None:
    """Zero-byte file raises ValueError."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        loader.load_csv(csv_path, text_col="whatever")


def test_duplicate_column_names_raises(tmp_path: Path) -> None:
    """Duplicate header names raise ValueError."""
    csv_path = tmp_path / "dup.csv"
    _write_csv(
        csv_path,
        "対応内容,対応内容\n"
        "text1,text2\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_csv(csv_path, text_col="対応内容")


def test_malformed_csv_raises_value_error(tmp_path: Path) -> None:
    """Rows with mismatching column counts raise with line numbers."""
    csv_path = tmp_path / "bad.csv"
    # Unclosed quote makes downstream rows have the wrong column count
    csv_path.write_text(
        'a,b,c\n"unclosed quote,x,y\nnext_row,p,q\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Column count mismatch"):
        loader.load_csv(csv_path, text_col="a")


def test_validation_catches_header_only(tmp_path: Path) -> None:
    """Header-only CSV raises ValueError."""
    csv_path = tmp_path / "header_only.csv"
    _write_csv(csv_path, "対応内容\n")
    with pytest.raises(ValueError):
        loader.load_csv(csv_path, text_col="対応内容")


# ---------------------------------------------------------------------------
# BOM / line endings / RFC 4180
# ---------------------------------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    """UTF-8 BOM CSV is read correctly."""
    csv_path = tmp_path / "bom.csv"
    csv_path.write_bytes(
        b"\xef\xbb\xbf\xe5\xaf\xbe\xe5\xbf\x9c\xe5\x86\x85\xe5\xae\xb9\n"
        b"\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88\n"
    )  # BOM + "対応内容\nテスト\n"

    df = loader.load_csv(csv_path, text_col="対応内容")
    assert len(df) == 1
    assert df["_normalized_text"].iloc[0] == "テスト"
    # No leftover BOM in the header
    assert "\ufeff" not in "".join(df.columns.astype(str))


def test_crlf_mixed_line_endings(tmp_path: Path) -> None:
    """Handle mixed CRLF and LF line endings."""
    csv_path = tmp_path / "mixed.csv"
    # row 1: CRLF, row 2: LF
    content = "col\r\n" + "first value\r\n" + "second value\n"
    csv_path.write_text(content, encoding="utf-8")

    df = loader.load_csv(csv_path, text_col="col")
    assert len(df) == 2
    assert df["_normalized_text"].tolist() == ["first value", "second value"]


def test_quoted_multiline_cell_is_preserved_as_single_row(tmp_path: Path) -> None:
    """RFC 4180: newlines inside quotes stay in one cell."""
    csv_path = tmp_path / "multiline.csv"
    content = 'col,other\n"line1\nline2\nline3",extra\nshort,plain\n'
    csv_path.write_text(content, encoding="utf-8")

    df = loader.load_csv(csv_path, text_col="col")
    # Two rows total; multiline quoted cell counts as one row
    assert len(df) == 2
    # Multi-line content is whitespace-collapsed into one line
    assert df["_normalized_text"].iloc[0] == "line1 line2 line3"


def test_shift_jis_encoding(tmp_path: Path) -> None:
    """Shift_JIS (CP932) エンコーディングの CSV も読めること."""
    csv_path = tmp_path / "sjis.csv"
    content = "対応内容\n返品希望\nサイズ違い\n"
    csv_path.write_bytes(content.encode("cp932"))

    df = loader.load_csv(csv_path, text_col="対応内容")
    assert len(df) == 2
    assert "返品希望" in df["_normalized_text"].tolist()


# ---------------------------------------------------------------------------
# Multi-column mode
# ---------------------------------------------------------------------------


def test_multi_column_combines_with_labels(tmp_path: Path) -> None:
    """Multiple columns are joined as label: value per row."""
    csv_path = tmp_path / "multi.csv"
    _write_csv(
        csv_path,
        "件名,本文\n"
        "返品の件,色が違いました\n"
        "配送遅延,まだ届きません\n",
    )

    df = loader.load_csv(
        csv_path,
        text_cols=["件名", "本文"],
        column_labels={"件名": "subject", "本文": "body"},
    )

    assert len(df) == 2
    texts = df["_normalized_text"].tolist()
    # Joined text contains both labels
    assert "subject: 返品の件" in texts[0]
    assert "body: 色が違いました" in texts[0]


def test_multi_column_defaults_label_to_column_name(tmp_path: Path) -> None:
    """Without column_labels, the column name is the label."""
    csv_path = tmp_path / "multi2.csv"
    _write_csv(csv_path, "A,B\nalpha,beta\n")

    df = loader.load_csv(csv_path, text_cols=["A", "B"])
    assert "A: alpha" in df["_normalized_text"].iloc[0]
    assert "B: beta" in df["_normalized_text"].iloc[0]


def test_multi_column_skips_empty_cells(tmp_path: Path) -> None:
    """In multi-column mode, empty cells are skipped."""
    csv_path = tmp_path / "partial.csv"
    _write_csv(csv_path, "subj,body\n件名のみ,\n,本文のみ\n")

    df = loader.load_csv(csv_path, text_cols=["subj", "body"])
    assert len(df) == 2
    assert df["_normalized_text"].iloc[0] == "subj: 件名のみ"
    assert df["_normalized_text"].iloc[1] == "body: 本文のみ"


def test_text_col_and_text_cols_are_exclusive(tmp_path: Path) -> None:
    """text_col and text_cols are mutually exclusive."""
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, "a,b\n1,2\n")

    with pytest.raises(ValueError, match="mutually exclusive"):
        loader.load_csv(csv_path, text_col="a", text_cols=["a", "b"])


def test_neither_text_col_nor_text_cols_raises(tmp_path: Path) -> None:
    """Specifying neither raises ValueError."""
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, "a\n1\n")

    with pytest.raises(ValueError, match="Specify either"):
        loader.load_csv(csv_path)
