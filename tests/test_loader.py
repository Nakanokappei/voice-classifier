"""loader モジュールのテスト."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import loader


def _write_csv(path: Path, content: str, encoding: str = "utf-8-sig") -> None:
    path.write_text(content, encoding=encoding)


def test_load_csv_basic(tmp_path: Path) -> None:
    """UTF-8 BOM 付きCSVを読み、正規化列が追加されること."""
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
    """NFKC で全角→半角に揃い、重複は集約されること."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "対応内容\n"
        "ＡＢＣ 商品\n"  # 全角
        "ABC 商品\n"      # 半角 — NFKC 後は同一
        "配達遅延\n",
    )

    df = loader.load_csv(csv_path, text_col="対応内容")

    # ABC 商品 は全角→半角で集約されて 2 件
    row = df[df["_normalized_text"] == "ABC 商品"].iloc[0]
    assert int(row["_duplicate_count"]) == 2

    # 配達遅延 は 1 件のまま
    assert int(df[df["_normalized_text"] == "配達遅延"]["_duplicate_count"].iloc[0]) == 1


def test_load_csv_drops_empty_rows(tmp_path: Path) -> None:
    """空文字・空白のみは除外されること."""
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
    """指定列がない場合は ValueError を投げること."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, "other\nabc\n")

    with pytest.raises(ValueError, match="対応内容"):
        loader.load_csv(csv_path, text_col="対応内容")


def test_load_csv_file_not_found(tmp_path: Path) -> None:
    """存在しないパスは FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load_csv(tmp_path / "missing.csv", text_col="対応内容")


def test_load_csv_truncates_long_text(tmp_path: Path) -> None:
    """MAX_TEXT_LENGTH を超えるテキストは切り詰められること."""
    csv_path = tmp_path / "input.csv"
    long_text = "あ" * (loader.MAX_TEXT_LENGTH + 500)
    _write_csv(csv_path, f"対応内容\n{long_text}\n")

    df = loader.load_csv(csv_path, text_col="対応内容")
    assert len(df["_normalized_text"].iloc[0]) == loader.MAX_TEXT_LENGTH


def test_all_empty_raises_value_error(tmp_path: Path) -> None:
    """有効行ゼロは ValueError."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, "対応内容\n\n\n")

    with pytest.raises(ValueError):
        loader.load_csv(csv_path, text_col="対応内容")
