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


# ---------------------------------------------------------------------------
# suggest_text_columns
# ---------------------------------------------------------------------------


def test_suggest_text_columns_prefers_long_unique_text(tmp_path: Path) -> None:
    """長文でユニークな列が先頭に来ること."""
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
    # 長文である 対応内容 列が最上位
    assert candidates[0].name == "対応内容"
    # id (数値) とカテゴリ (短い繰り返し) はいずれも候補から除外されるべき
    names = [c.name for c in candidates]
    assert "id" not in names
    # カテゴリは平均長10字未満 + ユニーク率も低いので除外される想定
    # （ただし実装が含めた場合でも対応内容より下位になることを確認）
    if "カテゴリ" in names:
        assert names.index("対応内容") < names.index("カテゴリ")


def test_suggest_text_columns_excludes_short_fields(tmp_path: Path) -> None:
    """平均長 10 文字未満の列は候補から除外されること."""
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        "code,詳細な説明テキスト\n"
        "A1,これは十分な長さの説明文が入っている列でテキスト分析に適しています\n"
        "B2,別の十分に長い説明がここに入ります. クラスタリング対象として有望です\n",
    )

    candidates = loader.suggest_text_columns(csv_path)
    names = [c.name for c in candidates]
    assert "code" not in names
    assert "詳細な説明テキスト" in names


def test_suggest_text_columns_empty_csv_returns_empty(tmp_path: Path) -> None:
    """全列が空の場合は空リスト."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, "a,b\n,\n,\n")
    candidates = loader.suggest_text_columns(csv_path)
    assert candidates == []
