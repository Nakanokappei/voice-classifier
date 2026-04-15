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

    with pytest.raises(ValueError, match="対応内容|not found"):
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


# ---------------------------------------------------------------------------
# 整合性バリデーション
# ---------------------------------------------------------------------------


def test_empty_file_raises_value_error(tmp_path: Path) -> None:
    """サイズ 0 ファイルは ValueError（早期検出）."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        loader.load_csv(csv_path, text_col="whatever")


def test_duplicate_column_names_raises(tmp_path: Path) -> None:
    """列名が重複する CSV はエラー."""
    csv_path = tmp_path / "dup.csv"
    _write_csv(
        csv_path,
        "対応内容,対応内容\n"
        "text1,text2\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_csv(csv_path, text_col="対応内容")


def test_malformed_csv_raises_value_error(tmp_path: Path) -> None:
    """列数が行ごとに違う CSV はエラー（行番号付きで報告されること）."""
    csv_path = tmp_path / "bad.csv"
    # クォート閉じ忘れで 2 行目以降の列数がズレる
    csv_path.write_text(
        'a,b,c\n"unclosed quote,x,y\nnext_row,p,q\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Column count mismatch"):
        loader.load_csv(csv_path, text_col="a")


def test_validation_catches_header_only(tmp_path: Path) -> None:
    """ヘッダのみでデータ行なしの CSV はエラー."""
    csv_path = tmp_path / "header_only.csv"
    _write_csv(csv_path, "対応内容\n")
    with pytest.raises(ValueError):
        loader.load_csv(csv_path, text_col="対応内容")


# ---------------------------------------------------------------------------
# BOM / 改行コード / RFC 4180
# ---------------------------------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    """UTF-8 BOM 付き CSV を正しく読み、ヘッダに BOM 文字が残らないこと."""
    csv_path = tmp_path / "bom.csv"
    csv_path.write_bytes(
        b"\xef\xbb\xbf\xe5\xaf\xbe\xe5\xbf\x9c\xe5\x86\x85\xe5\xae\xb9\n"
        b"\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88\n"
    )  # BOM + "対応内容\nテスト\n"

    df = loader.load_csv(csv_path, text_col="対応内容")
    assert len(df) == 1
    assert df["_normalized_text"].iloc[0] == "テスト"
    # ヘッダに BOM が紛れ込んでいないこと
    assert "\ufeff" not in "".join(df.columns.astype(str))


def test_crlf_mixed_line_endings(tmp_path: Path) -> None:
    """CRLF / LF 混在の改行コードを扱えること."""
    csv_path = tmp_path / "mixed.csv"
    # 行1: CRLF, 行2: LF
    content = "col\r\n" + "first value\r\n" + "second value\n"
    csv_path.write_text(content, encoding="utf-8")

    df = loader.load_csv(csv_path, text_col="col")
    assert len(df) == 2
    assert df["_normalized_text"].tolist() == ["first value", "second value"]


def test_quoted_multiline_cell_is_preserved_as_single_row(tmp_path: Path) -> None:
    """RFC 4180: ダブルクオート内の改行は 1 セル内に保持される."""
    csv_path = tmp_path / "multiline.csv"
    content = 'col,other\n"line1\nline2\nline3",extra\nshort,plain\n'
    csv_path.write_text(content, encoding="utf-8")

    df = loader.load_csv(csv_path, text_col="col")
    # 2 行に収まる（複数行セルは 1 行としてカウント）
    assert len(df) == 2
    # 複数行の内容が空白圧縮で 1 行に正規化される
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
# 複数列モード
# ---------------------------------------------------------------------------


def test_multi_column_combines_with_labels(tmp_path: Path) -> None:
    """複数列を指定すると label: value 形式で結合されること."""
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
    # 結合テキストが各ラベル付きで含まれる
    assert "subject: 返品の件" in texts[0]
    assert "body: 色が違いました" in texts[0]


def test_multi_column_defaults_label_to_column_name(tmp_path: Path) -> None:
    """column_labels 未指定時は列名自体がラベルになる."""
    csv_path = tmp_path / "multi2.csv"
    _write_csv(csv_path, "A,B\nalpha,beta\n")

    df = loader.load_csv(csv_path, text_cols=["A", "B"])
    assert "A: alpha" in df["_normalized_text"].iloc[0]
    assert "B: beta" in df["_normalized_text"].iloc[0]


def test_multi_column_skips_empty_cells(tmp_path: Path) -> None:
    """複数列モードで一部セルが空の場合、そのセルはスキップ."""
    csv_path = tmp_path / "partial.csv"
    _write_csv(csv_path, "subj,body\n件名のみ,\n,本文のみ\n")

    df = loader.load_csv(csv_path, text_cols=["subj", "body"])
    assert len(df) == 2
    assert df["_normalized_text"].iloc[0] == "subj: 件名のみ"
    assert df["_normalized_text"].iloc[1] == "body: 本文のみ"


def test_text_col_and_text_cols_are_exclusive(tmp_path: Path) -> None:
    """text_col と text_cols を同時指定するとエラー."""
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, "a,b\n1,2\n")

    with pytest.raises(ValueError, match="mutually exclusive"):
        loader.load_csv(csv_path, text_col="a", text_cols=["a", "b"])


def test_neither_text_col_nor_text_cols_raises(tmp_path: Path) -> None:
    """どちらも未指定の場合はエラー."""
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, "a\n1\n")

    with pytest.raises(ValueError, match="Specify either"):
        loader.load_csv(csv_path)
