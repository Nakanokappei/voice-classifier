"""CSVローダー — 入力CSVを読み込み、指定テキスト列を正規化する.

責務:
    - 文字コードを自動判定して `pandas.DataFrame` として読み込む
    - 指定テキスト列をNFKC正規化・空白圧縮・空行除去する
    - 重複テキストを集約し、件数を `_duplicate_count` として保持する
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 埋め込みAPIが許容する上限より手前で切り捨てる（日本語安全圏）
MAX_TEXT_LENGTH: int = 4000

# 試行するエンコーディング順。UTF-8系を優先し、Windows Excel互換にCP932を後段で試す
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932")


def load_csv(path: Path | str, text_col: str) -> pd.DataFrame:
    """CSVを読み込み、テキスト列を正規化した DataFrame を返す.

    Args:
        path: 入力CSVパス
        text_col: 分類対象テキストの列名

    Returns:
        元CSVの全列 + `_normalized_text`, `_duplicate_count` 列を付与した DataFrame

    Raises:
        FileNotFoundError: 指定パスが存在しない
        ValueError: 指定列が存在しない、または有効行が 0 件
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {path}")

    df = _read_csv_with_fallback(path)

    if text_col not in df.columns:
        available = ", ".join(df.columns.astype(str))
        raise ValueError(
            f"指定列 '{text_col}' がCSVに存在しません. 利用可能な列: {available}"
        )

    # テキスト列を文字列に統一し、正規化を適用
    df["_normalized_text"] = df[text_col].map(_normalize_text)

    # 空文字化したレコードを除外
    before = len(df)
    df = df[df["_normalized_text"].str.len() > 0].copy()
    dropped = before - len(df)
    if dropped:
        logger.info("空テキスト %d 行をドロップしました", dropped)

    if df.empty:
        raise ValueError("正規化後に有効なテキスト行がありません")

    # 重複テキストを集約（代表行は最初に現れたもの）
    df = _collapse_duplicates(df)

    # 長大テキストはトランケート
    df["_normalized_text"] = df["_normalized_text"].map(_truncate)

    logger.info("読込完了: %d 行（正規化・重複集約後）", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """エンコーディング候補を順に試してCSVを読み込む."""
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
            logger.info("encoding=%s で読込", encoding)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.debug("encoding=%s で失敗: %s", encoding, exc)

    # 候補すべて失敗
    raise UnicodeDecodeError(
        "multiple",
        b"",
        0,
        0,
        f"対応エンコーディングで読み込めませんでした: {ENCODING_CANDIDATES} (last={last_error})",
    )


def _normalize_text(value: object) -> str:
    """テキストを正規化する.

    - None / NaN → 空文字
    - NFKC 正規化
    - 改行を LF に統一し、連続空白を 1 スペースに圧縮
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    # NFKC: 全角英数・記号を半角へ正規化
    text = unicodedata.normalize("NFKC", text)

    # 改行コード統一
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 連続する空白類を 1 スペースに圧縮
    text = " ".join(text.split())

    return text


def _collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """同一 `_normalized_text` を集約し件数を `_duplicate_count` に記録."""
    counts = df.groupby("_normalized_text").size().rename("_duplicate_count")
    # 初出の行を残す（元の順序を保持）
    first_occurrence = df.drop_duplicates(subset="_normalized_text", keep="first")
    merged = first_occurrence.merge(counts, left_on="_normalized_text", right_index=True)
    return merged


def _truncate(text: str) -> str:
    """埋め込みAPI入力長の安全域に収める."""
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    logger.warning("%d 文字のテキストを %d 文字に切り詰めます", len(text), MAX_TEXT_LENGTH)
    return text[:MAX_TEXT_LENGTH]
