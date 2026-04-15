"""CSVローダー — 入力CSVを堅牢に読み込み、指定テキスト列を正規化する.

責務:
    - BOM (UTF-8 / UTF-16LE / UTF-16BE) を優先判定
    - 文字コード候補を順に試行（UTF-8 系 → Shift_JIS → EUC-JP）
    - RFC 4180 準拠（クオート内改行、ダブルクオートエスケープに対応）
    - CRLF/LF/CR 混在の改行を許容
    - 列名の重複・列数不整合を早期検出して明快なエラー
    - 単一列 (`text_col`) と複数列結合 (`text_cols`) の両モード
"""

from __future__ import annotations

import csv
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 埋め込みAPIが許容する上限より手前で切り捨てる（日本語安全圏）
MAX_TEXT_LENGTH: int = 4000

# BOM シグネチャ. 先頭バイトで優先的に判定
BOM_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# BOM が無い場合に試行するエンコーディング順
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "cp932", "euc-jp", "iso-2022-jp")

# 自動推定で「テキスト列っぽい」と判定する平均長の下限
AUTO_DETECT_MIN_AVG_LENGTH: float = 10.0


def load_csv(
    path: Path | str,
    text_col: str | None = None,
    text_cols: list[str] | None = None,
    column_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """CSVを読み込み、テキスト列を正規化した DataFrame を返す.

    Args:
        path: 入力CSVパス
        text_col: 単一列モードの列名
        text_cols: 複数列モードの列名リスト（`text_col` と排他）
        column_labels: 複数列モード時の `列名 → ラベル` マップ. 省略時は列名をそのまま使用

    Returns:
        元CSVの全列 + `_normalized_text`, `_duplicate_count` 列を付与した DataFrame

    Raises:
        FileNotFoundError: 指定パスが存在しない
        ValueError: 列指定が無効、または整合性エラー
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"入力CSVが空ファイルです: {path}")

    # 列指定モードの検証
    if text_col is None and not text_cols:
        raise ValueError(
            "`text_col` または `text_cols` のいずれかを指定してください"
        )
    if text_col is not None and text_cols:
        raise ValueError("`text_col` と `text_cols` は同時に指定できません")

    # エンコーディングを先行決定（BOM 優先）
    encoding = _detect_encoding(path)
    logger.info("encoding=%s で読込 (%s)", encoding, path.name)

    # pandas が自動リネームする前の生ヘッダで重複列名を検出 + 列数整合性チェック
    _validate_raw_structure(path, encoding)

    df = _read_csv_rfc4180(path, encoding)

    # CSV 全体の基本整合性
    _validate_csv_structure(df, path)

    # 列指定を検証
    requested_cols = [text_col] if text_col is not None else list(text_cols or [])
    for col in requested_cols:
        if col not in df.columns:
            available = ", ".join(df.columns.astype(str))
            raise ValueError(
                f"指定列 '{col}' がCSVに存在しません. 利用可能な列: {available}"
            )

    # 正規化テキストを構築
    if text_col is not None:
        df["_normalized_text"] = df[text_col].map(_normalize_text)
    else:
        df["_normalized_text"] = _build_multi_column_text(
            df, requested_cols, column_labels or {}
        )

    # 空テキスト行を除外
    before = len(df)
    df = df[df["_normalized_text"].str.len() > 0].copy()
    dropped = before - len(df)
    if dropped:
        logger.info("空テキスト %d 行をドロップしました", dropped)

    if df.empty:
        raise ValueError("正規化後に有効なテキスト行がありません")

    # 重複テキストを集約
    df = _collapse_duplicates(df)

    # 長大テキストはトランケート
    df["_normalized_text"] = df["_normalized_text"].map(_truncate)

    logger.info("読込完了: %d 行（正規化・重複集約後）", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 列候補推定（suggest_text_columns）
# ---------------------------------------------------------------------------


@dataclass
class ColumnCandidate:
    """テキスト列候補の評価情報.

    Attributes:
        name: 列名
        avg_length: 平均文字数（空文字列は除外）
        non_empty_ratio: 非空の行割合
        unique_ratio: ユニーク値の割合（重複が多いと低い = カテゴリ列の可能性）
        sample_values: 判定根拠を示す最大3件のサンプル
    """

    name: str
    avg_length: float
    non_empty_ratio: float
    unique_ratio: float
    sample_values: list[str]

    @property
    def score(self) -> float:
        """テキスト列らしさのスコア.

        長文かつ非空率が高く、ユニーク率も高いほど高スコア.
        カテゴリ列（"返品"/"配送" のような少数値の繰り返し）は低スコアに落ちる.
        """
        return self.avg_length * self.non_empty_ratio * (0.5 + 0.5 * self.unique_ratio)


def suggest_text_columns(path: Path | str, top_k: int = 5) -> list[ColumnCandidate]:
    """CSVの各列を解析し、テキスト列として有望な候補を score 降順で返す."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {path}")

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

        avg_length = float(non_empty.str.len().mean())
        if avg_length < AUTO_DETECT_MIN_AVG_LENGTH:
            continue

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


# ---------------------------------------------------------------------------
# Encoding / BOM detection
# ---------------------------------------------------------------------------


def _detect_encoding(path: Path) -> str:
    """BOM を優先し、なければ候補を順に試行して適切なエンコーディングを決定."""
    with path.open("rb") as f:
        head = f.read(4)

    # BOM 優先
    for signature, encoding in BOM_SIGNATURES:
        if head.startswith(signature):
            return encoding

    # BOM なし: 候補を順に試して decode 可能か検査
    sample_bytes = _peek(path, n_bytes=65536)
    for encoding in ENCODING_CANDIDATES:
        try:
            sample_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    # 全失敗は UTF-8 にフォールバック（read 時にエラーになって呼び出し元に伝わる）
    logger.warning("エンコーディング自動判定失敗. UTF-8 とみなして読み込みます")
    return "utf-8"


def _peek(path: Path, n_bytes: int = 65536) -> bytes:
    """ファイル先頭 n バイトを読み取り."""
    with path.open("rb") as f:
        return f.read(n_bytes)


# ---------------------------------------------------------------------------
# RFC 4180 CSV reading
# ---------------------------------------------------------------------------


def _read_csv_rfc4180(path: Path, encoding: str) -> pd.DataFrame:
    """RFC 4180 準拠で CSV を読む.

    - pandas の C エンジンは RFC 4180 をほぼ満たす（クオート内改行・エスケープ対応）
    - ``keep_default_na=False`` で "nan" や "null" などを文字列として扱う
    - ``dtype=str`` で全列を文字列に統一（数値混在の text_col を保護）
    """
    try:
        return pd.read_csv(
            path,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            # quoting=csv.QUOTE_MINIMAL はデフォルト
            # pandas はデフォルトで CRLF/LF/CR 混在をハンドル（universal newlines）
        )
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSVの構文が壊れています ({path}): {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSVにヘッダも行もありません ({path}): {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"encoding={encoding} でデコードに失敗しました ({path}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def _validate_raw_structure(path: Path, encoding: str) -> None:
    """pandas の自動補正前にヘッダ・列数を生パースで確認.

    Python の `csv` モジュールで行ごとに読み、以下を検出:
    - 列名の重複（pandas 2.0+ は .1 suffix でリネームして隠す）
    - データ行の列数がヘッダと一致しない（最初の10件までを報告）

    Raises:
        ValueError: 上記のいずれかが見つかった場合
    """
    # BOM 指定時は `encoding="utf-8-sig"` が pandas/io で BOM を自動除去.
    # csv モジュールは BOM を見る前に utf-8 でデコードしないと誤判定するため、
    # 対応する Python デコーダ名に正規化してから open する.
    py_encoding = _to_python_codec(encoding)

    try:
        with path.open("r", encoding=py_encoding, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return

            # ヘッダ先頭に残った BOM 文字を除去（utf-16 等で openrb 経由で来ると残存しうる）
            header = [_strip_bom(col) for col in header]

            # 列名の重複
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
                    f"CSVの列名が重複しています: {', '.join(repr(c) for c in duplicates)}. "
                    "重複列があると対象列を一意に指定できません."
                )

            # 列数不整合（先頭10件まで記録して一括エラー化）
            expected = len(header)
            bad_lines: list[int] = []
            line_no = 2
            for row in reader:
                # 完全に空の行は pandas 側でスキップされるので無視
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
                suffix = " (ほか)" if len(bad_lines) == 10 else ""
                raise ValueError(
                    f"列数が異なる行があります (行番号: {lines_repr}{suffix}). "
                    f"ヘッダは {expected} 列です. "
                    "引用符の閉じ忘れや、カンマを含む未クオート文字列が疑われます."
                )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"encoding={encoding} で読み込めませんでした ({path}): {exc}"
        ) from exc


def _validate_csv_structure(df: pd.DataFrame, path: Path) -> None:
    """DataFrame 化後の最終チェック."""
    if df.empty:
        raise ValueError(f"CSV にデータ行がありません: {path}")

    columns = df.columns.astype(str).tolist()
    if all(not col.strip() for col in columns):
        raise ValueError(
            f"CSVにヘッダ行がないようです ({path}). "
            "1行目を列名として読み込めませんでした."
        )

    empty_names = [i for i, col in enumerate(columns) if not col.strip()]
    if empty_names:
        logger.warning(
            "列名が空の列が %d 件あります (インデックス=%s)",
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
    """複数列を `label: value` 形式で結合した正規化済みテキスト列を返す.

    各行について:
    - 列ごとに値を取り出し、空ならスキップ
    - `label: value` 形式にフォーマット
    - 改行で連結
    - 最終的に `_normalize_text` で NFKC・空白圧縮を適用
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
    """テキストを正規化する."""
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
    """同一 `_normalized_text` を集約し件数を `_duplicate_count` に記録."""
    counts = df.groupby("_normalized_text").size().rename("_duplicate_count")
    first_occurrence = df.drop_duplicates(subset="_normalized_text", keep="first")
    merged = first_occurrence.merge(
        counts, left_on="_normalized_text", right_index=True
    )
    return merged


def _truncate(text: str) -> str:
    """埋め込みAPI入力長の安全域に収める."""
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    logger.warning(
        "%d 文字のテキストを %d 文字に切り詰めます", len(text), MAX_TEXT_LENGTH
    )
    return text[:MAX_TEXT_LENGTH]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _to_python_codec(encoding: str) -> str:
    """pandas 用エイリアスを Python 組み込み codec 名に正規化."""
    # pandas の `utf-8-sig`, `utf-16-le`, `utf-16-be` は Python でも有効
    return encoding


def _strip_bom(text: str) -> str:
    """UTF-8 の BOM 文字が文字列先頭に残留していた場合に除去."""
    if text.startswith("\ufeff"):
        return text[1:]
    return text
