"""namer モジュールのテスト — OpenAI Chat をモックして label+summary 生成を検証."""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from src import namer
from src.clusterer import ClusterSummary, NOISE_LABEL


# ---------------------------------------------------------------------------
# テストダブル
# ---------------------------------------------------------------------------


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeChatResponse:
    choices: list[_FakeChoice]


class _CountingChatClient:
    """OpenAI.chat.completions.create を模倣. label+summary の JSON を返す."""

    def __init__(self) -> None:
        self.call_count = 0
        self.requested_prompts: list[str] = []
        self._lock = threading.Lock()
        # `.chat.completions.create(...)` のネスト模倣
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **kwargs) -> _FakeChatResponse:
        with self._lock:
            self.call_count += 1
            user_msg = next(m for m in messages if m["role"] == "user")
            self.requested_prompts.append(user_msg["content"])
        # 代表テキストの先頭文字を利用してラベル・要約を決定的に生成
        user_content = messages[-1]["content"]
        first_char = next(
            (
                c for line in user_content.splitlines() if line.startswith("- ")
                for c in line[2:] if c.strip()
            ),
            "X",
        )
        payload = (
            f'{{"label": "{first_char}系の問題", '
            f'"summary": "{first_char}に関する代表的な問い合わせをまとめたクラスタです."}}'
        )
        return _FakeChatResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=payload))]
        )


def _make_summary(cid: int, texts: list[str]) -> ClusterSummary:
    return ClusterSummary(
        cluster_id=cid,
        size=len(texts),
        representative_indices=list(range(len(texts))),
        representative_texts=texts,
    )


# ---------------------------------------------------------------------------
# 基本動作
# ---------------------------------------------------------------------------


def test_generate_annotations_calls_chat_for_each_cluster(tmp_path: Path) -> None:
    """各クラスタに対して 1 回ずつ Chat API が呼ばれること."""
    summaries = [
        _make_summary(0, ["充電ができない", "充電できない"]),
        _make_summary(1, ["音が出ない", "片側聞こえない"]),
        _make_summary(NOISE_LABEL, []),
    ]

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        result = namer.generate_cluster_annotations(
            summaries, cache_dir=tmp_path / "cache", model="test-model"
        )

    assert fake_client.call_count == 2  # ノイズはスキップ
    # 各クラスタに label と summary が入っている
    assert result[0].label.endswith("問題")
    assert "に関する代表的な問い合わせ" in result[0].summary
    assert result[1].label.endswith("問題")
    # ノイズは固定文言
    assert result[NOISE_LABEL].label == "未分類"
    assert "密度の低い領域" in result[NOISE_LABEL].summary


def test_cache_hit_skips_api(tmp_path: Path) -> None:
    """同じ代表テキスト群は 2 回目以降 API を呼ばない."""
    summaries = [_make_summary(0, ["同一テキストA", "同一テキストB"])]
    cache_dir = tmp_path / "cache"

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        namer.generate_cluster_annotations(summaries, cache_dir=cache_dir, model="m")
        namer.generate_cluster_annotations(summaries, cache_dir=cache_dir, model="m")

    assert fake_client.call_count == 1


def test_generate_cluster_names_wraps_annotations(tmp_path: Path) -> None:
    """後方互換ラッパ: generate_cluster_names は label のみの dict を返す."""
    summaries = [_make_summary(0, ["テキスト"])]
    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        names = namer.generate_cluster_names(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    # str 値のみで、ClusterAnnotation を返さない
    assert isinstance(names[0], str)
    assert names[0].endswith("問題")


def test_empty_representatives_fallback(tmp_path: Path) -> None:
    """代表テキストが無いクラスタはフォールバックで、API を呼ばない."""
    summaries = [_make_summary(5, [])]
    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        result = namer.generate_cluster_annotations(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    assert result[5].label == "クラスタ #5"
    assert result[5].summary == "（代表テキストなし）"
    assert fake_client.call_count == 0


def test_parse_annotation_handles_malformed_json() -> None:
    """LLM 出力が不正 JSON でも解析失敗として返す."""
    parsed = namer._parse_annotation_json("not a json")
    assert parsed["label"] == "解析失敗"
    # 原文は summary に格納


def test_sanitize_summary_strips_quotes_and_truncates() -> None:
    """要約クリーニングの基本動作."""
    assert namer._sanitize_summary('"これは要約です"') == "これは要約です"
    # 複数行は空白で連結
    assert namer._sanitize_summary("行1\n行2\n行3") == "行1 行2 行3"
    # 超長は 400 字でトランケート
    long = "あ" * 500
    result = namer._sanitize_summary(long)
    assert result.endswith("…")
    assert len(result) == 401  # 400 + "…"


def test_failure_falls_back_to_default(tmp_path: Path) -> None:
    """API 失敗時はフォールバック値で継続."""

    class _AlwaysFailClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise RuntimeError("API dead")

    summaries = [_make_summary(7, ["text"])]
    with patch.object(namer, "_make_client", return_value=_AlwaysFailClient()):
        result = namer.generate_cluster_annotations(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    assert result[7].label == "クラスタ #7"
    assert result[7].summary == "（生成失敗）"


def test_cache_file_also_saves_json(tmp_path: Path) -> None:
    """pickle と並行して、人間可読な JSON も残ること."""
    summaries = [_make_summary(0, ["test"])]
    cache_dir = tmp_path / "cache"

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        namer.generate_cluster_annotations(summaries, cache_dir=cache_dir, model="m")

    pickle_path = cache_dir / "cluster_annotations_v2_m.pkl"
    json_path = cache_dir / "cluster_annotations_v2_m.json"
    assert pickle_path.exists()
    assert json_path.exists()

    import json
    with pickle_path.open("rb") as f:
        pkl_cache = pickle.load(f)
    with json_path.open() as f:
        json_cache = json.load(f)
    assert pkl_cache == json_cache
    # 内容は label + summary を含む
    first = next(iter(json_cache.values()))
    assert "label" in first and "summary" in first
