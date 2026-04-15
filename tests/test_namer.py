"""namer モジュールのテスト — OpenAI Chat をモックしてキャッシュ挙動を検証."""

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
    """OpenAI.chat.completions.create を模倣するテストダブル."""

    def __init__(self) -> None:
        self.call_count = 0
        self.requested_prompts: list[str] = []
        self._lock = threading.Lock()
        # `.chat.completions.create(...)` 形式のためネスト
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **kwargs) -> _FakeChatResponse:
        with self._lock:
            self.call_count += 1
            user_msg = next(m for m in messages if m["role"] == "user")
            self.requested_prompts.append(user_msg["content"])
        # 代表テキスト内の最初の文字を使って決定的なラベルを返す
        user_content = messages[-1]["content"]
        first_char = next(
            (c for line in user_content.splitlines() if line.startswith("- ")
             for c in line[2:] if c.strip()),
            "X",
        )
        return _FakeChatResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=f"{first_char}系の問題"))]
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


def test_generate_names_calls_chat_for_each_cluster(tmp_path: Path) -> None:
    """各クラスタに対して 1 回ずつ Chat API が呼ばれること."""
    summaries = [
        _make_summary(0, ["充電ができない", "充電できない"]),
        _make_summary(1, ["音が出ない", "片側聞こえない"]),
        _make_summary(NOISE_LABEL, []),  # ノイズは呼ばない
    ]

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        result = namer.generate_cluster_names(
            summaries, cache_dir=tmp_path / "cache", model="test-model"
        )

    assert fake_client.call_count == 2  # ノイズはスキップ
    assert result[0].endswith("問題")
    assert result[1].endswith("問題")
    assert result[NOISE_LABEL] == "未分類"


def test_cache_hit_skips_api(tmp_path: Path) -> None:
    """同じ代表テキスト群は 2 回目以降 API を呼ばない."""
    summaries = [_make_summary(0, ["同一テキストA", "同一テキストB"])]
    cache_dir = tmp_path / "cache"

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        namer.generate_cluster_names(summaries, cache_dir=cache_dir, model="m")
        # 2 回目呼び出し: API を叩かないはず
        namer.generate_cluster_names(summaries, cache_dir=cache_dir, model="m")

    assert fake_client.call_count == 1


def test_empty_representatives_fallback(tmp_path: Path) -> None:
    """代表テキストが無いクラスタはフォールバックラベルで、API を呼ばない."""
    summaries = [_make_summary(5, [])]  # rep_texts=[]
    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        result = namer.generate_cluster_names(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    assert result[5] == "クラスタ #5"
    assert fake_client.call_count == 0


def test_sanitize_strips_quotes_and_truncates() -> None:
    """出力クリーニングがクォート除去と長さ制限を行うこと."""
    assert namer._sanitize_label('「充電不良」') == "充電不良"
    assert namer._sanitize_label("\"結果\"\n説明文") == "結果"
    long = "あ" * 50
    assert len(namer._sanitize_label(long)) == 30
    assert namer._sanitize_label("   ") == "無題"


def test_cache_file_also_saves_json(tmp_path: Path) -> None:
    """pickle と並行して、人間可読な JSON も残ること."""
    summaries = [_make_summary(0, ["test"])]
    cache_dir = tmp_path / "cache"

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        namer.generate_cluster_names(summaries, cache_dir=cache_dir, model="m")

    pickle_path = cache_dir / "cluster_names_m.pkl"
    json_path = cache_dir / "cluster_names_m.json"
    assert pickle_path.exists()
    assert json_path.exists()

    # 内容一致
    with pickle_path.open("rb") as f:
        pkl_cache = pickle.load(f)
    import json
    with json_path.open() as f:
        json_cache = json.load(f)
    assert pkl_cache == json_cache


def test_failure_falls_back_to_default_name(tmp_path: Path) -> None:
    """API 失敗時はフォールバックラベルで継続し、例外を漏らさない."""

    class _AlwaysFailClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise RuntimeError("API dead")

    summaries = [_make_summary(7, ["text"])]
    with patch.object(namer, "_make_client", return_value=_AlwaysFailClient()):
        # _request_with_retry の中でリトライ上限を超えるので RuntimeError が throw されるが、
        # _fetch_parallel がキャッチしてデフォルトラベルに置き換える
        result = namer.generate_cluster_names(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    assert result[7] == "クラスタ #7"
