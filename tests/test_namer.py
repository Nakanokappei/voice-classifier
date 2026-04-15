"""Tests for the namer module; OpenAI Chat is mocked out."""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from src import namer
from src.clusterer import ClusterSummary, NOISE_LABEL


# ---------------------------------------------------------------------------
# Test doubles
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
    """Fake OpenAI.chat.completions.create returning label+summary JSON."""

    def __init__(self) -> None:
        self.call_count = 0
        self.requested_prompts: list[str] = []
        self._lock = threading.Lock()
        # Mirror the .chat.completions.create(...) nesting
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **kwargs) -> _FakeChatResponse:
        with self._lock:
            self.call_count += 1
            user_msg = next(m for m in messages if m["role"] == "user")
            self.requested_prompts.append(user_msg["content"])
        # Derive label/summary deterministically from the first representative character
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
    """Chat API is called exactly once per cluster."""
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

    assert fake_client.call_count == 2  # Noise is skipped
    # Each cluster has label and summary
    assert result[0].label.endswith("問題")
    assert "に関する代表的な問い合わせ" in result[0].summary
    assert result[1].label.endswith("問題")
    # Noise uses the fixed label
    assert result[NOISE_LABEL].label == "Unassigned"
    assert "low-density" in result[NOISE_LABEL].summary


def test_cache_hit_skips_api(tmp_path: Path) -> None:
    """Repeated identical rep_texts are served from cache on subsequent calls."""
    summaries = [_make_summary(0, ["同一テキストA", "同一テキストB"])]
    cache_dir = tmp_path / "cache"

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        namer.generate_cluster_annotations(summaries, cache_dir=cache_dir, model="m")
        namer.generate_cluster_annotations(summaries, cache_dir=cache_dir, model="m")

    assert fake_client.call_count == 1


def test_generate_cluster_names_wraps_annotations(tmp_path: Path) -> None:
    """Backwards-compat wrapper returns label-only dict."""
    summaries = [_make_summary(0, ["テキスト"])]
    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        names = namer.generate_cluster_names(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    # Values are plain strings, not ClusterAnnotation
    assert isinstance(names[0], str)
    assert names[0].endswith("問題")


def test_empty_representatives_fallback(tmp_path: Path) -> None:
    """Clusters with no representatives fall back without an API call."""
    summaries = [_make_summary(5, [])]
    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        result = namer.generate_cluster_annotations(
            summaries, cache_dir=tmp_path / "c", model="m"
        )
    assert result[5].label == "Cluster #5"
    assert result[5].summary == "(no representative text available)"
    assert fake_client.call_count == 0


def test_parse_annotation_handles_malformed_json() -> None:
    """Malformed LLM JSON is reported as a parse failure."""
    parsed = namer._parse_annotation_json("not a json")
    assert parsed["label"] == "Parse failure"
    # The raw text goes into summary


def test_sanitize_summary_strips_quotes_and_truncates() -> None:
    """Basic summary sanitisation behaviour."""
    assert namer._sanitize_summary('"これは要約です"') == "これは要約です"
    # Multi-line content is joined with spaces
    assert namer._sanitize_summary("行1\n行2\n行3") == "行1 行2 行3"
    # Over-long strings are truncated to 400 chars
    long = "あ" * 500
    result = namer._sanitize_summary(long)
    assert result.endswith("…")
    assert len(result) == 401  # 400 + "…"


def test_failure_falls_back_to_default(tmp_path: Path) -> None:
    """API failure yields a fallback value without crashing."""

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
    assert result[7].label == "Cluster #7"
    assert result[7].summary == "(generation failed)"


def test_cache_file_also_saves_json(tmp_path: Path) -> None:
    """Alongside pickle, a human-readable JSON copy is persisted."""
    summaries = [_make_summary(0, ["test"])]
    cache_dir = tmp_path / "cache"

    fake_client = _CountingChatClient()
    with patch.object(namer, "_make_client", return_value=fake_client):
        namer.generate_cluster_annotations(summaries, cache_dir=cache_dir, model="m")

    pickle_path = cache_dir / "cluster_annotations_v3_m.pkl"
    json_path = cache_dir / "cluster_annotations_v3_m.json"
    assert pickle_path.exists()
    assert json_path.exists()

    import json
    with pickle_path.open("rb") as f:
        pkl_cache = pickle.load(f)
    with json_path.open() as f:
        json_cache = json.load(f)
    assert pkl_cache == json_cache
    first = next(iter(json_cache.values()))
    assert "label" in first and "summary" in first


# ---------------------------------------------------------------------------
# Dataset context inference
# ---------------------------------------------------------------------------


class _ContextClient:
    """Mock for infer_dataset_context returning domain + granularity_hint."""

    def __init__(self, domain: str, hint: str) -> None:
        self.domain = domain
        self.hint = hint
        self.chat = self
        self.completions = self
        self.call_count = 0

    def create(self, **kwargs) -> _FakeChatResponse:
        self.call_count += 1
        payload = (
            f'{{"domain": "{self.domain}", "granularity_hint": "{self.hint}"}}'
        )
        return _FakeChatResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=payload))]
        )


def test_infer_dataset_context_samples_and_parses(tmp_path: Path) -> None:
    """Samples input texts and returns the LLM-inferred result."""
    texts = [
        "修理依頼: 充電できない",
        "配送遅延のクレーム",
        "サイズ違いで返品希望",
        "商品破損の報告",
        "使い方が分からない",
        "領収書の再発行依頼",
    ]
    client = _ContextClient(domain="Consumer electronics repair intake", hint="Break down by symptom")
    with patch.object(namer, "_make_client", return_value=client):
        context = namer.infer_dataset_context(texts, sample_size=3)

    assert client.call_count == 1
    assert context.domain == "Consumer electronics repair intake"
    assert context.granularity_hint == "Break down by symptom"
    assert len(context.sample_texts) == 3
    # grounding_hint embeds the domain
    hint = context.grounding_hint()
    assert "Consumer electronics repair intake" in hint and "Break down by symptom" in hint


def test_infer_dataset_context_empty_returns_fallback() -> None:
    """Empty input returns a fallback without calling the API."""
    with patch.object(namer, "_make_client") as mock_client:
        context = namer.infer_dataset_context([])
    mock_client.assert_not_called()
    assert context.domain == "unknown"


# ---------------------------------------------------------------------------
# Duplicate resolution
# ---------------------------------------------------------------------------


class _DifferentiatingClient:
    """Mock returning a unique label when given the differentiation prompt."""

    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.call_count = 0
        self._lock = threading.Lock()

    def create(self, **kwargs) -> _FakeChatResponse:
        with self._lock:
            self.call_count += 1
        # If the user message contains "Cluster A", treat it as a differentiation query
        messages = kwargs.get("messages", [])
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        # Pick the first character on the "Cluster A" side of the user message
        import re
        m = re.search(r"クラスタ A.*?-\s*(\S+)", user, re.DOTALL)
        marker = m.group(1)[:3] if m else f"X{self.call_count}"
        payload = (
            f'{{"label": "差別化された{marker}", "summary": "{marker}の詳細"}}'
        )
        return _FakeChatResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=payload))]
        )


def test_resolve_duplicates_keeps_larger_relabels_smaller() -> None:
    """On duplicates, the larger cluster keeps its label; the smaller one is regenerated."""
    summaries = [
        _make_summary(1, ["alpha1", "alpha2"]),  # 大
        _make_summary(2, ["beta1"]),             # 小
    ]
    # ClusterSummary.size is overridden here for the test
    summaries[0].size = 100
    summaries[1].size = 10

    annotations = {
        1: namer.ClusterAnnotation(label="同名ラベル", summary="Aの要約"),
        2: namer.ClusterAnnotation(label="同名ラベル", summary="Bの要約"),
    }

    client = _DifferentiatingClient()
    with patch.object(namer, "_make_client", return_value=client):
        resolved = namer.resolve_label_duplicates(
            summaries=summaries, annotations=annotations,
        )

    # The larger one (cid=1) keeps its label
    assert resolved[1].label == "同名ラベル"
    # The smaller one (cid=2) is regenerated with a differentiated label
    assert resolved[2].label.startswith("差別化された")
    # Differentiation called once (one duplicate group)
    assert client.call_count == 1


# ---------------------------------------------------------------------------
# Model-specific API adapters
# ---------------------------------------------------------------------------


def test_build_chat_kwargs_uses_max_completion_tokens_for_gpt5() -> None:
    """GPT-5 series uses max_completion_tokens."""
    kwargs = namer._build_chat_kwargs(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=200,
        temperature=0.3,
    )
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 200
    assert kwargs["temperature"] == 0.3


def test_build_chat_kwargs_uses_max_tokens_for_gpt4() -> None:
    """GPT-4o / gpt-3.5 keep the legacy max_tokens."""
    kwargs = namer._build_chat_kwargs(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=200,
        temperature=0.3,
    )
    assert "max_tokens" in kwargs
    assert "max_completion_tokens" not in kwargs


def test_build_chat_kwargs_omits_temperature_for_o_series() -> None:
    """Reasoning models (o1/o3) must not receive temperature."""
    kwargs = namer._build_chat_kwargs(
        model="o3-mini",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=100,
        temperature=0.5,
    )
    assert "temperature" not in kwargs
    # Reasoning models also use max_completion_tokens
    assert "max_completion_tokens" in kwargs


def test_build_chat_kwargs_response_format_toggle() -> None:
    """When json_mode=False, do not send response_format."""
    with_json = namer._build_chat_kwargs(
        model="gpt-4o", messages=[], max_tokens=100, json_mode=True,
    )
    without_json = namer._build_chat_kwargs(
        model="gpt-4o", messages=[], max_tokens=100, json_mode=False,
    )
    assert with_json["response_format"] == {"type": "json_object"}
    assert "response_format" not in without_json


def test_resolve_duplicates_noop_when_all_unique() -> None:
    """All unique labels → no API call."""
    summaries = [
        _make_summary(0, ["a"]),
        _make_summary(1, ["b"]),
    ]
    annotations = {
        0: namer.ClusterAnnotation(label="A", summary="A"),
        1: namer.ClusterAnnotation(label="B", summary="B"),
    }
    with patch.object(namer, "_make_client") as mock_client:
        resolved = namer.resolve_label_duplicates(
            summaries=summaries, annotations=annotations,
        )
    mock_client.assert_not_called()
    assert resolved[0].label == "A"
    assert resolved[1].label == "B"
