"""Tests for the advisor module; the OpenAI chat client is mocked out."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from src import advisor
from src.clusterer import ClusterSummary, NOISE_LABEL
from src.namer import ClusterAnnotation, DatasetContext
from src.tuner import BestConfig


# ---------------------------------------------------------------------------
# Fakes mirroring the OpenAI response shape
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


class _FakeAdvisorClient:
    """Minimal stand-in for ``openai.OpenAI``.

    Returns the canned markdown on every ``chat.completions.create`` call and
    records the prompts so tests can assert that the digest was included.
    """

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.prompts: list[list[dict]] = []
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **kwargs) -> _FakeChatResponse:
        self.prompts.append(messages)
        return _FakeChatResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=self.content))]
        )


# ---------------------------------------------------------------------------
# Helpers to build pipeline artefacts without the real pipeline
# ---------------------------------------------------------------------------


def _make_summaries(sizes: dict[int, int]) -> list[ClusterSummary]:
    """Build ClusterSummary list from ``{cluster_id: size}``.

    We only need the size field and the cluster_id to exercise the digest,
    so the other required fields are filled with deterministic stubs.
    """
    out: list[ClusterSummary] = []
    for cid, size in sizes.items():
        out.append(
            ClusterSummary(
                cluster_id=cid,
                size=size,
                representative_indices=[0],
                representative_texts=[f"sample for cluster {cid}"],
            )
        )
    return out


def _make_best(n_clusters: int, n_noise: int) -> BestConfig:
    """Construct a BestConfig with the minimal fields the digest relies on."""
    return BestConfig(
        algorithm="hdbscan",
        params={"min_cluster_size": 30, "min_samples": 5},
        silhouette=0.31,
        labels=np.zeros(1, dtype=int),
        n_clusters=n_clusters,
        n_noise=n_noise,
        all_trials=[],
        sweep_sample_size=5000,
        dim_before_pca=1536,
        dim_after_pca=30,
        target="faq",
    )


# ---------------------------------------------------------------------------
# Digest construction
# ---------------------------------------------------------------------------


def test_build_run_digest_reports_coverage_and_labels() -> None:
    """The digest records coverage at preset ranks and the top cluster labels."""
    # Four clusters sized 600/400/300/100 plus 600 noise rows -> 2000 total.
    sizes = {0: 600, 1: 400, 2: 300, 3: 100, NOISE_LABEL: 600}
    summaries = _make_summaries(sizes)
    annotations = {
        0: ClusterAnnotation(label="Login failures", summary="…"),
        1: ClusterAnnotation(label="Billing dispute", summary="…"),
        2: ClusterAnnotation(label="Shipping delay", summary="…"),
        3: ClusterAnnotation(label="Feature request", summary="…"),
        NOISE_LABEL: ClusterAnnotation(label="Unassigned", summary="…"),
    }
    best = _make_best(n_clusters=4, n_noise=600)
    total_rows = sum(sizes.values())
    ctx = DatasetContext(
        domain="SaaS support tickets", granularity_hint="by intent"
    )

    digest = advisor.build_run_digest(
        best=best,
        summaries=summaries,
        cluster_annotations=annotations,
        total_rows=total_rows,
        dataset_context=ctx,
        dedup_converged=True,
    )

    # Numeric counts match the input.
    assert digest.n_clusters == 4
    assert digest.n_noise == 600
    assert digest.total_rows == 2000

    # Noise share is 600/2000 = 30%.
    assert digest.noise_ratio_pct == 30.0
    # Max cluster share is 600/2000 = 30%.
    assert digest.max_share_pct == 30.0

    # Top-1 covers 600 of 2000 total rows (30%) and 600 of 1400 clustered rows (~42.9%).
    assert digest.coverage_top_n[0] == (1, 30.0)
    assert digest.coverage_top_n_ex_noise[0][0] == 1
    assert abs(digest.coverage_top_n_ex_noise[0][1] - 42.8571) < 0.01

    # Labels appear in size-descending order.
    label_order = [lbl for lbl, _ in digest.top_cluster_labels]
    assert label_order == [
        "Login failures",
        "Billing dispute",
        "Shipping delay",
        "Feature request",
    ]

    # Domain and hint flow through unchanged.
    assert digest.dataset_domain == "SaaS support tickets"
    assert digest.dataset_hint == "by intent"
    assert digest.dedup_converged is True


def test_run_digest_as_prompt_block_contains_key_fields() -> None:
    """The digest's textual form exposes the numbers the LLM needs."""
    summaries = _make_summaries({0: 10, 1: 5})
    best = _make_best(n_clusters=2, n_noise=0)
    digest = advisor.build_run_digest(
        best=best,
        summaries=summaries,
        cluster_annotations=None,
        total_rows=15,
        dataset_context=None,
        dedup_converged=True,
    )
    block = digest.as_prompt_block()
    # The target, algorithm, and numeric counts must appear literally so
    # the LLM can quote them back accurately.
    assert "Selected target: faq" in block
    assert "hdbscan" in block
    assert "Clusters: 2" in block


# ---------------------------------------------------------------------------
# LLM advisory generation (mocked)
# ---------------------------------------------------------------------------


def test_generate_run_advice_returns_markdown_from_mocked_client() -> None:
    """When the chat call succeeds, the returned markdown is passed through."""
    client = _FakeAdvisorClient(
        content=(
            "## Advisory\n"
            "### Verdict\n"
            "This is a sensible starting configuration.\n"
        )
    )
    summaries = _make_summaries({0: 10, 1: 5})
    best = _make_best(n_clusters=2, n_noise=0)
    digest = advisor.build_run_digest(
        best=best,
        summaries=summaries,
        cluster_annotations=None,
        total_rows=15,
        dataset_context=None,
        dedup_converged=True,
    )

    with patch("src.advisor._make_openai_client", return_value=client):
        result = advisor.generate_run_advice(digest=digest, api_key="x")

    assert result.startswith("## Advisory")
    assert "Verdict" in result

    # The digest block was embedded in the user prompt so the LLM has the
    # numbers to reason about.
    assert len(client.prompts) == 1
    user_prompt = next(
        m["content"] for m in client.prompts[0] if m["role"] == "user"
    )
    assert "Selected target: faq" in user_prompt
    assert "Clusters: 2" in user_prompt


def test_generate_run_advice_strips_accidental_code_fence() -> None:
    """Models occasionally wrap their whole output in a ```markdown fence."""
    raw = "```markdown\n## Advisory\nContent here.\n```"
    client = _FakeAdvisorClient(content=raw)

    summaries = _make_summaries({0: 10})
    best = _make_best(n_clusters=1, n_noise=0)
    digest = advisor.build_run_digest(
        best=best,
        summaries=summaries,
        cluster_annotations=None,
        total_rows=10,
        dataset_context=None,
        dedup_converged=True,
    )

    with patch("src.advisor._make_openai_client", return_value=client):
        result = advisor.generate_run_advice(digest=digest, api_key="x")

    assert not result.startswith("```")
    assert not result.rstrip().endswith("```")
    assert "## Advisory" in result


def test_generate_run_advice_returns_empty_string_on_retry_exhaustion() -> None:
    """When the OpenAI client keeps raising, generate_run_advice swallows it."""

    class _AlwaysFails:
        chat = None
        completions = None

        def __init__(self) -> None:
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise RuntimeError("429: rate limit")

    summaries = _make_summaries({0: 10})
    best = _make_best(n_clusters=1, n_noise=0)
    digest = advisor.build_run_digest(
        best=best,
        summaries=summaries,
        cluster_annotations=None,
        total_rows=10,
        dataset_context=None,
        dedup_converged=True,
    )

    with patch("src.advisor._make_openai_client", return_value=_AlwaysFails()):
        result = advisor.generate_run_advice(digest=digest, api_key="x")

    # Empty string means "no advisory section" — caller handles gracefully.
    assert result == ""
