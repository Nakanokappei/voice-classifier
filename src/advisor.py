"""LLM-based advisory note for the parameter-search report.

After the tuner has selected a configuration and the namer has labelled the
clusters, this module asks a stronger chat model (default ``gpt-5.4``) to
interpret the result the way a senior analyst would:

    - Does the chosen configuration look sensible for the stated target?
    - How should the operator use the clusters downstream (FAQ pages,
      chatbot intents, insight reports)?
    - Which concrete caveats need attention — noise share, lopsided sizes,
      unconverged label dedup, ambiguous labels?

The returned Markdown is inserted at the top of ``parameter_search.html`` so
that anyone reading the report gets a plain-language verdict before they
scroll through the raw trial tables.

Design choices:

    * Runs only when explicitly enabled (``--advise``). One extra chat call
      per run, but output is short, so cost stays low.
    * No caching: every run has a different result summary, so caching by
      content hash would miss more than it helps.
    * Falls back silently if the API call fails — the rest of the report is
      still valid without the advisory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from openai import OpenAI

from . import utils
from .clusterer import ClusterSummary, NOISE_LABEL
from .namer import (
    ClusterAnnotation,
    DatasetContext,
    _build_chat_kwargs,
)
from .tuner import BestConfig

logger = logging.getLogger(__name__)

# Flagship chat model. GPT-5.4 handles cross-table reasoning and written
# analysis well — a noticeable step up from the nano-class model used for
# per-cluster labelling, where we only need a noun phrase.
DEFAULT_MODEL: str = "gpt-5.4"

MAX_RETRIES: int = 3
BACKOFF_BASE_SEC: float = 2.0

# Hard cap on the number of clusters we feed the advisor. Beyond this the
# prompt grows without adding signal, and GPT-5.4 costs scale linearly.
TOP_CLUSTERS_IN_PROMPT: int = 15


@dataclass
class RunDigest:
    """Compact numeric snapshot of the run, passed to the LLM as plain text.

    The advisor gets raw numbers, not prose. Keeping this as a dataclass
    lets tests construct a digest directly without reproducing the full
    BestConfig / summaries graph.
    """

    target: str
    algorithm: str
    params_text: str
    silhouette: float
    n_clusters: int
    n_noise: int
    total_rows: int
    noise_ratio_pct: float
    max_share_pct: float
    coverage_top_n: list[tuple[int, float]]  # list of (N, cumulative_pct_of_rows)
    coverage_top_n_ex_noise: list[tuple[int, float]]
    top_cluster_labels: list[tuple[str, int]]  # list of (label, size)
    dataset_domain: str
    dataset_hint: str
    dedup_converged: bool

    def as_prompt_block(self) -> str:
        """Render the digest as a human-readable block for the LLM prompt.

        Uses plain-language labels instead of data-science jargon so the
        LLM is less likely to echo technical terms back into the advisory.
        """
        lines: list[str] = []
        target_label = {
            "faq": "FAQ page creation",
            "chatbot": "chatbot question-type design",
            "insight": "exploratory analysis",
        }.get(self.target, self.target)
        lines.append(f"- Purpose of this analysis: {target_label}")
        lines.append(
            f"- Sorting method used: {self.algorithm} "
            f"(settings: {self.params_text})"
        )
        lines.append(
            f"- Groups found: {self.n_clusters}"
        )
        lines.append(
            f"- Uncategorised inquiries: {self.n_noise} "
            f"({self.noise_ratio_pct:.1f}% of {self.total_rows:,} total)"
        )
        lines.append(
            f"- Largest single group covers {self.max_share_pct:.1f}% of "
            f"all inquiries"
        )
        lines.append(
            f"- Separation quality score: {self.silhouette:.4f} "
            f"(higher is better; above 0.25 is usable, above 0.40 is strong)"
        )
        lines.append(
            f"- Business domain: {self.dataset_domain}"
            if self.dataset_domain
            else "- Business domain: unknown"
        )
        if self.dataset_hint:
            lines.append(f"- Sorting guidance used: {self.dataset_hint}")
        lines.append(
            f"- All group names are unique: "
            f"{'yes' if self.dedup_converged else 'no — some groups share similar names'}"
        )
        if self.coverage_top_n:
            cov = ", ".join(
                f"top {n} groups = {pct:.1f}%"
                for n, pct in self.coverage_top_n
            )
            lines.append(
                f"- How many inquiries are covered by the largest groups "
                f"(% of ALL inquiries): {cov}"
            )
        if self.coverage_top_n_ex_noise:
            cov2 = ", ".join(
                f"top {n} groups = {pct:.1f}%"
                for n, pct in self.coverage_top_n_ex_noise
            )
            lines.append(
                f"- How many inquiries are covered by the largest groups "
                f"(% of CATEGORISED inquiries only): {cov2}"
            )
        if self.top_cluster_labels:
            labels_block = "\n".join(
                f"    - \"{label}\" ({size:,} inquiries)"
                for label, size in self.top_cluster_labels
            )
            lines.append("- Largest groups by volume:")
            lines.append(labels_block)
        return "\n".join(lines)


def build_run_digest(
    best: BestConfig,
    summaries: list[ClusterSummary],
    cluster_annotations: dict[int, ClusterAnnotation] | None,
    total_rows: int,
    dataset_context: DatasetContext | None,
    dedup_converged: bool,
) -> RunDigest:
    """Derive a RunDigest from the raw pipeline artefacts.

    This is pure data reduction — no API calls — so it is cheap to test and
    to call even when the advisory step is disabled (e.g. for the params.json
    output, should we ever want to persist it).
    """
    annotations = cluster_annotations or {}
    non_noise = sorted(
        [s for s in summaries if s.cluster_id != NOISE_LABEL],
        key=lambda s: s.size,
        reverse=True,
    )

    # Coverage-at-rank metrics. We report both "of total rows" (which the
    # product manager cares about — noise is still mail you have to answer)
    # and "of clustered rows" (which tells you how well the taxonomy itself
    # is doing once noise is set aside).
    ranks_of_interest = [1, 5, 10, 20, 50, 100]
    clustered_total = sum(s.size for s in non_noise)

    coverage_total: list[tuple[int, float]] = []
    coverage_ex_noise: list[tuple[int, float]] = []
    running = 0
    for idx, summary in enumerate(non_noise, start=1):
        running += summary.size
        if idx in ranks_of_interest:
            if total_rows:
                coverage_total.append((idx, running / total_rows * 100.0))
            if clustered_total:
                coverage_ex_noise.append(
                    (idx, running / clustered_total * 100.0)
                )
        if idx >= max(ranks_of_interest):
            break

    # Largest-cluster share — mirrors the max_cluster_share penalty used by
    # the tuner, so the advisor and the scorer agree on the same number.
    max_share_pct = 0.0
    if non_noise and total_rows:
        max_share_pct = non_noise[0].size / total_rows * 100.0

    noise_ratio_pct = (best.n_noise / total_rows * 100.0) if total_rows else 0.0

    top_labels: list[tuple[str, int]] = []
    for summary in non_noise[:TOP_CLUSTERS_IN_PROMPT]:
        ann = annotations.get(summary.cluster_id)
        label = ann.label if ann else f"Cluster #{summary.cluster_id}"
        top_labels.append((label, summary.size))

    params_text = " ".join(
        f"{k}={v}" for k, v in best.params.items()
    ) or "(no parameters)"

    return RunDigest(
        target=best.target,
        algorithm=best.algorithm,
        params_text=params_text,
        silhouette=best.silhouette,
        n_clusters=best.n_clusters,
        n_noise=best.n_noise,
        total_rows=total_rows,
        noise_ratio_pct=noise_ratio_pct,
        max_share_pct=max_share_pct,
        coverage_top_n=coverage_total,
        coverage_top_n_ex_noise=coverage_ex_noise,
        top_cluster_labels=top_labels,
        dataset_domain=(dataset_context.domain if dataset_context else ""),
        dataset_hint=(
            dataset_context.granularity_hint if dataset_context else ""
        ),
        dedup_converged=dedup_converged,
    )


def generate_run_advice(
    digest: RunDigest,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> str:
    """Ask the LLM to write a short Markdown advisory for the report.

    Returns an empty string on failure — the caller treats "no advisory" as
    a soft condition, not an error.
    """
    client = _make_openai_client(api_key)

    system = (
        "You are summarising the results of an automated customer-voice "
        "analysis for executive leadership. Write a clear advisory note "
        "in Markdown.\n\n"
        "## Audience\n\n"
        "The readers are **non-technical senior executives** (e.g. a VP of "
        "Customer Service, a board member) who make budget and staffing "
        "decisions based on this report. They understand business numbers "
        "(percentages, row counts) but have NO background in data science "
        "or statistics. Assume they have never heard words like "
        "'silhouette', 'clustering', 'algorithm', 'centroid', 'noise', "
        "'HDBSCAN', 'KMeans', 'Leiden', 'embedding', 'vector', or "
        "'dimensionality reduction'.\n\n"
        "## Writing rules\n\n"
        "1. **Absolute ban on jargon.** Never use the terms listed above "
        "or any other data-science / machine-learning vocabulary. Replace "
        "them with plain-language equivalents:\n"
        "   - 'cluster' → 'group' or 'category'\n"
        "   - 'noise / unassigned' → 'uncategorised inquiries'\n"
        "   - 'silhouette score' → 'separation quality' or drop entirely\n"
        "   - 'algorithm' → 'automatic sorting method'\n"
        "   - 'intent' → 'question type' or 'inquiry pattern'\n"
        "   - Explain percentages by saying what they mean in practice "
        "(e.g. '22% of inquiries could not be sorted into any group — "
        "roughly 1 in 5 messages').\n"
        "2. Use short sentences. One idea per sentence.\n"
        "3. Use concrete examples: quote actual group names from the "
        "digest (in quotation marks) so readers can picture the content.\n"
        "4. When you cite numbers, immediately follow with a plain "
        "interpretation (e.g. '412 inquiries — about the same volume as "
        "one agent handles in two weeks').\n"
        "5. The tone should be that of a trusted advisor briefing the CEO "
        "before a board meeting: direct, confident, no hedging, no "
        "unnecessary caveats.\n\n"
        "## Structure\n\n"
        "- Start with a level-2 heading `## Analysis Summary`.\n"
        "- Then four level-3 subsections in this order:\n"
        "  `### Overall assessment` — one paragraph (2-3 sentences). "
        "State whether this result is ready for operational use, and in "
        "one line why or why not.\n"
        "  `### What we can do with these results` — bullet list (3-5 "
        "items). Each bullet is a concrete business action.\n"
        "  `### Points that need attention` — bullet list. Each bullet "
        "names a specific risk or gap and what to do about it.\n"
        "  `### Recommended actions` — numbered list. Each item is a "
        "specific action with a clear owner suggestion (e.g. 'the "
        "customer-service team should…').\n"
        "- Use the EXACT group counts and names from the digest.\n"
        "- Write in the same language as the dataset domain description. "
        "If the domain description is in Japanese, write the entire note "
        "in Japanese; otherwise write in English.\n"
        "- Return ONLY Markdown — no code fences wrapping the whole "
        "answer, no preamble, no trailing remarks."
    )

    user = (
        "Here is the digest of this run:\n\n"
        f"{digest.as_prompt_block()}\n\n"
        "Write the advisory note now."
    )

    def _call() -> str:
        response = client.chat.completions.create(
            **_build_chat_kwargs(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.4,
                # The note can run long when there are many caveats; cap at
                # something comfortably above the expected output length.
                max_tokens=1200,
                # We want markdown prose, not JSON.
                json_mode=False,
            )
        )
        return response.choices[0].message.content or ""

    try:
        raw = utils.call_with_exponential_backoff(
            _call,
            log_prefix="advisor",
            max_retries=MAX_RETRIES,
            base_sec=BACKOFF_BASE_SEC,
        )
    except RuntimeError as exc:
        logger.warning("Advisor call failed: %s", exc)
        return ""

    advice = _sanitise_markdown(raw)
    if not advice:
        logger.warning("Advisor returned empty content; skipping advisory section")
    return advice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_client(api_key: str | None) -> OpenAI:
    """Build the OpenAI client (duplicated from namer to keep modules independent)."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in .env or the environment."
        )
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    return OpenAI(api_key=key, timeout=timeout)


def _sanitise_markdown(raw: str) -> str:
    """Strip accidental code fences and leading / trailing whitespace.

    Some models occasionally wrap their whole response in a ```markdown
    fence. We want the inner content only.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence (possibly with a language tag) and closing fence.
        text = text.lstrip("`")
        first_newline = text.find("\n")
        if first_newline != -1 and text[:first_newline].strip().lower() in {
            "markdown", "md", ""
        }:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    return text
