"""LLM-based cluster annotation (label + summary + duplicate resolution).

Overall flow:

    1. **Infer dataset meaning** (`infer_dataset_context`)
       Sample a handful of representative rows from the input data and ask
       the LLM for the business domain and an appropriate labelling
       granularity. The result feeds downstream prompts as grounding context.

    2. **Parallel annotation generation** (`generate_cluster_annotations`)
       Query the LLM concurrently for every cluster, returning `label` and
       `summary` as JSON. The grounding context is embedded into the system
       prompt so the model produces labels with the right granularity for
       this particular dataset.

    3. **Duplicate resolution loop** (`resolve_label_duplicates`)
       If multiple clusters receive the same label, keep the largest one and
       regenerate the smaller ones with a differentiating prompt that shows
       both clusters' data. Repeat until all labels are unique (max 3 passes).

Caching notes:
    Only the initial generation is cached. Differentiated labels are ephemeral
    because their "correct" output depends on the global label set at the time
    of generation (which changes between runs).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from . import utils
from .clusterer import ClusterSummary, NOISE_LABEL

logger = logging.getLogger(__name__)

# OpenAI's recommendation: GPT-5.4 nano targets classification / data
# extraction / ranking / sub-agent workloads where speed and cost matter.
# Cluster labelling matches that profile exactly.
DEFAULT_MODEL: str = "gpt-5.4-nano"

MAX_CONCURRENCY: int = 8
MAX_RETRIES: int = 4
BACKOFF_BASE_SEC: float = 2.0

# Upper bound on dedup iterations. Three passes catch the vast majority of cases.
MAX_DEDUP_ITERATIONS: int = 3

# Number of rows sampled for the dataset-meaning inference step.
DATASET_SAMPLE_SIZE: int = 5


# ---------------------------------------------------------------------------
# Model-specific Chat Completions kwargs
# ---------------------------------------------------------------------------
#
# OpenAI chat models differ in which parameters they accept:
#   - GPT-5 series, o-series: require `max_completion_tokens`; `max_tokens` 400s
#   - o-series (o1, o3): temperature is fixed; sending one is an error
#   - GPT-4o / GPT-4 / GPT-3.5: legacy `max_tokens` works as before
#
# Rather than patching every call site when a new series ships, we centralise
# the logic in `_build_chat_kwargs` and dispatch by model-name prefix.


def _uses_max_completion_tokens(model: str) -> bool:
    """True when the model uses the new `max_completion_tokens` parameter name."""
    prefixes = ("gpt-5", "o1", "o3", "o4")
    return any(model.startswith(p) for p in prefixes)


def _supports_custom_temperature(model: str) -> bool:
    """True when the model accepts an explicit temperature value."""
    # The o-series uses a fixed temperature; everything else is free.
    reasoning_prefixes = ("o1", "o3", "o4")
    return not any(model.startswith(p) for p in reasoning_prefixes)


def _supports_json_mode(model: str) -> bool:
    """Return True if the model supports ``response_format={"type": "json_object"}``."""
    # GPT-4o, GPT-5, o-series, gpt-4-turbo all support it. GPT-3.5 needs 0125+.
    return True


def _build_chat_kwargs(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.3,
    json_mode: bool = True,
) -> dict:
    """Assemble kwargs for ``ChatCompletions.create`` while absorbing model quirks."""
    kwargs: dict = {"model": model, "messages": messages}

    # Token limit parameter name differs across series.
    if _uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # o-series reasoning models reject `temperature`.
    if _supports_custom_temperature(model):
        kwargs["temperature"] = temperature

    if json_mode and _supports_json_mode(model):
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ClusterAnnotation:
    """LLM-generated annotation for a single cluster.

    Attributes:
        label: short label (10-20 chars), used as ``cluster_name``.
        summary: one- to three-sentence description, shown as the main
            "representative text" in reports.
    """

    label: str
    summary: str


@dataclass
class DatasetContext:
    """Result of the dataset-meaning inference step.

    Attributes:
        domain: business domain (e.g. ``"Consumer electronics repair intake"``).
        granularity_hint: how specific the cluster labels should be
            (e.g. ``"Break down by symptom type"``).
        sample_texts: samples used to derive the context (kept for audit/logs).
    """

    domain: str
    granularity_hint: str
    sample_texts: list[str] = field(default_factory=list)

    def grounding_hint(self) -> str:
        """Render the instruction that gets embedded in downstream prompts."""
        return (
            f"This dataset is '{self.domain}'. "
            f"Labelling guidance: {self.granularity_hint}. "
            "Avoid overly abstract generic terms (e.g. a bare 'issue' or "
            "'problem'); pick concrete noun phrases that fit this dataset's "
            "topic space."
        )


# ---------------------------------------------------------------------------
# Step 1: Dataset context inference
# ---------------------------------------------------------------------------


def infer_dataset_context(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    sample_size: int = DATASET_SAMPLE_SIZE,
    seed: int = 42,
    api_key: str | None = None,
) -> DatasetContext:
    """Sample a few rows and ask the LLM to describe the dataset."""
    if not texts:
        logger.warning("Texts are empty; skipping dataset context inference")
        return DatasetContext(
            domain="unknown",
            granularity_hint="label with concrete domain terms",
        )

    # Dedupe first so we don't over-sample repeated phrasings.
    unique_texts = list(dict.fromkeys(texts))
    rng = _numpy_rng(seed)
    n = min(sample_size, len(unique_texts))
    indices = rng.choice(len(unique_texts), size=n, replace=False)
    samples = [unique_texts[int(i)] for i in indices]

    client = _make_openai_client(api_key)
    bullets = "\n".join(f"- {_truncate_for_prompt(t)}" for t in samples)
    system = (
        "You analyse datasets to extract their business context. "
        "From a small sample of records, return the domain and a labelling "
        "granularity hint as JSON.\n\n"
        "Required schema:\n"
        "{\n"
        '  "domain": "<business domain. Examples: '
        "'Consumer electronics repair intake', "
        "'SaaS customer support tickets'. Be concrete.>\",\n"
        '  "granularity_hint": "<guidance on how specific cluster labels should '
        "be. Examples: 'Break down by symptom type', "
        "'Categorise by inquiry intent'>\"\n"
        "}\n\n"
        "Return ONLY valid JSON — no surrounding prose or code fences."
    )
    user = f"Here are {n} randomly sampled records from the dataset:\n{bullets}"

    def _call_infer() -> str:
        response = client.chat.completions.create(
            **_build_chat_kwargs(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=300,
            )
        )
        return response.choices[0].message.content or "{}"

    try:
        raw = utils.call_with_exponential_backoff(
            _call_infer,
            log_prefix="dataset context",
            max_retries=MAX_RETRIES,
            base_sec=BACKOFF_BASE_SEC,
        )
    except RuntimeError:
        logger.warning("Dataset context inference exhausted retries; using fallback")
        return DatasetContext(
            domain="unknown",
            granularity_hint="label with concrete domain terms",
            sample_texts=samples,
        )

    payload = _safe_json_loads(raw) or {}
    domain = str(payload.get("domain", "")).strip() or "unknown"
    hint = (
        str(payload.get("granularity_hint", "")).strip()
        or "label with concrete domain terms"
    )
    logger.info("Dataset context: domain=%s hint=%s", domain, hint)
    return DatasetContext(
        domain=domain, granularity_hint=hint, sample_texts=samples
    )


# ---------------------------------------------------------------------------
# Step 2: Parallel annotation generation
# ---------------------------------------------------------------------------


def generate_cluster_annotations(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    dataset_context: DatasetContext | None = None,
) -> dict[int, ClusterAnnotation]:
    """Generate a ``label`` and a ``summary`` for every cluster."""
    if not summaries:
        return {}

    cache_path = _cache_path_for(Path(cache_dir), model)
    cache: dict[str, dict[str, str]] = utils.load_pickle_cache(cache_path)

    # Cache key includes the grounding hint as salt: the same rep_texts can
    # produce different labels depending on the dataset context.
    grounding_salt = dataset_context.grounding_hint() if dataset_context else ""

    tasks: list[tuple[int, str, list[str]]] = []
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            continue
        if not summary.representative_texts:
            continue
        key = _annotation_cache_key(summary.representative_texts, grounding_salt)
        tasks.append((summary.cluster_id, key, summary.representative_texts))

    pending = [(cid, key, texts) for cid, key, texts in tasks if key not in cache]
    if pending:
        logger.info(
            "Annotation generation: %d via API (cache hits: %d)",
            len(pending), len(tasks) - len(pending),
        )
        client = _make_openai_client(api_key)
        new_annotations = _annotate_clusters_in_parallel(
            client, pending, model, dataset_context
        )
        cache.update(new_annotations)
        utils.save_pickle_cache(cache_path, cache, also_json=True)
    else:
        logger.info(
            "Annotation generation: all %d clusters served from cache", len(tasks)
        )

    return _assemble_annotations(summaries, cache, grounding_salt)


def _assemble_annotations(
    summaries: list[ClusterSummary],
    cache: dict[str, dict[str, str]],
    grounding_salt: str,
) -> dict[int, ClusterAnnotation]:
    """Turn cached `{label, summary}` entries into a cluster_id-keyed dict.

    Handles the three fallback cases (noise, missing reps, missing cache entry)
    in one place so ``generate_cluster_annotations`` stays readable.
    """
    result: dict[int, ClusterAnnotation] = {}
    for summary in summaries:
        if summary.cluster_id == NOISE_LABEL:
            result[summary.cluster_id] = ClusterAnnotation(
                label="Unassigned",
                summary=(
                    "Samples in low-density regions that were not assigned "
                    "to any cluster."
                ),
            )
            continue
        if not summary.representative_texts:
            result[summary.cluster_id] = ClusterAnnotation(
                label=f"Cluster #{summary.cluster_id}",
                summary="(no representative text available)",
            )
            continue
        key = _annotation_cache_key(summary.representative_texts, grounding_salt)
        entry = cache.get(key)
        if entry is None:
            result[summary.cluster_id] = ClusterAnnotation(
                label=f"Cluster #{summary.cluster_id}",
                summary="(generation failed)",
            )
        else:
            result[summary.cluster_id] = ClusterAnnotation(
                label=entry.get("label", f"Cluster #{summary.cluster_id}"),
                summary=entry.get("summary", ""),
            )
    return result


# ---------------------------------------------------------------------------
# Step 3: Duplicate resolution
# ---------------------------------------------------------------------------


def resolve_label_duplicates(
    summaries: list[ClusterSummary],
    annotations: dict[int, ClusterAnnotation],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    dataset_context: DatasetContext | None = None,
    max_iterations: int = MAX_DEDUP_ITERATIONS,
) -> dict[int, ClusterAnnotation]:
    """Iteratively differentiate any duplicate cluster labels.

    Strategy:
        - When several clusters share the same label, keep the largest one
          (ties broken by ascending cluster id).
        - Regenerate the rest with a differentiating prompt that shows both
          clusters' data.
        - Since resolving one duplicate group can create new conflicts, run
          the loop up to ``max_iterations`` times.

    Returns:
        A new annotations dict with duplicates resolved. The input is not mutated.
    """
    size_by_cid = {s.cluster_id: s.size for s in summaries}
    reps_by_cid = {s.cluster_id: s.representative_texts for s in summaries}
    working = dict(annotations)  # shallow copy

    for iteration in range(1, max_iterations + 1):
        duplicates = _find_duplicate_labels(working)
        if not duplicates:
            if iteration > 1:
                logger.info(
                    "Duplicate resolution converged in %d iteration(s)", iteration
                )
            break

        logger.info(
            "Duplicate resolution iteration %d: %d duplicate label groups",
            iteration, len(duplicates),
        )

        tasks = _build_differentiation_tasks(
            duplicates, size_by_cid, reps_by_cid
        )
        if not tasks:
            break

        client = _make_openai_client(api_key)
        new_labels = _differentiate_labels_in_parallel(
            client, tasks, model, dataset_context
        )

        for cid, (new_label, new_summary) in new_labels.items():
            old = working[cid]
            working[cid] = ClusterAnnotation(
                label=new_label,
                # Only the label is being differentiated; reuse existing summary.
                summary=old.summary if new_summary is None else new_summary,
            )
    else:
        logger.warning(
            "Duplicate resolution did not converge within %d iterations. "
            "Manual review recommended.",
            max_iterations,
        )

    return working


def _find_duplicate_labels(
    annotations: dict[int, ClusterAnnotation],
) -> dict[str, list[int]]:
    """Return ``{label: [cluster_id, ...]}`` for every label appearing ≥ 2 times."""
    groups: dict[str, list[int]] = defaultdict(list)
    for cid, ann in annotations.items():
        if cid == NOISE_LABEL:
            continue
        groups[ann.label].append(cid)
    return {label: cids for label, cids in groups.items() if len(cids) >= 2}


def _build_differentiation_tasks(
    duplicates: dict[str, list[int]],
    size_by_cid: dict[int, int],
    reps_by_cid: dict[int, list[str]],
) -> list[tuple[int, list[str], int, list[str]]]:
    """Decide which clusters keep their label and which need regeneration.

    Returns a list of ``(losing_cid, losing_reps, keeper_cid, keeper_reps)``.
    """
    tasks: list[tuple[int, list[str], int, list[str]]] = []
    for label, cids in duplicates.items():
        # Largest cluster keeps its label; all others get regenerated.
        ordered = sorted(cids, key=lambda c: (-size_by_cid.get(c, 0), c))
        keeper = ordered[0]
        keeper_reps = reps_by_cid.get(keeper, [])
        for losing in ordered[1:]:
            losing_reps = reps_by_cid.get(losing, [])
            if not losing_reps:
                continue
            tasks.append((losing, losing_reps, keeper, keeper_reps))
        logger.debug(
            "  duplicate '%s': keeper=#%d(size=%d), regenerate=%s",
            label, keeper, size_by_cid.get(keeper, 0),
            [c for c in ordered[1:]],
        )
    return tasks


# ---------------------------------------------------------------------------
# Backwards compatibility: generate_cluster_names
# ---------------------------------------------------------------------------


def generate_cluster_names(
    summaries: list[ClusterSummary],
    cache_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[int, str]:
    """Legacy wrapper: returns ``cluster_id -> label`` only."""
    annotations = generate_cluster_annotations(
        summaries, cache_dir, model, api_key
    )
    return {cid: ann.label for cid, ann in annotations.items()}


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


def _make_openai_client(api_key: str | None) -> OpenAI:
    """Build the OpenAI client."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in .env or the environment."
        )
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    return OpenAI(api_key=key, timeout=timeout)


def _numpy_rng(seed: int):
    """Lazily import numpy and return a default_rng (helps tests mock this out)."""
    import numpy as np

    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Annotation: parallel execution and single-call helper
# ---------------------------------------------------------------------------


def _build_annotation_system_prompt(context: DatasetContext | None) -> str:
    """Assemble the system prompt with optional grounding context."""
    grounding = context.grounding_hint() if context else ""
    base = (
        "You are a data analyst. From a few representative texts of a cluster, "
        "return a short label and a summary sentence in JSON.\n\n"
    )
    if grounding:
        base += f"[Dataset context]\n{grounding}\n\n"
    base += (
        "Required schema:\n"
        "{\n"
        '  "label": "<10-20 characters. Noun phrase. Concrete business term.>",\n'
        '  "summary": "<1-3 sentences. Describe what these records have in common. '
        'Stick to what the representatives actually say; do not speculate.>"\n'
        "}\n\n"
        "Constraints:\n"
        "- No qualifiers, emojis, quotation marks, or stray punctuation in the label.\n"
        "- The summary states only what the representative texts support.\n"
        "- Write the label and summary in the **same language as the input records**.\n"
        "- Return ONLY valid JSON — no code fences or surrounding prose."
    )
    return base


def _annotate_clusters_in_parallel(
    client: OpenAI,
    pending: list[tuple[int, str, list[str]]],
    model: str,
    context: DatasetContext | None,
) -> dict[str, dict[str, str]]:
    """Issue label/summary requests for each cluster in parallel.

    Results are keyed by the cache key so callers can merge them into the
    persistent cache directly.
    """
    results: dict[str, dict[str, str]] = {}
    lock = threading.Lock()
    system_prompt = _build_annotation_system_prompt(context)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_meta = {
            executor.submit(
                _invoke_annotation_llm, client, texts, model, system_prompt
            ): (cid, key)
            for cid, key, texts in pending
        }
        with tqdm(
            total=len(pending), desc="Annotating", unit="cluster", leave=False
        ) as pbar:
            for future in as_completed(future_to_meta):
                cid, key = future_to_meta[future]
                try:
                    annotation = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Annotation failed for cluster %d: %s", cid, exc
                    )
                    annotation = {
                        "label": f"Cluster #{cid}",
                        "summary": "(generation failed)",
                    }
                with lock:
                    results[key] = annotation
                pbar.update(1)
    return results


def _invoke_annotation_llm(
    client: OpenAI,
    rep_texts: list[str],
    model: str,
    system_prompt: str,
) -> dict[str, str]:
    """Single Chat Completions call for one cluster's annotation."""
    bullets = "\n".join(f"- {_truncate_for_prompt(t)}" for t in rep_texts)
    user_prompt = (
        "Here are representative records of one cluster:\n"
        f"{bullets}\n\n"
        "Return the label and summary as JSON."
    )

    def _call() -> str:
        response = client.chat.completions.create(
            **_build_chat_kwargs(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
        )
        return response.choices[0].message.content or "{}"

    raw = utils.call_with_exponential_backoff(
        _call,
        log_prefix="annotation",
        max_retries=MAX_RETRIES,
        base_sec=BACKOFF_BASE_SEC,
    )
    return _parse_annotation_json(raw)


# ---------------------------------------------------------------------------
# Differentiation: parallel execution and single-call helper
# ---------------------------------------------------------------------------


def _differentiate_labels_in_parallel(
    client: OpenAI,
    tasks: list[tuple[int, list[str], int, list[str]]],
    model: str,
    context: DatasetContext | None,
) -> dict[int, tuple[str, str | None]]:
    """Run the differentiation prompt in parallel for every duplicate.

    Args:
        tasks: list of ``(losing_cid, losing_reps, keeper_cid, keeper_reps)``
    Returns:
        ``losing_cid -> (new_label, new_summary_or_None)``
    """
    results: dict[int, tuple[str, str | None]] = {}
    lock = threading.Lock()
    grounding = context.grounding_hint() if context else ""

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_cid = {
            executor.submit(
                _invoke_differentiation_llm,
                client, losing_reps, keeper_reps, model, grounding,
            ): losing_cid
            for losing_cid, losing_reps, _, keeper_reps in tasks
        }
        with tqdm(
            total=len(tasks), desc="Dedup labels", unit="cluster", leave=False
        ) as pbar:
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    label, summary = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Differentiating label failed for cluster %d: %s",
                        cid, exc,
                    )
                    label, summary = f"Cluster #{cid}", None
                with lock:
                    results[cid] = (label, summary)
                pbar.update(1)
    return results


def _invoke_differentiation_llm(
    client: OpenAI,
    losing_reps: list[str],
    keeper_reps: list[str],
    model: str,
    grounding: str,
) -> tuple[str, str | None]:
    """Request a label for cluster A that clearly separates it from cluster B."""
    losing_bullets = "\n".join(f"- {_truncate_for_prompt(t)}" for t in losing_reps)
    keeper_bullets = "\n".join(f"- {_truncate_for_prompt(t)}" for t in keeper_reps)

    system = (
        "You are a data analyst. Two clusters A and B received the same "
        "auto-generated label. Your task is to re-label cluster A so it is "
        "clearly distinguishable from cluster B.\n\n"
    )
    if grounding:
        system += f"[Dataset context]\n{grounding}\n\n"
    system += (
        "Required schema:\n"
        "{\n"
        '  "label": "<10-20 characters. Concrete noun phrase that differentiates A '
        'from B.>",\n'
        '  "summary": "<optional; 1-3 sentences describing cluster A. '
        'Omit if the existing summary should stay.>"\n'
        "}\n\n"
        "Constraints:\n"
        "- The label MUST be clearly different from cluster B's label.\n"
        "- Avoid lazy abstractions (bare 'issue' or 'problem').\n"
        "- Match the input records' language.\n"
        "- Return ONLY valid JSON."
    )
    user = (
        "=== Cluster A (this is the one to re-label) ===\n"
        f"{losing_bullets}\n\n"
        "=== Cluster B (reference; keeps its existing label) ===\n"
        f"{keeper_bullets}\n\n"
        "Return the new label and (optionally) summary for cluster A as JSON."
    )

    def _call() -> str:
        response = client.chat.completions.create(
            **_build_chat_kwargs(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.35,
                max_tokens=400,
            )
        )
        return response.choices[0].message.content or "{}"

    raw = utils.call_with_exponential_backoff(
        _call,
        log_prefix="dedup",
        max_retries=MAX_RETRIES,
        base_sec=BACKOFF_BASE_SEC,
    )
    parsed = _parse_annotation_json(raw)
    label = parsed["label"]
    # Summary is optional in the differentiation schema.
    summary = parsed["summary"] if parsed.get("summary") else None
    return label, summary


# ---------------------------------------------------------------------------
# Parsing / sanitisation
# ---------------------------------------------------------------------------


def _parse_annotation_json(raw: str) -> dict[str, str]:
    """Parse the LLM response JSON into ``{label, summary}``."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]
    payload = _safe_json_loads(text)
    if payload is None:
        return {"label": "Parse failure", "summary": text[:200]}

    label = _sanitize_label(str(payload.get("label", "")))
    summary = _sanitize_summary(str(payload.get("summary", "")))
    return {"label": label, "summary": summary}


def _safe_json_loads(text: str) -> dict | None:
    """JSON parse that returns ``None`` on failure (and logs)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failure: %s / raw=%r", exc, text[:120])
        return None


# Characters we strip from LLM-produced labels. Keep this tuple small — we only
# want to remove common decoration, not mangle legitimate content.
_LABEL_JUNK_CHARS: tuple[str, ...] = (
    '"', "'", "「", "」", "『", "』", "`",
)


def _sanitize_label(raw: str) -> str:
    """Clean up an LLM label: strip junk characters, keep the first line, cap length."""
    label = raw.strip()
    for ch in _LABEL_JUNK_CHARS:
        label = label.replace(ch, "")
    for line in label.splitlines():
        line = line.strip()
        if line:
            label = line
            break
    if len(label) > 30:
        label = label[:30]
    if not label:
        label = "Untitled"
    return label


def _sanitize_summary(raw: str) -> str:
    """Clean up an LLM summary: collapse whitespace, cap length."""
    summary = raw.strip()
    # Strip surrounding quotes only when they wrap the whole string.
    for ch in ('"', "`"):
        if summary.startswith(ch) and summary.endswith(ch):
            summary = summary[1:-1].strip()
    summary = " ".join(line.strip() for line in summary.splitlines() if line.strip())
    if len(summary) > 400:
        summary = summary[:400] + "…"
    if not summary:
        summary = "(no summary)"
    return summary


def _truncate_for_prompt(text: str, max_chars: int = 500) -> str:
    """Collapse newlines and cap an individual record's length inside a prompt."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


# ---------------------------------------------------------------------------
# Cache key / path
# ---------------------------------------------------------------------------


def _annotation_cache_key(rep_texts: list[str], salt: str) -> str:
    """Stable key that factors in both the representatives and the grounding salt."""
    payload = "\n".join(rep_texts) + "||" + salt
    return utils.content_hash(payload)


def _cache_path_for(cache_dir: Path, model: str) -> Path:
    """Per-model cache file path.

    The ``v3`` suffix distinguishes this schema (label + summary with grounding
    salt) from older caches (v1 was label-only; v2 had no grounding salt).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    return cache_dir / f"cluster_annotations_v3_{safe_model}.pkl"


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------


def _label_frequencies(annotations: dict[int, ClusterAnnotation]) -> Counter:
    """Count label frequencies — used by duplicate-detection tests and the pipeline."""
    return Counter(ann.label for ann in annotations.values())
