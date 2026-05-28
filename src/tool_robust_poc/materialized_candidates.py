"""Helpers for storing and auto-generating materialized candidate responses."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

from fllmingo import (
    CostTracker,
    ResponseCache,
    cached_chat_completion,
    make_openai_client,
    resolve_model,
)

from tool_robust_poc.core_types import JsonDict
from tool_robust_poc.runner_utils import run_job_queue

_LABEL_CLEAN_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ModelGenerationSpec:
    """One baseline-generation source to materialize for each item."""

    label: str
    provider: str
    model: str
    generation_mode: str = "direct_answer"


@dataclass(frozen=True)
class MaterializedCandidate:
    """One saved candidate response that can be reused across experiments."""

    item_id: str
    label: str
    response: str
    source_provider: str
    source_model: str
    generation_mode: str
    metadata: JsonDict = field(default_factory=dict)


def _clean_label_part(text: str) -> str:
    lowered = text.lower()
    cleaned = _LABEL_CLEAN_RE.sub("_", lowered).strip("_")
    return cleaned or "candidate"


def parse_model_generation_specs(
    specs: Sequence[str],
    *,
    default_provider: str,
) -> list[ModelGenerationSpec]:
    """Parse CLI model specs into generation sources with stable labels."""
    parsed: list[ModelGenerationSpec] = []
    seen_labels: set[str] = set()
    for spec in specs:
        provider, model = _parse_model_spec(spec, default_provider=default_provider)
        label = f"baseline_{_clean_label_part(provider)}_{_clean_label_part(model)}"
        if label in seen_labels:
            raise ValueError(f"Duplicate baseline label derived from spec {spec!r}: {label}")
        seen_labels.add(label)
        parsed.append(
            ModelGenerationSpec(
                label=label,
                provider=provider,
                model=model,
            )
        )
    return parsed


def _parse_model_spec(spec: str, *, default_provider: str) -> tuple[str, str]:
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider, model
    return default_provider, spec


def load_materialized_candidates(path: Path) -> list[MaterializedCandidate]:
    """Load materialized candidates from JSONL, returning an empty list if absent."""
    if not path.exists():
        return []
    candidates: list[MaterializedCandidate] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        candidates.append(
            MaterializedCandidate(
                item_id=data["item_id"],
                label=data["label"],
                response=data["response"],
                source_provider=data["source_provider"],
                source_model=data["source_model"],
                generation_mode=data.get("generation_mode", "direct_answer"),
                metadata=data.get("metadata", {}),
            )
        )
    return candidates


def save_materialized_candidates(
    path: Path,
    candidates: Sequence[MaterializedCandidate],
) -> None:
    """Write materialized candidates to JSONL in stable order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candidates, key=lambda item: (item.item_id, item.label))
    lines = [json.dumps(asdict(candidate), sort_keys=True) for candidate in ordered]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def group_materialized_candidates_by_item_id(
    candidates: Iterable[MaterializedCandidate],
) -> dict[str, list[MaterializedCandidate]]:
    """Group materialized candidates by dataset item id."""
    grouped: dict[str, list[MaterializedCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.item_id, []).append(candidate)
    return {
        item_id: sorted(values, key=lambda candidate: candidate.label)
        for item_id, values in grouped.items()
    }


async def ensure_materialized_candidates[TItem](
    items: Sequence[TItem],
    *,
    item_id_fn: Callable[[TItem], str],
    render_messages: Callable[[TItem], list[dict[str, str]]],
    source_specs: Sequence[ModelGenerationSpec],
    output_path: Path,
    cache_root: Path,
    max_concurrency: int = 4,
    temperature: float = 0.2,
    max_completion_tokens: int = 2048,
) -> list[MaterializedCandidate]:
    """Load or generate materialized candidates for the requested item/source pairs."""
    existing = load_materialized_candidates(output_path)
    existing_map = {(candidate.item_id, candidate.label): candidate for candidate in existing}
    all_candidates = dict(existing_map)

    for source_spec in source_specs:
        missing_items = [
            item
            for item in items
            if (item_id_fn(item), source_spec.label) not in existing_map
        ]
        if not missing_items:
            print(
                f"Using cached materialized candidates for {source_spec.label} "
                f"({source_spec.provider}:{source_spec.model})."
            )
            continue

        print(
            f"Generating {len(missing_items)} materialized candidates for {source_spec.label} "
            f"({source_spec.provider}:{source_spec.model})..."
        )
        generated = await _generate_candidates_for_source(
            missing_items,
            item_id_fn=item_id_fn,
            render_messages=render_messages,
            source_spec=source_spec,
            cache_root=cache_root,
            max_concurrency=max_concurrency,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        for candidate in generated:
            all_candidates[(candidate.item_id, candidate.label)] = candidate

    save_materialized_candidates(output_path, list(all_candidates.values()))
    return load_materialized_candidates(output_path)


async def _generate_candidates_for_source[TItem](
    items: Sequence[TItem],
    *,
    item_id_fn: Callable[[TItem], str],
    render_messages: Callable[[TItem], list[dict[str, str]]],
    source_spec: ModelGenerationSpec,
    cache_root: Path,
    max_concurrency: int,
    temperature: float,
    max_completion_tokens: int,
) -> list[MaterializedCandidate]:
    api_model = resolve_model(source_spec.model, source_spec.provider)
    client = make_openai_client(provider=source_spec.provider)
    poc_root = Path(__file__).resolve().parents[2]
    tracker = CostTracker(
        model=source_spec.model,
        provider=source_spec.provider,
        experiment=f"materialized_candidates_{source_spec.label}",
        ledger_path=poc_root / "cost_ledger.jsonl",
    )
    cache = ResponseCache(cache_root / source_spec.model.replace("/", "_"))
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def make_one(item: TItem) -> MaterializedCandidate:
        item_id = item_id_fn(item)
        async with sem:
            completion = await cached_chat_completion(
                client,
                model=api_model,
                messages=render_messages(item),
                cache=cache,
                tracker=tracker,
                cache_tag=f"materialized_{source_spec.label}_{item_id}",
                provider=source_spec.provider,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
        return MaterializedCandidate(
            item_id=item_id,
            label=source_spec.label,
            response=completion.text.strip(),
            source_provider=source_spec.provider,
            source_model=source_spec.model,
            generation_mode=source_spec.generation_mode,
            metadata={
                "finish_reason": completion.finish_reason,
            },
        )

    job_factories: list[Callable[[], Awaitable[MaterializedCandidate]]] = [
        (lambda item=item: make_one(item))
        for item in items
    ]
    generated = await run_job_queue(job_factories, worker_count=max_concurrency)
    print(tracker.summary())
    return generated
