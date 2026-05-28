"""Shared runtime and sampling helpers for experiment modules."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from fllmingo import CostTracker, ResponseCache, make_openai_client, resolve_model

from tool_robust_poc.cache import get_cache
from tool_robust_poc.conditions_building import assemble_messages
from tool_robust_poc.core_types import (
    AttackSet,
    CandidatePayload,
    Condition,
    ConditionBlueprint,
    EvalCase,
    JsonDict,
)
from tool_robust_poc.runner_utils import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_CONCURRENCY,
    VLLM_DEFAULT_CONCURRENCY,
    VLLM_DEFAULT_MAX_CONCURRENCY,
    AdaptiveRequestController,
    ProgressTracker,
)
from tool_robust_poc.sampling import deterministic_sample, stable_digest

SampleOrder = Literal["hash", "file"]

POC_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = POC_ROOT / "results"


@dataclass(frozen=True)
class CommonExperimentConfig:
    """Shared config fields used across experiment families."""
    model: str
    provider: str = "openai"
    conditions: tuple[Condition, ...] = (Condition.BASELINE,)
    output: Path | None = None
    summary_output: Path | None = None
    attack_set: AttackSet = "static"
    controls_only: bool = False
    """If True, skip attack cases entirely; only run controls. Used for non-
    adversarial baseline / equivalence sweeps."""
    sample_limit: int | None = None
    sample_order: SampleOrder = "hash"
    sample_seed: str = "tool-robust-poc-v1"
    concurrency: int | None = None  # None = auto (higher for vllm)
    max_concurrency: int | None = None


@dataclass(frozen=True)
class SampledItem[TItem]:
    """A deterministically sampled dataset item with stable identifiers."""

    ordered_index: int
    original_index: int
    item_id: str
    item: TItem


@dataclass
class ExperimentRuntime:
    """Shared runtime state for adaptive experiment execution."""

    client: object
    tracker: CostTracker
    cache: ResponseCache
    api_model: str
    canonical_model: str
    model_short: str
    initial_concurrency: int
    request_controller: AdaptiveRequestController
    progress: ProgressTracker
    event_queue: asyncio.Queue
    controller_task: asyncio.Task


@dataclass(frozen=True)
class PreparedEvaluation[TItem]:
    """One fully rendered evaluation ready to turn into a job."""

    sampled_item: SampledItem[TItem]
    condition: Condition
    case: EvalCase[TItem]
    messages: list[dict]
    extra_api_kwargs: dict | None


def sample_items[TItem](
    items: list[TItem],
    *,
    stable_key_fn: Callable[[TItem], str],
    config: CommonExperimentConfig,
) -> list[SampledItem[TItem]]:
    """Return deterministically ordered dataset items as typed sample records."""
    sampled_records = deterministic_sample(
        items,
        stable_key_fn=stable_key_fn,
        seed=config.sample_seed,
        limit=config.sample_limit,
        order=config.sample_order,
    )
    return [
        SampledItem(
            ordered_index=record["ordered_index"],
            original_index=record["original_index"],
            item_id=record["item_id"],
            item=record["item"],
        )
        for record in sampled_records
    ]


def write_json(path: Path, data: object, *, indent: int = 2) -> None:
    """Write JSON, using gzip compression if path ends with .gz."""
    text = json.dumps(data, indent=indent)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)


def summary_path(raw_results_path: Path, requested_path: Path | None) -> Path:
    if requested_path is not None:
        return requested_path
    stem = raw_results_path.stem
    if raw_results_path.suffix == ".gz":
        stem = Path(stem).stem
    return raw_results_path.with_name(f"{stem}_summary.json.gz")


def sample_tag(*, order: SampleOrder, seed: str) -> str:
    if order == "file":
        return "fileorder"
    return f"hash_{stable_digest(seed, length=8)}"


def default_results_path(
    *,
    task_name: str,
    n_items: int,
    config: CommonExperimentConfig,
) -> Path:
    model_tag = config.model.replace("/", "_").replace(".", "")
    current_sample_tag = sample_tag(order=config.sample_order, seed=config.sample_seed)
    return RESULTS_DIR / f"{task_name}_{n_items}q_{current_sample_tag}_{model_tag}.json.gz"


def build_runtime(
    *,
    experiment_name: str,
    config: CommonExperimentConfig,
) -> ExperimentRuntime:
    """Create the shared API client, cache, tracker, and controller state."""
    canonical_model = config.model
    api_model = resolve_model(canonical_model, config.provider)
    client = make_openai_client(provider=config.provider)
    tracker = CostTracker(
        model=canonical_model,
        provider=config.provider,
        experiment=experiment_name,
        ledger_path=POC_ROOT / "cost_ledger.jsonl",
    )
    cache = get_cache(canonical_model)
    is_vllm = config.provider == "vllm"
    concurrency = config.concurrency or (VLLM_DEFAULT_CONCURRENCY if is_vllm else DEFAULT_CONCURRENCY)
    max_concurrency = config.max_concurrency or (VLLM_DEFAULT_MAX_CONCURRENCY if is_vllm else DEFAULT_MAX_CONCURRENCY)
    initial_concurrency = min(concurrency, max_concurrency)
    request_controller = AdaptiveRequestController(
        initial_concurrency=initial_concurrency,
        max_concurrency=max_concurrency,
    )
    progress = ProgressTracker(
        total=0,
        model_label=canonical_model,
        controller=request_controller,
    )
    event_queue: asyncio.Queue = asyncio.Queue()
    controller_task = asyncio.create_task(request_controller.run(event_queue))
    model_short = canonical_model.replace("/", "_").replace(".", "").replace(":", "_")
    return ExperimentRuntime(
        client=client,
        tracker=tracker,
        cache=cache,
        api_model=api_model,
        canonical_model=canonical_model,
        model_short=model_short,
        initial_concurrency=initial_concurrency,
        request_controller=request_controller,
        progress=progress,
        event_queue=event_queue,
        controller_task=controller_task,
    )


async def shutdown_runtime(runtime: ExperimentRuntime) -> None:
    """Drain controller state and stop background progress tasks."""
    await runtime.progress.stop_heartbeat()
    await runtime.event_queue.join()
    runtime.controller_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runtime.controller_task


def build_case_metadata[TItem](
    sampled_item: SampledItem[TItem],
    *,
    condition: str,
    case: EvalCase[TItem],
    candidate: str,
    extra: JsonDict | None = None,
) -> JsonDict:
    """Return standard result metadata for one evaluated case."""
    metadata: JsonDict = {
        "q_index": sampled_item.ordered_index,
        "item_id": sampled_item.item_id,
        "original_index": sampled_item.original_index,
        "condition": condition,
        "candidate_label": case.label,
        "candidate": candidate,
        "case_id": case.case_id,
        "is_under_attack": case.is_under_attack,
        **case.metadata,
    }
    if extra:
        metadata.update(extra)
    return metadata


def make_attack_case[TItem](
    *,
    item: TItem,
    item_id: str,
    label: str,
    candidate: str,
) -> EvalCase[TItem]:
    """Wrap an adversarial string candidate as an attack eval case."""
    return EvalCase(
        case_id=f"{item_id}::{label}",
        label=label,
        item=item,
        candidate=CandidatePayload(response=candidate),
        is_under_attack=True,
    )


def prepare_evaluations[TItem](
    *,
    sampled_records: Sequence[SampledItem[TItem]],
    conditions: dict[Condition, ConditionBlueprint],
    make_payload: Callable[[TItem, str], dict[str, str]],
    make_controls: Callable[[TItem], list[EvalCase[TItem]]],
    attacks: Sequence[tuple[str, str]],
) -> list[PreparedEvaluation[TItem]]:
    """Render all control and attack cases into prompt-ready evaluations."""
    prepared: list[PreparedEvaluation[TItem]] = []

    for record in sampled_records:
        item = record.item
        controls = make_controls(item)

        for condition, blueprint in conditions.items():
            all_cases = list(controls)
            all_cases.extend(
                make_attack_case(
                    item=item,
                    item_id=record.item_id,
                    label=label,
                    candidate=candidate,
                )
                for label, candidate in attacks
            )
            for case in all_cases:
                candidate = case.candidate.response
                payload = make_payload(item, candidate)
                messages, extra = assemble_messages(blueprint, payload)
                prepared.append(
                    PreparedEvaluation(
                        sampled_item=record,
                        condition=condition,
                        case=case,
                        messages=messages,
                        extra_api_kwargs=extra or None,
                    )
                )

    return prepared


def build_run_metadata(
    *,
    task_name: str,
    data_path: Path,
    config: CommonExperimentConfig,
    runtime: ExperimentRuntime,
    raw_dataset_size: int,
    sampled_items: Sequence[SampledItem[object]],
    extra: JsonDict | None = None,
) -> JsonDict:
    """Return stable metadata stored alongside raw results and summaries."""
    metadata: JsonDict = {
        "task": task_name,
        "data_path": str(data_path),
        "provider": config.provider,
        "model": runtime.canonical_model,
        "api_model": runtime.api_model,
        "conditions": [str(c) for c in config.conditions],
        "attack_set": config.attack_set,
        "sample_order": config.sample_order,
        "sample_seed": config.sample_seed,
        "sample_limit": config.sample_limit,
        "initial_concurrency": runtime.initial_concurrency,
        "max_concurrency": config.max_concurrency,
        "dataset_size_raw": raw_dataset_size,
        "dataset_size_used": len(sampled_items),
        "sample_item_ids": [record.item_id for record in sampled_items],
    }
    if extra:
        metadata.update(extra)
    return metadata
