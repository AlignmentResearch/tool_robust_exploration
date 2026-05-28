"""Shared orchestration for scalar score experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from tool_robust_poc.attacks import get_attack_labels
from tool_robust_poc.core_types import (
    AttackSet,
    Condition,
    ConditionBlueprint,
    EvalCase,
    JsonDict,
)
from tool_robust_poc.reporting_scalar import build_scalar_summary
from tool_robust_poc.runner_utils import eval_one_scalar, run_job_queue
from tool_robust_poc.tasks import ScalarJudgeSpec
from tool_robust_poc.experiment.common import (
    CommonExperimentConfig,
    PreparedEvaluation,
    RESULTS_DIR,
    SampledItem,
    build_case_metadata,
    build_run_metadata,
    build_runtime,
    default_results_path,
    prepare_evaluations,
    shutdown_runtime,
    summary_path,
    sample_items,
    write_json,
)


def _empty_case_metadata[TItem](_case: EvalCase[TItem]) -> JsonDict:
    return {}


@dataclass(frozen=True)
class ScalarExperimentSpec[TItem]:
    """Task-local callbacks and metadata needed for a scalar run."""

    task_name: str
    title: str
    data_path: Path
    task: ScalarJudgeSpec
    conditions: dict[Condition, ConditionBlueprint]
    load_data: Callable[[Path], list[TItem]]
    stable_item_key: Callable[[TItem], str]
    make_payload: Callable[[TItem, str], dict[str, str]]
    make_controls: Callable[[TItem], list[EvalCase[TItem]]]
    get_attacks: Callable[[AttackSet], list[tuple[str, str]]]
    build_case_metadata: Callable[[EvalCase[TItem]], JsonDict] = _empty_case_metadata
    run_metadata_extra: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarExperimentArtifacts[TItem]:
    """Saved outputs and rich context returned from a scalar run."""

    results: list[dict]
    summary: dict
    results_path: Path
    summary_path: Path
    attack_labels: list[str]
    sampled_items: list[SampledItem[TItem]]
    questions: list[TItem]
    tracker_summary: str


def _make_scalar_job_factory[TItem](
    *,
    prepared: PreparedEvaluation[TItem],
    spec: ScalarExperimentSpec[TItem],
    config: CommonExperimentConfig,
    runtime,
) -> Callable[[], Awaitable[dict]]:
    """Build one deferred scalar eval job from a prepared evaluation."""
    case = prepared.case
    candidate = case.candidate.response
    record = prepared.sampled_item
    condition = prepared.condition
    return lambda case=case, candidate=candidate, record=record, condition=condition: eval_one_scalar(
        runtime.client,
        runtime.api_model,
        prepared.messages,
        cache=runtime.cache,
        tracker=runtime.tracker,
        cache_tag=(
            f"{spec.task_name}_{runtime.model_short}_{condition}_{record.item_id}_{case.label}"
        ),
        parse_score=spec.task.parse_score,
        provider=config.provider,
        progress=runtime.progress,
        extra_api_kwargs=prepared.extra_api_kwargs,
        request_controller=runtime.request_controller,
        event_sink=runtime.event_queue,
        metadata=build_case_metadata(
            record,
            condition=condition,
            case=case,
            candidate=candidate,
            extra=spec.build_case_metadata(case),
        ),
    )


async def run_scalar_experiment[TItem](
    *,
    spec: ScalarExperimentSpec[TItem],
    config: CommonExperimentConfig,
) -> ScalarExperimentArtifacts[TItem]:
    """Run a scalar score experiment and persist raw + summary outputs."""
    raw_items = spec.load_data(spec.data_path)
    sampled_records = sample_items(
        raw_items,
        stable_key_fn=spec.stable_item_key,
        config=config,
    )
    questions = [record.item for record in sampled_records]
    if not questions:
        raise ValueError(f"No {spec.title} items selected after deterministic sampling.")

    runtime = build_runtime(
        experiment_name=f"{spec.task_name}_{len(questions)}q",
        config=config,
    )

    # Filter conditions to those requested in config
    active_conditions = {
        c: bp for c, bp in spec.conditions.items()
        if c in config.conditions
    }

    all_attacks = [] if config.controls_only else spec.get_attacks(config.attack_set)
    attack_labels = get_attack_labels(all_attacks)
    prepared_evaluations = prepare_evaluations(
        sampled_records=sampled_records,
        conditions=active_conditions,
        make_payload=spec.make_payload,
        make_controls=spec.make_controls,
        attacks=all_attacks,
    )
    job_factories = [
        _make_scalar_job_factory(
            prepared=prepared,
            spec=spec,
            config=config,
            runtime=runtime,
        )
        for prepared in prepared_evaluations
    ]

    runtime.progress.total = len(job_factories)
    runtime.progress.reset_timing()
    runtime.progress.start_heartbeat()

    print(
        f"Launching {len(job_factories)} scalar evaluations "
        f"({len(questions)} questions x {len(active_conditions)} conditions x "
        f"{len(all_attacks) + len(spec.make_controls(questions[0]))} candidates), "
        f"model={runtime.canonical_model} (api: {runtime.api_model}), "
        f"provider={config.provider}, initial {runtime.initial_concurrency} concurrent, "
        f"max {runtime.request_controller.max_concurrency}..."
    )

    try:
        results = await run_job_queue(
            job_factories,
            worker_count=runtime.request_controller.max_concurrency,
        )
    finally:
        await shutdown_runtime(runtime)
    sys.stderr.write("\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = config.output or default_results_path(
        task_name=spec.task_name,
        n_items=len(questions),
        config=config,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(results_path, results)
    print(f"Raw results saved to {results_path}")

    conditions_list = [str(c) for c in active_conditions]
    summary = build_scalar_summary(
        results,
        conditions=conditions_list,
        attack_labels=attack_labels,
        title=spec.title,
        model=runtime.canonical_model,
        run_metadata=build_run_metadata(
            task_name=spec.task_name,
            data_path=spec.data_path,
            config=config,
            runtime=runtime,
            raw_dataset_size=len(raw_items),
            sampled_items=sampled_records,
            extra={
                "judge_attacker_goal": spec.task.attacker_goal,
                "judge_attacker_success_threshold": spec.task.attacker_success_threshold,
                **(spec.run_metadata_extra or {}),
            },
        ),
    )
    output_summary_path = summary_path(results_path, config.summary_output)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_summary_path, summary)
    print(f"Summary saved to {output_summary_path}")

    return ScalarExperimentArtifacts(
        results=results,
        summary=summary,
        results_path=results_path,
        summary_path=output_summary_path,
        attack_labels=attack_labels,
        sampled_items=sampled_records,
        questions=questions,
        tracker_summary=runtime.tracker.summary(),
    )
