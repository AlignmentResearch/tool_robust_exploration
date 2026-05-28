"""Shared orchestration for pairwise preference experiments.

Unlike pointwise experiments, each (item, condition, attack) produces TWO
evaluations — one with the attack in position A and one in position B.
This position-swap protocol cancels position bias.

Controls use the vendored baseline answer in both positions (no attack).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from tool_robust_poc.attacks import get_attack_labels
from tool_robust_poc.conditions_building import assemble_messages
from tool_robust_poc.sampling import stable_digest
from tool_robust_poc.core_types import (
    AttackSet,
    CandidatePayload,
    Condition,
    ConditionBlueprint,
    JsonDict,
    PairwiseEvalCase,
    PairwiseVerdict,
)
from tool_robust_poc.reporting_pairwise import build_pairwise_summary
from tool_robust_poc.runner_utils import eval_one_pairwise, run_job_queue
from tool_robust_poc.tasks import PairwiseJudgeSpec
from tool_robust_poc.experiment.common import (
    CommonExperimentConfig,
    RESULTS_DIR,
    SampledItem,
    build_run_metadata,
    build_runtime,
    default_results_path,
    shutdown_runtime,
    summary_path,
    sample_items,
    write_json,
)


@dataclass(frozen=True)
class PairwiseExperimentSpec[TItem]:
    """Task-local callbacks and metadata needed for a pairwise run."""

    task_name: str
    title: str
    data_path: Path
    task: PairwiseJudgeSpec
    conditions: dict[Condition, ConditionBlueprint]
    load_data: Callable[[Path], list[TItem]]
    stable_item_key: Callable[[TItem], str]
    make_payload: Callable[[TItem, str, str], dict[str, str]]
    get_attacks: Callable[[AttackSet], list[tuple[str, str]]]
    get_baseline_answer: Callable[[TItem], str]
    """Return the real baseline answer for a given item (e.g. vendored o3-mini)."""
    get_weak_baseline_answer: Callable[[TItem], str | None] | None = None
    """Optional: return a weaker-model answer for a non-adversarial control case.
    When provided and non-None for an item, an extra control eval pits the weak
    answer against the strong baseline (weak in the position an attacker would
    occupy, deterministic via _pick_position)."""
    run_metadata_extra: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseExperimentArtifacts[TItem]:
    """Saved outputs and rich context returned from a pairwise run."""

    results: list[dict]
    summary: dict
    results_path: Path
    summary_path: Path
    attack_labels: list[str]
    sampled_items: list[SampledItem[TItem]]
    questions: list[TItem]
    tracker_summary: str


# ---------------------------------------------------------------------------
# Prepared evaluation — one API call ready to dispatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedPairwiseEvaluation[TItem]:
    """One fully rendered pairwise evaluation ready to turn into a job."""

    sampled_item: SampledItem[TItem]
    condition: Condition
    case: PairwiseEvalCase[TItem]
    messages: list[dict]
    extra_api_kwargs: dict | None


# ---------------------------------------------------------------------------
# Preparation — build all evaluations with random position assignment
# ---------------------------------------------------------------------------

def _pick_position(item_id: str) -> str:
    """Deterministically assign attack position A or B for an item.

    Hashes only by item_id so that all attacks on the same question land
    on the same side. This keeps cross-attack and cross-condition
    comparisons within an item consistent; position bias washes out
    across the sample.
    """
    digest = stable_digest(item_id, length=8)
    return "A" if int(digest, 16) % 2 == 0 else "B"


def prepare_pairwise_evaluations[TItem](
    *,
    sampled_records: Sequence[SampledItem[TItem]],
    conditions: dict[Condition, ConditionBlueprint],
    make_payload: Callable[[TItem, str, str], dict[str, str]],
    get_baseline_answer: Callable[[TItem], str],
    attacks: Sequence[tuple[str, str]],
    get_weak_baseline_answer: Callable[[TItem], str | None] | None = None,
    include_baseline_vs_baseline: bool = True,
) -> list[PreparedPairwiseEvaluation[TItem]]:
    """Render all control and attack cases into prompt-ready pairwise evaluations.

    For each (item, condition):
      - When `include_baseline_vs_baseline` is True (default), one control
        case pits the strong baseline against itself. This is mostly a
        sanity check (judge should tie identical responses) and adds no
        signal once that's confirmed; pass False to skip it.
      - If `get_weak_baseline_answer` is provided and returns a non-None
        weak answer for this item, one additional non-adversarial control
        case pitting weak vs strong baseline. The weak answer is placed in
        the deterministic _pick_position(item_id) slot, mirroring how
        attacker candidates are placed.
      - For each attack, one case with deterministically random position
        assignment (A or B) based on hash(item_id, attack_label).
        This halves eval count vs. full position swap while still mixing
        positions across items to reduce position bias.
    """
    prepared: list[PreparedPairwiseEvaluation[TItem]] = []

    for record in sampled_records:
        item = record.item
        item_id = record.item_id
        baseline = get_baseline_answer(item)
        weak_answer = (
            get_weak_baseline_answer(item) if get_weak_baseline_answer else None
        )

        for condition, blueprint in conditions.items():
            if include_baseline_vs_baseline:
                control_case = PairwiseEvalCase(
                    case_id=f"{item_id}::control_baseline_vs_baseline",
                    label="control_baseline_vs_baseline",
                    item=item,
                    candidate_a=CandidatePayload(response=baseline),
                    candidate_b=CandidatePayload(response=baseline),
                    metadata={"is_attack": False, "attacked_position": None},
                )
                payload = make_payload(item, baseline, baseline)
                messages, extra = assemble_messages(blueprint, payload)
                prepared.append(
                    PreparedPairwiseEvaluation(
                        sampled_item=record,
                        condition=condition,
                        case=control_case,
                        messages=messages,
                        extra_api_kwargs=extra or None,
                    )
                )

            # Optional weak-vs-strong non-adversarial control
            if weak_answer is not None:
                weak_position = _pick_position(item_id)
                if weak_position == "A":
                    weak_a, weak_b = weak_answer, baseline
                else:
                    weak_a, weak_b = baseline, weak_answer
                weak_case = PairwiseEvalCase(
                    case_id=f"{item_id}::control_weak_vs_strong::pos_{weak_position}",
                    label="control_weak_vs_strong",
                    item=item,
                    candidate_a=CandidatePayload(response=weak_a),
                    candidate_b=CandidatePayload(response=weak_b),
                    metadata={
                        "is_attack": False,
                        # Record the weak-candidate position so per-label scoring
                        # interprets "weak side win rate" consistently across items.
                        "attacked_position": weak_position,
                    },
                )
                payload = make_payload(item, weak_a, weak_b)
                msgs, extra = assemble_messages(blueprint, payload)
                prepared.append(
                    PreparedPairwiseEvaluation(
                        sampled_item=record,
                        condition=condition,
                        case=weak_case,
                        messages=msgs,
                        extra_api_kwargs=extra or None,
                    )
                )

            # Attack cases: deterministic random position per (item, attack)
            for label, attack_text in attacks:
                position = _pick_position(item_id)

                if position == "A":
                    cand_a = attack_text
                    cand_b = baseline
                else:
                    cand_a = baseline
                    cand_b = attack_text

                case = PairwiseEvalCase(
                    case_id=f"{item_id}::{label}::pos_{position}",
                    label=label,
                    item=item,
                    candidate_a=CandidatePayload(response=cand_a),
                    candidate_b=CandidatePayload(response=cand_b),
                    metadata={
                        "is_attack": True,
                        "attacked_position": position,
                        "attack_label": label,
                    },
                )
                payload = make_payload(item, cand_a, cand_b)
                msgs, extra = assemble_messages(blueprint, payload)
                prepared.append(
                    PreparedPairwiseEvaluation(
                        sampled_item=record,
                        condition=condition,
                        case=case,
                        messages=msgs,
                        extra_api_kwargs=extra or None,
                    )
                )

    return prepared


# ---------------------------------------------------------------------------
# Job factory
# ---------------------------------------------------------------------------

def _make_pairwise_job_factory[TItem](
    *,
    prepared: PreparedPairwiseEvaluation[TItem],
    spec: PairwiseExperimentSpec[TItem],
    config: CommonExperimentConfig,
    runtime,
) -> Callable[[], Awaitable[dict]]:
    """Build one deferred pairwise eval job from a prepared evaluation."""
    case = prepared.case
    record = prepared.sampled_item
    condition = prepared.condition

    metadata: JsonDict = {
        "q_index": record.ordered_index,
        "item_id": record.item_id,
        "original_index": record.original_index,
        "condition": str(condition),
        "candidate_label": case.label,
        "case_id": case.case_id,
        **case.metadata,
    }

    return lambda: eval_one_pairwise(
        runtime.client,
        runtime.api_model,
        prepared.messages,
        cache=runtime.cache,
        tracker=runtime.tracker,
        cache_tag=(
            f"{spec.task_name}_{runtime.model_short}_{condition}_{record.item_id}_{case.case_id}"
        ),
        parse_preference=spec.task.parse_preference,
        provider=config.provider,
        progress=runtime.progress,
        extra_api_kwargs=prepared.extra_api_kwargs,
        request_controller=runtime.request_controller,
        event_sink=runtime.event_queue,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

async def run_pairwise_experiment[TItem](
    *,
    spec: PairwiseExperimentSpec[TItem],
    config: CommonExperimentConfig,
) -> PairwiseExperimentArtifacts[TItem]:
    """Run a pairwise preference experiment and persist raw + summary outputs."""
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
    # When running controls-only with a weak baseline, the trivial
    # baseline-vs-baseline control adds no signal (judge always ties
    # identical responses). Skip it to halve the cost of controls runs.
    include_self_control = not (
        config.controls_only and spec.get_weak_baseline_answer is not None
    )
    prepared_evaluations = prepare_pairwise_evaluations(
        sampled_records=sampled_records,
        conditions=active_conditions,
        make_payload=spec.make_payload,
        get_baseline_answer=spec.get_baseline_answer,
        attacks=all_attacks,
        get_weak_baseline_answer=spec.get_weak_baseline_answer,
        include_baseline_vs_baseline=include_self_control,
    )
    job_factories = [
        _make_pairwise_job_factory(
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
        f"Launching {len(job_factories)} pairwise evaluations "
        f"({len(questions)} questions x {len(active_conditions)} conditions x "
        f"({len(all_attacks)} attacks + 1 control)), "
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
    summary = build_pairwise_summary(
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
                **(spec.run_metadata_extra or {}),
            },
        ),
    )
    output_summary_path = summary_path(results_path, config.summary_output)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_summary_path, summary)
    print(f"Summary saved to {output_summary_path}")

    return PairwiseExperimentArtifacts(
        results=results,
        summary=summary,
        results_path=results_path,
        summary_path=output_summary_path,
        attack_labels=attack_labels,
        sampled_items=sampled_records,
        questions=questions,
        tracker_summary=runtime.tracker.summary(),
    )
