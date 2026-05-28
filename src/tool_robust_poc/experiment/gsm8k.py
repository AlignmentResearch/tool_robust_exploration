"""Reusable GSM8K experiment wiring for scripts and sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from tool_robust_poc.core_types import EvalCase
from tool_robust_poc.experiment.binary import BinaryExperimentSpec
from tool_robust_poc.tasks.gsm8k import (
    CONDITIONS,
    Gsm8kItem,
    Gsm8kPromptVariant,
    build_conditions,
    build_task,
    get_attacks,
    load_data,
    make_controls,
    make_payload,
    negative_verdict,
    stable_item_key,
)

POC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = POC_ROOT / "data" / "gsm8k_sample_100.json"


def expected_gsm8k_verdict(case: EvalCase[Gsm8kItem]) -> str:
    """Return the verdict the judge should produce for a GSM8K case."""
    if case.is_under_attack:
        return "INCORRECT"  # attacks always present wrong answers
    return "CORRECT" if bool(case.metadata.get("expected_correct")) else "INCORRECT"


def make_expected_gsm8k_verdict(
    prompt_variant: Gsm8kPromptVariant = "verdict",
) -> Callable[[EvalCase[Gsm8kItem]], str]:
    """Build the expected-verdict function for a GSM8K prompt variant."""
    positive = build_task(prompt_variant).positive_verdict
    negative = negative_verdict(prompt_variant)

    def expected(case: EvalCase[Gsm8kItem]) -> str:
        if case.is_under_attack:
            return negative
        return positive if bool(case.metadata.get("expected_correct")) else negative

    return expected


def build_gsm8k_spec(
    data_path: Path = DEFAULT_DATA,
    prompt_variant: Gsm8kPromptVariant = "verdict",
) -> BinaryExperimentSpec[Gsm8kItem]:
    """Build the shared GSM8K binary experiment spec."""
    task = build_task(prompt_variant)
    return BinaryExperimentSpec(
        task_name="gsm8k",
        title="GSM8K",
        data_path=data_path,
        task=task,
        conditions=(
            CONDITIONS
            if prompt_variant == "verdict"
            else build_conditions(task.judge_instructions)
        ),
        load_data=load_data,
        stable_item_key=stable_item_key,
        make_payload=make_payload,
        make_controls=make_controls,
        get_attacks=lambda attack_set: get_attacks(
            attack_set,
            positive_verdict=task.positive_verdict,
        ),
        expected_verdict=make_expected_gsm8k_verdict(prompt_variant),
        run_metadata_extra={"prompt_variant": prompt_variant},
    )
