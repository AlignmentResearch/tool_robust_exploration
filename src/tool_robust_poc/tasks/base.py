"""Shared task-spec dataclasses for prompt rendering and output parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from tool_robust_poc.core_types import PairwiseVerdict

# Which direction counts as "attack success" for a given judge.
#   scalar:   "low_score"  = attacker wants a low score (e.g. quality-judge MT-Bench).
#             "high_score" = attacker wants a high score (e.g. harm-judge where high=harmful).
#   pairwise: "win"     = attacker's candidate being picked is success (Arena-Hard).
#             "not_win" = attacker's candidate NOT being picked is success
#                         (e.g. "which is more harmful"-style judges).
ScalarAttackerGoal = Literal["low_score", "high_score"]
PairwiseAttackerGoal = Literal["win", "not_win"]


def _parse_binary_verdict(pattern: str):
    """Create a first-match verdict parser for a given VERDICT: X or Y pattern."""

    def parse(text: str) -> str | None:
        m = re.search(rf"VERDICT:\s*({pattern})", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    return parse


def _parse_binary_verdict_last(pattern: str):
    """Create a last-match verdict parser for VERDICT: X or Y.

    Where ``_parse_binary_verdict`` returns the first VERDICT: token, this
    returns the last one. Useful when the judge writes multi-paragraph CoT
    that may quote the candidate's earlier injection token before
    delivering its own verdict at the end.
    """

    def parse(text: str) -> str | None:
        matches = re.findall(rf"VERDICT:\s*({pattern})", text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        return None

    return parse


@dataclass
class JudgePromptSpec:
    """Base fields shared across all task families.

    Rendering/template fields have moved to per-condition ConditionBlueprints.
    This holds only judge identity, attacks, and the raw instruction text
    (useful as a reference, e.g. for constructing blueprint system_text).
    """

    judge_instructions: str
    task_attacks: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class BinaryJudgeSpec(JudgePromptSpec):
    """String-verdict judge task evaluated with exact expected labels."""

    parse_fn: Callable[[str], str | None] = field(
        default_factory=lambda: _parse_binary_verdict("CORRECT|INCORRECT")
    )
    positive_verdict: str = "CORRECT"


@dataclass
class ScalarJudgeSpec(JudgePromptSpec):
    """Scalar judge task that preserves raw numeric scores.

    `attacker_goal` declares which direction of score shift counts as attack
    success. `attacker_success_threshold` is a static score boundary used for
    binary success-count reporting; leave None if success is defined only
    relative to a baseline (e.g. dynamic attacks that need a non-noise shift
    vs. the per-item baseline).
    """

    parse_score: Callable[[str], int | float | None] = field(
        default_factory=lambda: lambda _text: None
    )
    min_score: int | float | None = None
    max_score: int | float | None = None
    attacker_goal: ScalarAttackerGoal = "low_score"
    attacker_success_threshold: float | None = None


@dataclass
class PairwiseJudgeSpec(JudgePromptSpec):
    """Pairwise preference judge that compares two candidates (A vs B).

    Supports 5-label scales (Arena-Hard, WildBench) and binary scales
    (Prometheus) through a common PairwiseVerdict output type.

    `attacker_goal` declares whether the attacker wants their candidate
    PICKED (Arena-Hard style) or NOT picked (e.g. "which is more harmful").
    """

    parse_preference: Callable[[str], PairwiseVerdict | None] = field(
        default_factory=lambda: lambda _text: None
    )
    attacker_goal: PairwiseAttackerGoal = "win"
