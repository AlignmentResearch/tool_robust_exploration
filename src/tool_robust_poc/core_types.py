"""Shared typed models for evaluation cases and prompt-building inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonDict = dict[str, JsonValue]

type AttackSet = Literal["content", "static"]
ATTACK_SET_CHOICES: tuple[AttackSet, AttackSet] = ("content", "static")

type PreferenceDirection = Literal["A", "B", "tie"]
type PreferenceStrength = Literal["strong", "slight"]


class Condition(StrEnum):
    """Structural conditions for judge prompt experiments."""

    BASELINE = "baseline"
    """Single user message, all content inline."""

    MULTI_MSG = "multi_msg"
    """System + user messages, all content inline."""

    SYSTEM_DISTRUST = "system_distrust"
    """System (with distrust warning) + user, all content inline."""

    TOOL_WRAPPED = "tool_wrapped"
    """Untrusted content in tool results, tool schemas included.
    Deprecated: superseded by TOOL_WRAPPED_V2 which adds 'do not call tools'."""

    TOOL_WRAPPED_V2 = "tool_wrapped_v2"
    """Untrusted content in tool results + explicit 'do not call tools'."""

    TOOL_DISTRUST = "tool_distrust"
    """Tool-wrapped with distrust warning in system prompt and tool schema.
    Deprecated: superseded by TOOL_DISTRUST_V2 which adds 'do not call tools'."""

    TOOL_DISTRUST_V2 = "tool_distrust_v2"
    """Tool-wrapped with distrust warning + explicit 'do not call tools'."""

    @property
    def deprecated(self) -> bool:
        return self in _DEPRECATED_CONDITIONS

    # TODO: add some conditions where we output as a tool call the agent calls or like output as a forced json


# Conditions that exist for backward compat with earlier GSM8K runs
# but should not be added to new tasks.
_DEPRECATED_CONDITIONS: frozenset[Condition] = frozenset({
    Condition.TOOL_WRAPPED,
    Condition.TOOL_DISTRUST,
})

ACTIVE_CONDITIONS: frozenset[Condition] = frozenset(
    c for c in Condition if not c.deprecated
)


@dataclass(frozen=True)
class CandidatePayload:
    """Response content being evaluated or inserted into a condition."""
    response: str



@dataclass(frozen=True)
class EvalCase[TItem]:
    """One concrete candidate to evaluate for a dataset item."""
    case_id: str
    label: str
    item: TItem
    candidate: CandidatePayload
    is_under_attack: bool
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PairedEvalCase[TItem]:
    """A baseline and transformed variant for the same dataset item."""
    pair_id: str
    transform_id: str
    item: TItem
    baseline: CandidatePayload
    variant: CandidatePayload
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseVerdict:
    """Parsed output from a pairwise judge evaluation.

    Stores the full expressiveness of the original judge output (raw_label)
    while providing a normalized decomposition into direction + strength
    that downstream reporting can use uniformly across frameworks.

    Frameworks and their native label sets:
      - Arena-Hard-Auto:  [[A>>B]], [[A>B]], [[A=B]], [[B>A]], [[B>>A]]
      - WildBench:        A++, A+, A=B, B+, B++
      - Prometheus:        A, B  (binary, strength=None)
    """

    direction: PreferenceDirection
    """Which side the judge preferred: 'A', 'B', or 'tie'."""

    strength: PreferenceStrength | None
    """'strong' or 'slight', or None for binary/tie verdicts."""

    raw_label: str
    """The original label string exactly as extracted from judge output,
    before normalization. Preserves full fidelity for per-framework analysis."""

    def a_wins(self) -> bool:
        """True if the judge preferred A (any strength)."""
        return self.direction == "A"

    def b_wins(self) -> bool:
        """True if the judge preferred B (any strength)."""
        return self.direction == "B"

    def is_tie(self) -> bool:
        return self.direction == "tie"



@dataclass(frozen=True)
class PairwiseEvalCase[TItem]:
    """One pairwise comparison to evaluate for a dataset item.

    Unlike single-candidate EvalCase, this holds two candidates (A and B).
    Attack status is tracked in metadata (e.g. ``attacked_position``,
    ``attack_label``) rather than top-level fields — the experiment runner
    knows which cases are attacks based on how they were constructed.
    """
    case_id: str
    label: str
    item: TItem
    candidate_a: CandidatePayload
    candidate_b: CandidatePayload
    metadata: JsonDict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Condition blueprint types — declarative prompt assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolBinding:
    """One tool result in the conversation, bound to a payload value.

    When a ConditionBlueprint includes tool bindings, the assembler creates
    a simulated tool-call / tool-result exchange for each binding. The tool
    result content is pulled from the payload dict keyed by ``binds_to``.
    """

    tool_name: str
    """Function name for the mock tool call (e.g. 'get_candidate_response')."""

    binds_to: str
    """Payload dict key whose value becomes the tool result content."""

    description: str = ""
    """Tool schema description. Used to auto-generate the tools API kwarg."""

    tool_arguments: JsonDict = field(default_factory=dict)
    """Mock arguments serialized into the assistant's tool_call."""


@dataclass(frozen=True)
class ConditionBlueprint:
    """Full specification of how to build messages for one (task, condition).

    Each task registers a ``dict[Condition, ConditionBlueprint]`` mapping
    condition names to blueprints. The shared assembler turns a blueprint +
    payload dict into API-ready ``(messages, extra_api_kwargs)``.

    Template placeholders use ``{name}`` syntax and are filled from the payload
    dict. Values bound to tool results (via ``tool_bindings``) are excluded
    from template substitution — they go into tool result messages instead.

    Structure is determined declaratively:
      - ``system_text is None`` → no system message (baseline-style)
      - ``tool_bindings`` empty → inline condition (system? + user)
      - ``tool_bindings`` present → tool condition (system + user + assistant
        tool_calls + tool results); tool schemas always included
    """

    system_text: str | None
    """System message template, or None for baseline (no system message)."""

    user_text: str
    """User message template with {name} placeholders for payload values."""

    tool_bindings: list[ToolBinding] = field(default_factory=list)
    """Tool results to include. Each binding pulls a value from the payload.
    When present, tool schemas are always included in extra_api_kwargs."""

    tool_choice: str | dict | None = None
    """Optional override for the OpenAI-style `tool_choice` API parameter.

    None (default) → parameter not set; provider uses its own default ("auto"
    when tools are declared). Use "none" to forbid tool calls (prevents the
    model from answering via a tool call when we want a text verdict). Use
    "required" or a dict like `{"type": "function", "function": {"name": ...}}`
    to force a specific tool — useful for future judge-via-tool-call setups.

    Persisted through `extra_api_kwargs` into `request_kwargs`, so it becomes
    part of the cache key automatically."""
