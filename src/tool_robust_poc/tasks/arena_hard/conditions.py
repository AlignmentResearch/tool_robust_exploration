"""Arena-Hard condition blueprints.

Both candidates are untrusted in tool conditions — each gets its own tool
binding with distinct names. Skips deprecated TOOL_DISTRUST (v1).
"""

from dataclasses import replace

from tool_robust_poc.conditions_text import (
    DISTRUST_ADDENDUM,
    TOOL_UNTRUSTED_DESCRIPTION_SUFFIX,
    TOOL_USER_SUFFIX,
    TOOL_USER_V2_SUFFIX,
)
from tool_robust_poc.core_types import (
    Condition,
    ConditionBlueprint,
    ToolBinding,
)
from tool_robust_poc.tasks.arena_hard.task import JUDGE_INSTRUCTIONS

_SYS = JUDGE_INSTRUCTIONS
_SYS_DISTRUST = _SYS + DISTRUST_ADDENDUM

# ---------------------------------------------------------------------------
# Tool bindings — one per candidate (both untrusted in tool conditions)
# ---------------------------------------------------------------------------

_CANDIDATE_A_TOOL = ToolBinding(
    tool_name="get_candidate_a_response",
    binds_to="candidate_a",
    description="Retrieves Assistant A's answer for evaluation.",
)
_CANDIDATE_A_TOOL_UNTRUSTED = replace(
    _CANDIDATE_A_TOOL,
    description=_CANDIDATE_A_TOOL.description + TOOL_UNTRUSTED_DESCRIPTION_SUFFIX,
)

_CANDIDATE_B_TOOL = ToolBinding(
    tool_name="get_candidate_b_response",
    binds_to="candidate_b",
    description="Retrieves Assistant B's answer for evaluation.",
)
_CANDIDATE_B_TOOL_UNTRUSTED = replace(
    _CANDIDATE_B_TOOL,
    description=_CANDIDATE_B_TOOL.description + TOOL_UNTRUSTED_DESCRIPTION_SUFFIX,
)

# ---------------------------------------------------------------------------
# Inline user text — matches Arena-Hard prompt_template delimiters
# ---------------------------------------------------------------------------

_INLINE_USER = (
    "<|User Prompt|>\n{question}\n\n"
    "<|The Start of Assistant A's Answer|>\n{candidate_a}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{candidate_b}\n"
    "<|The End of Assistant B's Answer|>"
)

_QUESTION_PROMPT = "<|User Prompt|>\n{question}\n\n"

# Tool conditions: question inline, both candidates in tool results
_TOOL_USER = _QUESTION_PROMPT + (
    "The candidate responses will be provided in the tools "
    "(one tool per assistant). " + TOOL_USER_SUFFIX
)
_TOOL_USER_V2 = _QUESTION_PROMPT + (
    "The candidate responses will be provided in the tools "
    "(one tool per assistant). " + TOOL_USER_V2_SUFFIX
)

CONDITIONS: dict[Condition, ConditionBlueprint] = {
    Condition.BASELINE: ConditionBlueprint(
        system_text=None,
        user_text=f"{_SYS}\n\n{_INLINE_USER}",
    ),
    Condition.MULTI_MSG: ConditionBlueprint(
        system_text=_SYS,
        user_text=_INLINE_USER,
    ),
    Condition.SYSTEM_DISTRUST: ConditionBlueprint(
        system_text=_SYS_DISTRUST,
        user_text=_INLINE_USER,
    ),
    Condition.TOOL_WRAPPED: ConditionBlueprint(
        system_text=_SYS,
        user_text=_TOOL_USER,
        tool_bindings=[_CANDIDATE_A_TOOL, _CANDIDATE_B_TOOL],
    ),
    Condition.TOOL_WRAPPED_V2: ConditionBlueprint(
        system_text=_SYS,
        user_text=_TOOL_USER_V2,
        tool_bindings=[_CANDIDATE_A_TOOL, _CANDIDATE_B_TOOL],
        tool_choice="none",
    ),
    Condition.TOOL_DISTRUST_V2: ConditionBlueprint(
        system_text=_SYS_DISTRUST,
        user_text=_TOOL_USER_V2,
        tool_bindings=[_CANDIDATE_A_TOOL_UNTRUSTED, _CANDIDATE_B_TOOL_UNTRUSTED],
        tool_choice="none",
    ),
}
