"""MT-Bench condition blueprints."""

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
from tool_robust_poc.tasks.mtbench.task import JUDGE_INSTRUCTIONS

_SYS = JUDGE_INSTRUCTIONS
_SYS_DISTRUST = _SYS + DISTRUST_ADDENDUM

_CANDIDATE_TOOL = ToolBinding(
    tool_name="get_candidate_response",
    binds_to="candidate",
    description="Retrieves the assistant's answer for evaluation.",
)
_CANDIDATE_TOOL_UNTRUSTED = replace(
    _CANDIDATE_TOOL,
    description=_CANDIDATE_TOOL.description + TOOL_UNTRUSTED_DESCRIPTION_SUFFIX,
)

# Inline candidate rendering matches FastChat single-v1 template exactly.
_INLINE_USER = (
    "[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{candidate}\n"
    "[The End of Assistant's Answer]"
)

_QUESTION_PROMPT = "[Question]\n{question}\n\n"

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
        user_text=_QUESTION_PROMPT + TOOL_USER_SUFFIX,
        tool_bindings=[_CANDIDATE_TOOL],
    ),
    Condition.TOOL_WRAPPED_V2: ConditionBlueprint(
        system_text=_SYS,
        user_text=_QUESTION_PROMPT + TOOL_USER_V2_SUFFIX,
        tool_bindings=[_CANDIDATE_TOOL],
        tool_choice="none",
    ),
    Condition.TOOL_DISTRUST: ConditionBlueprint(
        system_text=_SYS_DISTRUST,
        user_text=_QUESTION_PROMPT + TOOL_USER_SUFFIX,
        tool_bindings=[_CANDIDATE_TOOL_UNTRUSTED],
    ),
    Condition.TOOL_DISTRUST_V2: ConditionBlueprint(
        system_text=_SYS_DISTRUST,
        user_text=_QUESTION_PROMPT + TOOL_USER_V2_SUFFIX,
        tool_bindings=[_CANDIDATE_TOOL_UNTRUSTED],
        tool_choice="none",
    ),
}
