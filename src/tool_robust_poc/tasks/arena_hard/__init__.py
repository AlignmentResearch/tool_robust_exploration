"""Arena-Hard-Auto pairwise preference judge task."""

from tool_robust_poc.tasks.arena_hard.attacks import TASK_ATTACKS
from tool_robust_poc.tasks.arena_hard.conditions import CONDITIONS
from tool_robust_poc.tasks.arena_hard.task import (
    DEFAULT_BASELINE_ANSWERS_PATH,
    DEFAULT_DATA_PATH,
    JUDGE_INSTRUCTIONS,
    TASK,
    ArenaHardItem,
    get_attacks,
    load_baseline_answers,
    load_data,
    make_payload,
    parse_preference,
    stable_item_key,
)

__all__ = [
    "ArenaHardItem",
    "CONDITIONS",
    "DEFAULT_BASELINE_ANSWERS_PATH",
    "DEFAULT_DATA_PATH",
    "JUDGE_INSTRUCTIONS",
    "TASK",
    "TASK_ATTACKS",
    "get_attacks",
    "load_baseline_answers",
    "load_data",
    "make_payload",
    "parse_preference",
    "stable_item_key",
]
