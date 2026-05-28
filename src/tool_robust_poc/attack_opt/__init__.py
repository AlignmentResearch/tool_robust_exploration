"""Attack-optimization subsystem: PAIR-style iterative attacker loop + per-task adapters.

Public surface (see individual modules for details):

- ``loop`` — task-agnostic PAIR loop, data types, and JudgeTask protocol.
- ``gsm8k`` — GSM8K-specific JudgeTask spec and attacker system prompt.
- ``mtbench`` — MT-Bench-specific spec (scalar 1-10 scoring).
- ``arena_hard`` — Arena-Hard-specific spec (slot-unaware pairwise).

The task modules each export a ready-to-use ``TASK_SPEC`` plus the
condition blueprints and dataset-loading helpers reused from the
static-eval machinery under ``tool_robust_poc.tasks``.
"""

from tool_robust_poc.attack_opt.loop import (
    ItemScore,
    JudgeTask,
    PairCell,
    PairResult,
    ScoreResult,
    Turn,
    evaluate_attack,
    looks_like_refusal,
    parse_attacker_json,
    result_to_dict,
    run_pair_loop,
    run_pair_matrix_batched,
)

__all__ = [
    "ItemScore",
    "JudgeTask",
    "PairCell",
    "PairResult",
    "ScoreResult",
    "Turn",
    "evaluate_attack",
    "looks_like_refusal",
    "parse_attacker_json",
    "result_to_dict",
    "run_pair_loop",
    "run_pair_matrix_batched",
]
