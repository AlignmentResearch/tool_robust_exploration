"""Arena-Hard-Auto dataset types, loading, baseline answers, and payload building."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tool_robust_poc.core_types import (
    AttackSet,
    PairwiseVerdict,
)
from tool_robust_poc.tasks import PairwiseJudgeSpec

from tool_robust_poc.tasks.arena_hard.attacks import TASK_ATTACKS

# ---------------------------------------------------------------------------
# Vendored data location
# ---------------------------------------------------------------------------
_VENDORED_DIR = Path(__file__).parent / "from_arena_hard"
DEFAULT_DATA_PATH = _VENDORED_DIR / "question.jsonl"
DEFAULT_BASELINE_ANSWERS_PATH = _VENDORED_DIR / "o3_mini_answers.jsonl"

# ---------------------------------------------------------------------------
# Judge prompt — loaded from vendored file.
# ---------------------------------------------------------------------------

JUDGE_INSTRUCTIONS = (_VENDORED_DIR / "judge_prompt.txt").read_text().strip()

# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

# Output parser — matches Arena-Hard-Auto gen_judgment.get_score exactly:
#   patterns (tried in order):
#     primary : \[\[([AB<>=]+)\]\]
#     fallback: \[([AB<>=]+)\]
#   both uppercased, empties filtered, last-match wins.
#   last-match: opposite of FastChat MT-Bench — this is what the Arena-Hard
#   reference implementation does, not a choice we've made.
_PREFERENCE_RE_PRIMARY  = re.compile(r"\[\[([AB<>=]+)\]\]")
_PREFERENCE_RE_FALLBACK = re.compile(r"\[([AB<>=]+)\]")

_LABEL_MAP: dict[str, PairwiseVerdict] = {
    "A>>B": PairwiseVerdict(direction="A", strength="strong", raw_label="A>>B"),
    "A>B":  PairwiseVerdict(direction="A", strength="slight", raw_label="A>B"),
    "A=B":  PairwiseVerdict(direction="tie", strength=None, raw_label="A=B"),
    "B>A":  PairwiseVerdict(direction="B", strength="slight", raw_label="B>A"),
    "B>>A": PairwiseVerdict(direction="B", strength="strong", raw_label="B>>A"),
}


def parse_preference(text: str) -> PairwiseVerdict | None:
    """Parse Arena-Hard preference — reference-faithful: upper, last-match, fallback."""
    if not text:
        return None
    upper = text.upper()
    for pattern in (_PREFERENCE_RE_PRIMARY, _PREFERENCE_RE_FALLBACK):
        matches = [m for m in pattern.findall(upper) if m]
        if matches:
            return _LABEL_MAP.get(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------

TASK = PairwiseJudgeSpec(
    judge_instructions=JUDGE_INSTRUCTIONS,
    parse_preference=parse_preference,
    task_attacks=TASK_ATTACKS,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaHardItem:
    """Trusted dataset content for one Arena-Hard evaluation item."""

    uid: str
    category: str
    subcategory: str
    prompt: str


def load_data(path: Path | None = None) -> list[ArenaHardItem]:
    """Load Arena-Hard questions from JSONL (hard_prompt only)."""
    if path is None:
        path = DEFAULT_DATA_PATH
    items: list[ArenaHardItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                items.append(
                    ArenaHardItem(
                        uid=raw["uid"],
                        category=raw["category"],
                        subcategory=raw.get("subcategory", ""),
                        prompt=raw["prompt"],
                    )
                )
    return items


def load_baseline_answers(
    path: Path | None = None,
) -> dict[str, str]:
    """Load vendored o3-mini baseline answers.

    Returns:
        Mapping from question uid to answer text.
    """
    if path is None:
        path = DEFAULT_BASELINE_ANSWERS_PATH
    answers: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                uid = raw["uid"]
                # o3-mini answers have messages[1]['content'] as dict with 'answer' key
                assistant_msg = raw["messages"][1]["content"]
                if isinstance(assistant_msg, dict):
                    answers[uid] = assistant_msg["answer"]
                elif isinstance(assistant_msg, str):
                    answers[uid] = assistant_msg
                else:
                    raise ValueError(
                        f"Unexpected assistant content type for uid {uid}: "
                        f"{type(assistant_msg)}"
                    )
    return answers


def make_payload(
    item: ArenaHardItem,
    candidate_a: str,
    candidate_b: str,
) -> dict[str, str]:
    """Build the payload dict for condition assembly.

    For pairwise tasks, the payload has three keys: question, candidate_a, candidate_b.
    """
    return {
        "question": item.prompt,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
    }


def stable_item_key(item: ArenaHardItem) -> str:
    """Return a stable per-item key for deterministic ordering."""
    return f"arena_hard_{item.uid}"


def get_attacks(attack_set: AttackSet) -> list[tuple[str, str]]:
    """Return Arena-Hard attacks."""
    from tool_robust_poc.attacks import EMPTY_MINIMAL, RAINA_PHRASES, REASONING_OPENERS
    from tool_robust_poc.attacks.structural import STRUCTURAL_TOKEN_ATTACKS

    content_attacks = EMPTY_MINIMAL + REASONING_OPENERS + RAINA_PHRASES + TASK_ATTACKS
    if attack_set == "content":
        return content_attacks
    if attack_set == "static":
        return content_attacks + STRUCTURAL_TOKEN_ATTACKS
    raise ValueError(f"Unknown attack_set: {attack_set!r}")
