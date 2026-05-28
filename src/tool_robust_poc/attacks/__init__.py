"""Attack registry for judge robustness experiments.

Attacks are adversarial candidate strings designed to fool LLM judges into
giving incorrect verdicts. Organized by mechanism:

  catalog.py    — content-level attacks (empty, injection, mimicry, etc.)
  structural.py — token-level attacks (ChatML/LLaMA breakout)

NOTE: This categorization is a first pass. Boundaries may shift.
"""

from __future__ import annotations

from tool_robust_poc.attacks.catalog import (
    ALL_CONTENT_ATTACKS,
    EMPTY_MINIMAL,
    INSTRUCTION_INJECTION,
    OUTPUT_MIMICRY,
    RAINA_PHRASES,
    REASONING_OPENERS,
    get_content_attacks,
)
from tool_robust_poc.attacks.structural import (
    STRUCTURAL_TOKEN_ATTACKS,
    get_structural_token_attacks,
)

ALL_STATIC_ATTACKS = ALL_CONTENT_ATTACKS + STRUCTURAL_TOKEN_ATTACKS


def get_static_attacks(positive_verdict: str = "CORRECT") -> list[tuple[str, str]]:
    """Return content plus structural attacks for the target verdict."""
    return get_content_attacks(positive_verdict) + get_structural_token_attacks(
        positive_verdict
    )


def get_attack_labels(attacks: list[tuple[str, str]]) -> list[str]:
    """Extract just the labels from an attack list."""
    return [label for label, _ in attacks]
