"""Shared CLI argument patterns for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from fllmingo import DEFAULT_MODEL, PROVIDERS

from tool_robust_poc.conditions_text import DEFAULT_CONDITIONS
from tool_robust_poc.core_types import ATTACK_SET_CHOICES, Condition
from tool_robust_poc.runner_utils import DEFAULT_CONCURRENCY, DEFAULT_MAX_CONCURRENCY

ALL_CONDITION_VALUES = [c.value for c in Condition]

# Canonical model list for cross-model matrix runs. Keep in sync with
# design/model_picking.md. Individual matrix scripts may override via --models.
DEFAULT_MATRIX_MODEL_SPECS: list[str] = [
    "openrouter:gpt-5.4",
    "openrouter:gpt-5.4-mini",
    "openrouter:claude-haiku-4.5",
    "openrouter:google/gemma-4-26b-a4b-it",
    "openrouter:qwen3.5-flash-02-23",
]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard experiment arguments to a parser.

    Scripts can add their own args before or after calling this.
    """
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--provider", default="openai",
        choices=list(PROVIDERS),
        help="API provider (default: %(default)s)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=[c.value for c in DEFAULT_CONDITIONS],
        choices=ALL_CONDITION_VALUES,
        help=f"Prompt conditions to test (default: {[c.value for c in DEFAULT_CONDITIONS]})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--summary-output", type=Path, default=None,
        help="Summary JSON path (defaults next to the raw output)",
    )
    parser.add_argument(
        "--attack-set", default="static",
        choices=ATTACK_SET_CHOICES,
        help="Attack pool to use (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=None,
        help="Max number of dataset items to evaluate after deterministic ordering",
    )
    parser.add_argument(
        "--sample-order", default="hash",
        choices=["hash", "file"],
        help="How to order dataset items before taking a prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-seed", default="tool-robust-poc-v1",
        help="Seed string for deterministic hash ordering",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help=f"Initial concurrent API calls target (default: {DEFAULT_CONCURRENCY}, or 256 for vllm)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=None,
        help=f"Upper bound for adaptive concurrency scaling (default: {DEFAULT_MAX_CONCURRENCY}, or 1024 for vllm)",
    )
    return parser


def conditions_from_args(args: argparse.Namespace) -> tuple[Condition, ...]:
    """Convert string condition args to Condition enum tuple."""
    return tuple(Condition(c) for c in args.conditions)


def add_matrix_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
    default_model_specs: list[str] | None = None,
    default_provider: str = "openrouter",
) -> argparse.ArgumentParser:
    """Add the standard cross-model matrix-runner arguments.

    Covers the flags that are identical across task matrix wrappers:
    --models, --provider, --conditions, --attack-set, --sample-limit,
    --sample-order, --sample-seed, --concurrency, --max-concurrency,
    --output-dir, --matrix-summary-output, --continue-on-error.

    Each task's matrix script adds its own task-specific flags (e.g. --data,
    --baseline-answers) separately.
    """
    specs = default_model_specs or DEFAULT_MATRIX_MODEL_SPECS
    parser.add_argument(
        "--models", nargs="+", default=specs,
        help="Model specs to sweep (use 'provider:model' or bare model names).",
    )
    parser.add_argument(
        "--provider", default=default_provider,
        choices=list(PROVIDERS),
        help="Default provider for bare model names (default: %(default)s)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=[c.value for c in DEFAULT_CONDITIONS],
        choices=ALL_CONDITION_VALUES,
        help=f"Prompt conditions to test (default: {[c.value for c in DEFAULT_CONDITIONS]})",
    )
    parser.add_argument(
        "--attack-set", default="static",
        choices=ATTACK_SET_CHOICES,
        help="Attack pool to use (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=None,
        help="Max dataset items per model run (after deterministic ordering).",
    )
    parser.add_argument(
        "--sample-order", default="hash", choices=["hash", "file"],
        help="How to order dataset items before taking a prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-seed", default="tool-robust-poc-v1",
        help="Seed string for deterministic hash ordering (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help=(
            f"Initial per-model concurrency (default: {DEFAULT_CONCURRENCY}, "
            f"or 256 for vllm)"
        ),
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=None,
        help=(
            f"Upper bound for adaptive per-model concurrency "
            f"(default: {DEFAULT_MAX_CONCURRENCY}, or 1024 for vllm)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_dir,
        help="Directory for per-model raw and summary outputs.",
    )
    parser.add_argument(
        "--matrix-summary-output", type=Path, default=None,
        help="Aggregated matrix summary path (default: <output-dir>/matrix_summary.json.gz)",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going if one model run fails.",
    )
    parser.add_argument(
        "--controls-only", action="store_true",
        help="Skip the attack pool and only run controls (cheap "
             "non-adversarial sweep for equivalence checks).",
    )
    return parser
