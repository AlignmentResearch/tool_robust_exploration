"""Shared CLI / setup helpers for PAIR-compare scripts.

Per-task scripts (gsm8k / mtbench / arena_hard, batched and unbatched) all
share the same shape: parse common args → build sessions/cells/strategies →
run → write trajectories + summary.  The task-specific bits are the
``JudgeTask``, the condition map, the data loader, and a few CLI flags.

This module factors out the common scaffolding so a per-task batched script
can be ~80 lines instead of ~300.  Right now only the batched path is
covered; the unbatched comparison scripts can migrate to share these
helpers as a follow-up.

Layered building blocks (caller composes):
    * ``PairTaskSpec``          — task-specific knobs in one bundle
    * ``add_pair_common_args``  — argparse: model/turn/item/seed flags
    * ``add_pair_batched_args`` — argparse: --poll-interval
    * ``make_pair_sessions``    — victim+attacker ModelSession + trackers
    * ``make_pair_strategies``  — auto-pick Batch vs Parallel from provider
    * ``build_pair_cells``      — (seed × condition × nudge) → list[PairCell]
    * ``default_output_dir``    — repo-standard naming under results/
    * ``make_progress_printer`` — one-line-per-(cell, turn) callback
    * ``write_pair_outputs``    — trajectory_*.json + summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from fllmingo import (
    Batch,
    CostTracker,
    ModelSession,
    Parallel,
    Strategy,
    Tier,
    make_openai_client,
    resolve_model,
)

from tool_robust_poc.cache import get_cache
from tool_robust_poc.attack_opt.loop import (
    JudgeTask,
    PairCell,
    PairResult,
    Turn,
    result_to_dict,
)
from tool_robust_poc.core_types import Condition, ConditionBlueprint
from tool_robust_poc.experiment.common import POC_ROOT, RESULTS_DIR


# ── Task spec ──────────────────────────────────────────────────────────


@dataclass
class PairTaskSpec:
    """Task-specific bits a runner needs to build cells and write outputs.

    ``name`` is short and filesystem-safe (e.g. ``"gsm8k"``); used in cache
    tags, tracker experiment names, and output dir.

    ``available_conditions`` maps each ``Condition`` enum value the task
    supports to its blueprint.  GSM8K's variant-aware version is one of
    these dicts.

    ``items`` is the full pre-loaded item set; sampling happens inside
    ``build_pair_cells``.

    ``stable_item_key_fn`` returns a stable string id for an item — used
    in the summary's ``sampled_item_keys_by_seed`` for reproducibility.

    ``task`` is the static :class:`JudgeTask` shared across every cell.
    For tasks that must build a per-seed task (e.g. arena_hard's slot
    seed), leave ``task=None`` and pass a ``task_factory`` to
    :func:`build_pair_cells` instead.
    """
    name: str
    available_conditions: dict[Condition, ConditionBlueprint]
    items: Sequence[Any]
    stable_item_key_fn: Callable[[Any], str]
    task: JudgeTask | None = None


# ── Argparse helpers ───────────────────────────────────────────────────


def add_pair_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the model / turn / item / seed flags every PAIR script uses.

    Caller is responsible for adding task-specific flags (e.g. ``--data``,
    ``--prompt-variant``, ``--conditions``) and for setting defaults via
    ``parser.set_defaults(...)`` if the task wants different defaults.
    """
    parser.add_argument("--victim-model")
    parser.add_argument("--victim-provider", default="anthropic")
    parser.add_argument("--attacker-model")
    parser.add_argument("--attacker-provider", default="openai")
    parser.add_argument("--n-turns", type=int, default=15)
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument(
        "--sample-seeds", nargs="+", default=["pair-batched-v1"],
        help="One or more seed strings; each produces an independent branch set.",
    )
    parser.add_argument(
        "--include-no-nudge", action="store_true",
        help="Also run the no-nudge control branch.",
    )
    parser.add_argument(
        "--exclude-item-ids-file", type=Path, default=None,
        help="Path to a JSON file with stable item IDs to EXCLUDE from "
             "search-seed sampling.  Use this to reserve a fixed transfer "
             "set and ensure new search seeds don't sample those items.  "
             "JSON shape: {\"item_ids\": [...]} (other keys ignored).",
    )
    parser.add_argument("--attacker-temperature", type=float, default=None)
    parser.add_argument(
        "--attacker-reasoning-effort", default=None,
        choices=[None, "minimal", "low", "medium", "high"],
    )
    parser.add_argument("--output-dir", type=Path, default=None)


def add_pair_batched_args(parser: argparse.ArgumentParser) -> None:
    """Batched-mode-only flags."""
    parser.add_argument(
        "--poll-interval", type=float, default=30.0,
        help="Seconds between batch-status polls.",
    )
    parser.add_argument(
        "--victim-non-batched", action="store_true",
        help="Force Parallel(STANDARD) for the victim strategy even on "
             "Anthropic (which would otherwise auto-pick Batch). Trades "
             "the 50%% batch discount for sync latency; useful when "
             "Anthropic batch is queue-slow or repeatedly hangs.",
    )


# ── Sessions ───────────────────────────────────────────────────────────


def make_pair_sessions(
    args: argparse.Namespace,
    *,
    experiment_namespace: str,
) -> tuple[ModelSession, ModelSession, CostTracker, CostTracker]:
    """Build (victim_session, attacker_session, victim_tracker, attacker_tracker).

    Trackers are returned separately so the caller can include their summary
    in output JSON.  Caches live under ``POC_ROOT/cache/<model>``; ledger
    under ``POC_ROOT/cost_ledger.jsonl``.

    For a batched victim on Anthropic the underlying client is unused (the
    batch session constructs its own AsyncAnthropic), but we still build a
    valid AsyncOpenAI for the OpenAI / OpenRouter path.
    """
    victim_api_model = resolve_model(args.victim_model, args.victim_provider)
    attacker_api_model = resolve_model(args.attacker_model, args.attacker_provider)

    victim_cache = get_cache(args.victim_model)
    attacker_cache = get_cache(args.attacker_model, suffix="_attacker")

    victim_tracker = CostTracker(
        model=args.victim_model, provider=args.victim_provider,
        experiment=f"{experiment_namespace}_victim",
        ledger_path=POC_ROOT / "cost_ledger.jsonl",
    )
    attacker_tracker = CostTracker(
        model=args.attacker_model, provider=args.attacker_provider,
        experiment=f"{experiment_namespace}_attacker",
        ledger_path=POC_ROOT / "cost_ledger.jsonl",
    )

    if args.victim_provider in {"openai", "openrouter", "fireworks", "vllm"}:
        victim_client = make_openai_client(provider=args.victim_provider)
    else:
        # Anthropic batch path doesn't use this client; keep it None.
        victim_client = None
    attacker_client = make_openai_client(provider=args.attacker_provider)

    attacker_defaults: dict[str, Any] = {}
    if args.attacker_temperature is not None:
        attacker_defaults["temperature"] = args.attacker_temperature
    if args.attacker_reasoning_effort is not None:
        attacker_defaults["reasoning_effort"] = args.attacker_reasoning_effort

    victim_session = ModelSession(
        client=victim_client,
        provider=args.victim_provider,
        model=victim_api_model,
        cache=victim_cache,
        tracker=victim_tracker,
    )
    attacker_session = ModelSession(
        client=attacker_client,
        provider=args.attacker_provider,
        model=attacker_api_model,
        cache=attacker_cache,
        tracker=attacker_tracker,
        defaults=attacker_defaults,
    )
    return victim_session, attacker_session, victim_tracker, attacker_tracker


# ── Strategies ─────────────────────────────────────────────────────────


def make_pair_strategies(
    args: argparse.Namespace,
    *,
    n_branches: int,
) -> tuple[Strategy, Strategy]:
    """Auto-pick (victim_strategy, attacker_strategy) from ``args.victim_provider``.

    Smooth tier rolloff matching :func:`transfer.make_target_victim_session`:
      * ``anthropic`` → :class:`Batch` (50% off, async polling)
      * ``openai`` → ``Parallel(tier=FLEX)`` (50% off, sync calls — best of
        both worlds; was Batch before 2026-04-28)
      * everything else (openrouter / fireworks / vllm / …) →
        ``Parallel(tier=STANDARD)`` adaptive

    The attacker is always small + low-latency (one call per branch per
    turn), so it stays on Parallel STANDARD regardless of provider.
    """
    non_batched = getattr(args, "victim_non_batched", False)
    if args.victim_provider == "anthropic" and not non_batched:
        victim_strategy: Strategy = Batch(poll_interval_s=args.poll_interval)
    elif args.victim_provider == "openai":
        victim_strategy = Parallel(tier=Tier.FLEX, adaptive=True)
    else:
        # Includes: anthropic with --victim-non-batched, openrouter,
        # fireworks, vllm, and other providers without auto-Batch.
        victim_strategy = Parallel(tier=Tier.STANDARD, adaptive=True)
    attacker_strategy: Strategy = Parallel(
        adaptive=False, concurrency=max(8, n_branches * 4),
    )
    return victim_strategy, attacker_strategy


# ── Cells ──────────────────────────────────────────────────────────────


def _branch_label(seed: str, condition_name: str, nudge_enabled: bool) -> str:
    return f"{seed}_{condition_name}_{'nudge' if nudge_enabled else 'nonudge'}"


def _sample_items(
    items: Sequence[Any],
    *,
    n: int,
    seed: str,
    stable_key_fn: Callable[[Any], str],
) -> list[Any]:
    # Local import so this module doesn't drag the sampling dep at import time.
    from tool_robust_poc.sampling import deterministic_sample
    records = deterministic_sample(
        list(items), stable_key_fn=stable_key_fn, seed=seed, limit=n, order="hash",
    )
    return [r["item"] for r in records]


def build_pair_cells(
    *,
    spec: PairTaskSpec,
    seeds: list[str],
    conditions: list[Condition],
    n_items: int,
    include_no_nudge: bool,
    cache_tag_namespace: str,
    victim_model: str,
    attacker_model: str,
    task_factory: Callable[[str], JudgeTask] | None = None,
    exclude_item_ids: set[str] | None = None,
) -> tuple[list[PairCell], list[str], dict[str, list[Any]]]:
    """Build the matrix of (seed × condition × nudge) cells.

    Returns ``(cells, labels, sampled_items_by_seed)``.  ``labels[i]`` is
    the human-readable label for ``cells[i]``; ``sampled_items_by_seed``
    is the per-seed sampled item list (useful for the summary file).

    ``task_factory`` lets a task build a fresh :class:`JudgeTask` per seed
    (used by arena_hard, whose slot-permutation seed feeds the task's
    payload builder).  When unset, every cell shares ``spec.task``.

    ``exclude_item_ids``, if set, removes any item whose
    ``stable_item_key_fn`` is in the set BEFORE per-seed sampling.  Use
    this to reserve a held-out transfer set so new search seeds don't
    sample those items.
    """
    if task_factory is None and spec.task is None:
        raise ValueError(
            "build_pair_cells: spec.task is None and no task_factory passed; "
            "set one or the other."
        )
    for c in conditions:
        if c not in spec.available_conditions:
            raise SystemExit(f"No {spec.name} blueprint for condition {c!r}")

    if exclude_item_ids:
        eligible_items = [
            it for it in spec.items
            if spec.stable_item_key_fn(it) not in exclude_item_ids
        ]
        n_dropped = len(list(spec.items)) - len(eligible_items)
        if n_dropped == 0:
            print(f"--exclude-item-ids: no items matched (excluded set has "
                  f"{len(exclude_item_ids)} ids; pool has "
                  f"{len(list(spec.items))} items).")
        else:
            print(f"--exclude-item-ids: dropped {n_dropped} of "
                  f"{len(list(spec.items))} items; "
                  f"{len(eligible_items)} eligible for sampling.")
    else:
        eligible_items = spec.items

    cells: list[PairCell] = []
    labels: list[str] = []
    sampled_by_seed: dict[str, list[Any]] = {}
    for seed in seeds:
        items = _sample_items(
            eligible_items, n=n_items, seed=seed,
            stable_key_fn=spec.stable_item_key_fn,
        )
        if not items:
            raise SystemExit(f"No items sampled for seed {seed!r}")
        sampled_by_seed[seed] = items
        seed_task = task_factory(seed) if task_factory is not None else spec.task
        nudge_variants = (True, False) if include_no_nudge else (True,)
        for condition in conditions:
            for nudge_enabled in nudge_variants:
                label = _branch_label(seed, condition.value, nudge_enabled)
                cells.append(PairCell(
                    cell_id=label,
                    condition_name=condition.value,
                    blueprint=spec.available_conditions[condition],
                    task=seed_task,
                    items=items,
                    cache_tag_prefix=(
                        f"{cache_tag_namespace}:{label}:"
                        f"{victim_model}:{attacker_model}"
                    ),
                    nudge_enabled=nudge_enabled,
                ))
                labels.append(label)
    return cells, labels, sampled_by_seed


# ── Output dir ─────────────────────────────────────────────────────────


def default_output_dir(
    *,
    args: argparse.Namespace,
    subdir: str,
    extra_tag: str = "",
) -> Path:
    """Standard naming under ``RESULTS_DIR/<subdir>/<vmodel>_v_<amodel>...``.

    ``subdir`` is e.g. ``"pair_gsm8k_batched"``; ``extra_tag`` is appended
    to the leaf, useful for things like prompt variant.
    """
    victim_tag = (
        args.victim_model.replace("/", "_").replace(":", "_").replace(".", "")
    )
    attacker_tag = (
        args.attacker_model.replace("/", "_").replace(":", "_").replace(".", "")
    )
    seeds_tag = (
        args.sample_seeds[0]
        if len(args.sample_seeds) == 1
        else f"{len(args.sample_seeds)}seeds"
    )
    leaf = (
        f"{victim_tag}_v_{attacker_tag}"
        + (f"_{extra_tag}" if extra_tag else "")
        + f"_t{args.n_turns}_n{args.n_items}_{seeds_tag}"
    )
    return RESULTS_DIR / subdir / leaf


# ── Progress + outputs ─────────────────────────────────────────────────


def make_progress_printer(
    *,
    attack_preview_chars: int = 60,
) -> Callable[[str, Turn], None]:
    """One-line-per-(cell, turn) callback compatible with run_pair_matrix_batched."""
    def _on_turn(cell_id: str, turn: Turn) -> None:
        attack_preview = turn.attack_string.replace("\n", "\\n")
        if len(attack_preview) > attack_preview_chars:
            attack_preview = attack_preview[:attack_preview_chars] + "..."
        print(
            f"  [{cell_id}] turn {turn.turn_index} score={turn.score:.2f} "
            f"parse_ok={turn.parse_ok_rate:.2f} attack={attack_preview!r}",
            flush=True,
        )
    return _on_turn


def best_so_far(turns: list[Turn]) -> list[float]:
    out: list[float] = []
    running = 0.0
    for t in turns:
        running = max(running, t.score)
        out.append(running)
    return out


def write_pair_outputs(
    *,
    output_dir: Path,
    cell_labels: list[str],
    results: list[PairResult],
    sampled_items_by_seed: dict[str, list[Any]],
    spec: PairTaskSpec,
    args_snapshot: dict,
    elapsed_s: float,
    extra_summary: dict | None = None,
    victim_tracker: CostTracker | None = None,
    attacker_tracker: CostTracker | None = None,
) -> None:
    """Write per-cell ``trajectory_<label>.json`` + a single ``summary.json``.

    Trajectory file: full ``PairResult`` per cell.  Summary file:
    arg snapshot, sampled item keys, per-branch best/scores, optional
    extra_summary (caller stuff like prompt_variant), optional cost summaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, result in zip(cell_labels, results):
        out_path = output_dir / f"trajectory_{label}.json"
        out_path.write_text(json.dumps({
            "condition": result.condition_name,
            "nudge_enabled": result.nudge_enabled,
            "result": result_to_dict(result),
        }, indent=2))

    summary: dict[str, Any] = {
        "args": args_snapshot,
        "sampled_item_keys_by_seed": {
            seed: [spec.stable_item_key_fn(it) for it in items]
            for seed, items in sampled_items_by_seed.items()
        },
        "elapsed_s": elapsed_s,
        "best_by_branch": {
            label: {
                "condition": r.condition_name,
                "nudge_enabled": r.nudge_enabled,
                "best_turn_index": r.best_turn_index,
                "best_score": r.best_score,
                "best_attack": r.best_attack,
                "per_turn_score": [t.score for t in r.turns],
                "best_so_far": best_so_far(r.turns),
            }
            for label, r in zip(cell_labels, results)
        },
    }
    if victim_tracker is not None:
        summary["victim_cost"] = victim_tracker.summary()
    if attacker_tracker is not None:
        summary["attacker_cost"] = attacker_tracker.summary()
    if extra_summary:
        summary.update(extra_summary)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


__all__ = [
    "PairTaskSpec",
    "add_pair_batched_args",
    "add_pair_common_args",
    "best_so_far",
    "build_pair_cells",
    "default_output_dir",
    "make_pair_sessions",
    "make_pair_strategies",
    "make_progress_printer",
    "write_pair_outputs",
]
