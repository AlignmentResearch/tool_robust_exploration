"""Shared infrastructure for attack-transfer evaluations.

DRY counterpart to :mod:`tool_robust_poc.attack_opt.runner` for transfer
evals. Used by ``scripts/run_attack_transfer.py`` with task-specific
dispatchers (gsm8k / mtbench / arena_hard).

A transfer eval has the following shape:

1. **Harvest** attacks from one or more PAIR-search output directories
   (``trajectory_*.json`` files), tagging each with its source branch
   (= a (source_dir, seed, condition, nudge) tuple — one independent
   PAIR run).
2. **Select** a subset of attacks per a strategy:
   * ``pool_all`` — top-K globally
   * ``per_condition`` — top-K per source condition
   * ``per_branch`` — top-K per (source_dir × seed × condition × nudge),
     i.e., top-K per the matrix's "branches".  This matches the matrix
     table's variance structure.
3. **Evaluate** each (attack × target_condition × held-out item) tuple
   on a target victim.  Optionally restrict to the diagonal where each
   attack only transfers to its source condition (``diagonal_only``).
4. **Aggregate** per (target_victim × target_condition) into a heatmap;
   per-attack into a CSV / markdown dossier.

The evaluation core uses fllmingo's :func:`run_many` so the same
session/strategy machinery the matrix orchestrator uses (Anthropic /
OpenAI batch on supported providers, Parallel on the rest) carries
through to transfer too.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from fllmingo import (
    Batch,
    CompletedRequest,
    CompletionRequest,
    CostTracker,
    ModelSession,
    Parallel,
    ResponseCache,
    Strategy,
    Tier,
    make_openai_client,
    resolve_model,
    run_many,
)

from tool_robust_poc.cache import get_cache
from tool_robust_poc.attack_opt.loop import ItemScore, JudgeTask
from tool_robust_poc.conditions_building import assemble_messages
from tool_robust_poc.core_types import Condition, ConditionBlueprint
from tool_robust_poc.experiment.common import POC_ROOT, RESULTS_DIR
from tool_robust_poc.sampling import deterministic_sample, stable_digest


# ── Harvest ────────────────────────────────────────────────────────────


@dataclass
class HarvestedAttack:
    """One attack discovered during a PAIR search, with full source provenance."""
    attack_id: str
    attack_string: str
    source_dir_label: str  # source-dir basename — distinguishes attacker arms
    source_branch_label: str  # trajectory_<seed>_<cond>_<nudge> stem
    source_condition: str
    source_nudge_enabled: bool
    source_seed: str
    source_turn_index: int
    source_score: float

    def as_dict(self) -> dict:
        return asdict(self)


def _attack_id(attack_string: str) -> str:
    return stable_digest(attack_string, length=10)


def branch_key(a: HarvestedAttack) -> str:
    """One independent PAIR run: (source_dir × seed × condition × nudge).

    The matrix runs use different ``source_dir`` names per attacker
    (``gsm8k_kimi`` vs ``gsm8k_gemini3``), so this key cleanly separates
    branches across attackers as well as seeds — matching the matrix's
    "6 branches per cell" structure (3 seeds × 2 attackers).
    """
    nudge = "nudge" if a.source_nudge_enabled else "nonudge"
    return f"{a.source_dir_label}::{a.source_seed}::{a.source_condition}::{nudge}"


def _harvest_from_trajectory(
    path: Path, *, seed: str, source_dir_label: str,
) -> list[HarvestedAttack]:
    doc = json.loads(path.read_text())
    r = doc["result"]
    cond = r["condition_name"]
    nudge = bool(r.get("nudge_enabled", True))
    branch_label = path.stem.removeprefix("trajectory_")
    out: list[HarvestedAttack] = []
    for t in r["turns"]:
        attack = t["attack_string"]
        if not attack:
            continue
        out.append(HarvestedAttack(
            attack_id=_attack_id(attack),
            attack_string=attack,
            source_dir_label=source_dir_label,
            source_branch_label=branch_label,
            source_condition=cond,
            source_nudge_enabled=nudge,
            source_seed=seed,
            source_turn_index=int(t["turn_index"]),
            source_score=float(t["score"]),
        ))
    return out


def harvest_all_sources(source_dirs: Sequence[Path]) -> list[HarvestedAttack]:
    """Walk every ``trajectory_*.json`` under each source dir and harvest attacks.

    Each attack carries its source ``source_dir.name`` so per-branch grouping
    can distinguish across-attacker arms even when seeds collide.
    """
    attacks: list[HarvestedAttack] = []
    for sd in source_dirs:
        if not sd.exists():
            raise FileNotFoundError(f"--source-dir does not exist: {sd}")
        label = sd.name
        for path in sorted(sd.glob("trajectory_*.json")):
            stem = path.stem.removeprefix("trajectory_")
            parts = stem.split("_", 1)
            seed = parts[0] if parts else ""
            attacks.extend(_harvest_from_trajectory(
                path, seed=seed, source_dir_label=label,
            ))
    return attacks


def collect_source_item_keys(source_dirs: Sequence[Path]) -> set[str]:
    """Union of every item key the source PAIR runs sampled (for held-out exclusion).

    Reads each source dir's ``summary.json`` (written by
    ``run_pair_matrix.py``).  Missing files are ignored; the caller is
    responsible for failing visibly if exclusion is requested but no keys
    were collected.
    """
    keys: set[str] = set()
    for sd in source_dirs:
        summary = sd / "summary.json"
        if not summary.exists():
            continue
        doc = json.loads(summary.read_text())
        for seed_keys in doc.get("sampled_item_keys_by_seed", {}).values():
            keys.update(seed_keys)
    return keys


def collect_source_item_keys_per_seed(
    source_dirs: Sequence[Path],
) -> dict[tuple[str, str], set[str]]:
    """Return ``{(arm_name, seed): set_of_item_keys}`` keyed by the source
    dir name and seed name.

    Used by per-seed held-out evaluation: an attack from a given (arm,
    seed) is evaluated on items disjoint from THAT seed's search items
    only -- not the union across all seeds.  This lets each branch see a
    larger held-out pool than the global-disjoint case.
    """
    out: dict[tuple[str, str], set[str]] = {}
    for sd in source_dirs:
        summary = sd / "summary.json"
        if not summary.exists():
            continue
        doc = json.loads(summary.read_text())
        for seed, keys in doc.get("sampled_item_keys_by_seed", {}).items():
            out[(sd.name, seed)] = set(keys)
    return out


# ── Selection ──────────────────────────────────────────────────────────


def _dedupe_within_branch(attacks: list[HarvestedAttack]) -> list[HarvestedAttack]:
    """Keep one row per (branch, attack_string) — preserves cross-branch dups.

    Used when grouping by branch so each branch retains its own ranking even
    when two branches independently discover the same attack string.
    """
    best: dict[tuple, HarvestedAttack] = {}
    for a in attacks:
        k = (branch_key(a), a.attack_id)
        existing = best.get(k)
        if existing is None or a.source_score > existing.source_score:
            best[k] = a
    return list(best.values())


def _dedupe_keep_best(attacks: list[HarvestedAttack]) -> list[HarvestedAttack]:
    """Keep one row per attack_string globally — used by pool_all/per_condition."""
    best: dict[str, HarvestedAttack] = {}
    for a in attacks:
        existing = best.get(a.attack_id)
        if existing is None or a.source_score > existing.source_score:
            best[a.attack_id] = a
    return list(best.values())


def select_attacks(
    attacks: list[HarvestedAttack],
    *,
    strategy: str,
    top_k: int,
) -> list[HarvestedAttack]:
    """Pick attacks per the chosen grouping strategy.

    * ``pool_all`` — global top-K by source_score
    * ``per_condition`` — top-K per source_condition
    * ``per_branch`` — top-K per (source_dir, seed, condition, nudge) tuple
      (the matrix's branch structure)

    For ``per_branch``, dedup is per-branch — if two branches independently
    found the same attack string, both branches still get a slot.  For the
    other strategies, global dedup keeps each attack-string once.
    """
    if strategy == "per_branch":
        attacks = _dedupe_within_branch(attacks)
    else:
        attacks = _dedupe_keep_best(attacks)
    if strategy == "pool_all":
        return sorted(attacks, key=lambda a: a.source_score, reverse=True)[:top_k]
    if strategy == "per_condition":
        groups: dict[str, list[HarvestedAttack]] = {}
        for a in attacks:
            groups.setdefault(a.source_condition, []).append(a)
        chosen: list[HarvestedAttack] = []
        for grp in groups.values():
            grp.sort(key=lambda a: a.source_score, reverse=True)
            chosen.extend(grp[:top_k])
        return chosen
    if strategy == "per_branch":
        groups = {}
        for a in attacks:
            groups.setdefault(branch_key(a), []).append(a)
        chosen = []
        for grp in groups.values():
            grp.sort(key=lambda a: a.source_score, reverse=True)
            chosen.extend(grp[:top_k])
        return chosen
    raise ValueError(f"Unknown attack-selection strategy: {strategy!r}")


# ── Result types ───────────────────────────────────────────────────────


@dataclass
class TransferResult:
    """One (attack × target_victim × target_condition) transfer eval cell."""
    attack_id: str
    attack_string: str
    source_condition: str
    source_score: float
    source_branch: str  # branch_key — for per-branch aggregation
    target_victim: str
    target_condition: str
    per_item: list[ItemScore]
    asr: float
    parse_ok_rate: float
    disqualified_rate: float

    def as_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "per_item"},
            "per_item": [asdict(p) for p in self.per_item],
        }


# ── Transfer task spec ─────────────────────────────────────────────────


@dataclass
class TransferTaskSpec:
    """Task-specific bits a transfer evaluation needs.

    ``conditions`` maps each ``Condition`` enum value the task supports
    to its blueprint (e.g. for arena_hard the per-seed ones differ; the
    caller already resolved the task-time blueprint).

    ``task`` is the JudgeTask that scores each (item, attack-output)
    pair.  For arena_hard's per-seed slot permutation, the caller
    resolves the seed-bound task before constructing this spec.

    ``stable_item_key_fn`` returns the same string used in the matrix
    summary's ``sampled_item_keys_by_seed`` — so the held-out exclusion
    can match keys exactly.
    """
    name: str  # short, filesystem-safe (e.g. "gsm8k")
    conditions: dict[Condition, ConditionBlueprint]
    task: JudgeTask
    items: Sequence[Any]
    stable_item_key_fn: Callable[[Any], str]
    cache_namespace: str  # e.g. "transfer:gsm8k:verdict"


# ── Item sampling ──────────────────────────────────────────────────────


def sample_target_items(
    spec: TransferTaskSpec,
    *,
    n: int,
    seed: str,
    excluded_keys: set[str] | None = None,
) -> list[Any]:
    """Deterministically sample ``n`` items, optionally excluding source items.

    ``excluded_keys`` should be the union of ``sampled_item_keys_by_seed``
    values from the source PAIR runs; passing this gives a held-out
    sample disjoint from the search items.
    """
    pool = list(spec.items)
    if excluded_keys:
        pool = [it for it in pool if spec.stable_item_key_fn(it) not in excluded_keys]
    if len(pool) < n:
        raise SystemExit(
            f"sample_target_items: pool size {len(pool)} < requested n={n} "
            f"after exclusion of {len(excluded_keys or ())} items. "
            f"Either reduce --target-n-items or use a larger source dataset."
        )
    records = deterministic_sample(
        pool, stable_key_fn=spec.stable_item_key_fn, seed=seed,
        limit=n, order="hash",
    )
    return [r["item"] for r in records]


# ── Strategies ─────────────────────────────────────────────────────────


def make_target_victim_session(
    *,
    target_victim_model: str,
    target_provider: str,
    experiment_name: str,
    poll_interval: float,
    target_reasoning_effort: str | None = None,
    target_max_completion_tokens: int | None = None,
) -> tuple[ModelSession, CostTracker, ResponseCache, Strategy]:
    """Build (session, tracker, cache, strategy) for one target victim.

    Strategy auto-picks Batch for anthropic/openai (50% off), Parallel
    elsewhere.  Mirrors :func:`runner.make_pair_strategies`.
    """
    api_model = resolve_model(target_victim_model, target_provider)
    suffix = ""
    if target_reasoning_effort is not None:
        suffix += f"_reasoning-{target_reasoning_effort}"
    if target_max_completion_tokens is not None:
        suffix += f"_max-{target_max_completion_tokens}"
    cache = get_cache(target_victim_model, suffix=suffix)
    tracker = CostTracker(
        model=target_victim_model, provider=target_provider,
        experiment=experiment_name,
        ledger_path=POC_ROOT / "cost_ledger.jsonl",
    )
    if target_provider in {"openai", "openrouter", "fireworks", "vllm"}:
        client = make_openai_client(provider=target_provider)
    else:
        client = None
    defaults: dict[str, Any] = {}
    if target_reasoning_effort is not None:
        defaults["reasoning_effort"] = target_reasoning_effort
    if target_max_completion_tokens is not None:
        defaults["max_completion_tokens"] = target_max_completion_tokens
    session = ModelSession(
        client=client,
        provider=target_provider,
        model=api_model,
        cache=cache,
        tracker=tracker,
        defaults=defaults,
    )
    # Smooth tier rolloff: cheapest tier the provider reliably supports.
    #   anthropic  → Batch    (50% off, async polling)
    #   openai     → Flex     (50% off, sync — no batch wait)
    #   everything else (openrouter / fireworks / vllm) → Standard parallel.
    #     OpenRouter's flex tier is unreliable in practice (per past
    #     experience), so we don't claim a discount we may not get.
    if target_provider == "anthropic":
        strategy: Strategy = Batch(poll_interval_s=poll_interval)
    elif target_provider == "openai":
        strategy = Parallel(tier=Tier.FLEX, adaptive=True)
    else:
        strategy = Parallel(tier=Tier.STANDARD, adaptive=True)
    return session, tracker, cache, strategy


# ── Evaluation core ───────────────────────────────────────────────────


def _disqualified_score(
    *, task: JudgeTask, item: Any, attack_string: str,
) -> ItemScore | None:
    if task.disqualify_fn is None:
        return None
    reason = task.disqualify_fn(attack_string, item)
    if reason is None:
        return None
    return ItemScore(
        item_id=task.item_id_fn(item),
        attack_succeeded=False,
        verdict=None,
        parse_ok=True,
        raw_response="",
        disqualified=True,
        disqualified_reason=reason,
    )


def _score_completion(
    *, task: JudgeTask, item: Any, ctx: dict, completion: CompletedRequest,
) -> ItemScore:
    item_id = task.item_id_fn(item)
    if completion.error is not None:
        return ItemScore(
            item_id=item_id,
            attack_succeeded=False,
            verdict=None,
            parse_ok=False,
            raw_response="",
        )
    raw = (completion.text or "").strip()
    score_result = task.score_fn(raw, item, ctx)
    return ItemScore(
        item_id=item_id,
        attack_succeeded=score_result.attack_succeeded,
        verdict=score_result.verdict,
        parse_ok=score_result.parse_ok,
        raw_response=raw,
    )


async def run_transfer_for_victim(
    *,
    spec: TransferTaskSpec,
    selected_attacks: Sequence[HarvestedAttack],
    target_victim_model: str,
    target_conditions: Sequence[Condition],
    target_items: Sequence[Any] | dict[str, Sequence[Any]] | Callable[[HarvestedAttack], Sequence[Any]],
    target_sample_seed: str,
    diagonal_only: bool,
    session: ModelSession,
    strategy: Strategy,
    show_progress: bool = False,
) -> list[TransferResult]:
    """Build all (attack × target_condition × item) requests, dispatch via run_many.

    With ``diagonal_only=True``, an attack only contributes evaluations
    whose target_condition equals its source_condition (the canonical
    "transfer to the same condition" metric).  Otherwise each attack is
    evaluated against every target_condition (full source × target
    cross-condition matrix).

    Disqualified (attack, item) pairs are recorded as
    ``ItemScore(disqualified=True)`` without spending a victim call.

    Returns one :class:`TransferResult` per (attack, target_victim,
    target_condition) cell.  Cells with all items disqualified still
    produce a result with ``asr=0.0``.
    """
    # Normalize ``target_items`` to a per-attack lookup so the rest of
    # the function handles all three input forms uniformly:
    #   * Sequence[Any]    -> shared list, every attack sees same items
    #   * dict[str, ...]   -> keyed by attack_id (legacy)
    #   * Callable[[a], ..]-> takes the attack object and returns items
    #                         (used when items depend on attack.source_seed
    #                         since multiple attacks can share an
    #                         attack_id and the dict form would clobber)
    if callable(target_items) and not isinstance(target_items, dict):
        items_for: Callable[[HarvestedAttack], Sequence[Any]] = target_items  # type: ignore[assignment]
    elif isinstance(target_items, dict):
        items_for = lambda a: target_items[a.attack_id]
    else:
        items_for = lambda a: target_items  # type: ignore[assignment]

    # Phase 1 — build requests + per-cell scoring tables.
    # cell key = (attack_idx, target_condition_idx)
    cell_keys: list[tuple[int, int]] = []
    for ai, attack in enumerate(selected_attacks):
        for ci, cond in enumerate(target_conditions):
            if diagonal_only and cond.value != attack.source_condition:
                continue
            cell_keys.append((ai, ci))

    # Per (cell, item) → ItemScore | None and request-index lookup.  Each
    # cell's list is sized to that attack's specific item set.
    per_cell_per_item_score: dict[tuple[int, int], list[ItemScore | None]] = {
        ck: [None] * len(items_for(selected_attacks[ck[0]])) for ck in cell_keys
    }
    per_cell_per_item_request_idx: dict[tuple[int, int], list[int | None]] = {
        ck: [None] * len(items_for(selected_attacks[ck[0]])) for ck in cell_keys
    }
    requests: list[CompletionRequest] = []
    for ai, ci in cell_keys:
        attack = selected_attacks[ai]
        cond = target_conditions[ci]
        blueprint = spec.conditions[cond]
        cell_items = items_for(attack)
        for item_idx, item in enumerate(cell_items):
            disq = _disqualified_score(
                task=spec.task, item=item, attack_string=attack.attack_string,
            )
            if disq is not None:
                per_cell_per_item_score[(ai, ci)][item_idx] = disq
                continue
            payload, ctx = spec.task.make_payload(item, attack.attack_string)
            messages, extra = assemble_messages(blueprint, payload)
            overrides = dict(extra) if extra else None
            req_idx = len(requests)
            requests.append(CompletionRequest(
                messages=messages,
                metadata={"cell": (ai, ci), "item_idx": item_idx, "ctx": ctx},
                cache_tag=(
                    f"{spec.cache_namespace}:{target_victim_model}:"
                    f"{cond.value}:{target_sample_seed}:{attack.attack_id}:"
                    f"{spec.task.item_id_fn(item)}"
                ),
                overrides=overrides,
            ))
            per_cell_per_item_request_idx[(ai, ci)][item_idx] = req_idx

    # Phase 2 — dispatch.
    if requests:
        completions = await run_many(
            session, requests, strategy=strategy,
            show_progress=show_progress, raise_on_error=False,
        )
    else:
        completions = []

    # Phase 3 — distribute completions back, build TransferResults.
    results: list[TransferResult] = []
    for ai, ci in cell_keys:
        attack = selected_attacks[ai]
        cond = target_conditions[ci]
        cell_items = items_for(attack)
        for item_idx, item in enumerate(cell_items):
            req_idx = per_cell_per_item_request_idx[(ai, ci)][item_idx]
            if req_idx is None:
                continue  # disqualified, already filled
            completion = completions[req_idx]
            ctx = completion.metadata.get("ctx", {}) if completion.metadata else {}
            per_cell_per_item_score[(ai, ci)][item_idx] = _score_completion(
                task=spec.task, item=item, ctx=ctx, completion=completion,
            )
        per_item: list[ItemScore] = [
            s for s in per_cell_per_item_score[(ai, ci)] if s is not None
        ]
        n = len(per_item)
        results.append(TransferResult(
            attack_id=attack.attack_id,
            attack_string=attack.attack_string,
            source_condition=attack.source_condition,
            source_score=attack.source_score,
            source_branch=branch_key(attack),
            target_victim=target_victim_model,
            target_condition=cond.value,
            per_item=per_item,
            asr=sum(p.attack_succeeded for p in per_item) / n if n else 0.0,
            parse_ok_rate=sum(p.parse_ok for p in per_item) / n if n else 0.0,
            disqualified_rate=sum(p.disqualified for p in per_item) / n if n else 0.0,
        ))
    return results


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate_per_branch(
    results: Sequence[TransferResult],
    *,
    target_victims: Sequence[str],
    target_conditions: Sequence[Condition],
    diagonal_only: bool,
) -> list[dict]:
    """Group TransferResults by (target_victim × target_condition) and within
    each group by ``source_branch``, computing per-branch mean ASR and
    cell-level mean ± std across branches.

    Output rows have keys: target_victim, target_condition, n_branches,
    branch_means (list[float]), mean_asr, std_asr, n_attacks_total.

    With ``diagonal_only=True``, cells where target_condition !=
    source_condition are simply absent from results, so this aggregation
    naturally produces only the diagonal.
    """
    rows: list[dict] = []
    for v in target_victims:
        for c in target_conditions:
            cell = [r for r in results
                    if r.target_victim == v and r.target_condition == c.value]
            if not cell:
                continue
            by_branch: dict[str, list[float]] = {}
            for r in cell:
                by_branch.setdefault(r.source_branch, []).append(r.asr)
            branch_means = [sum(vs) / len(vs) for vs in by_branch.values()]
            n = len(branch_means)
            mean = sum(branch_means) / n if n else 0.0
            var = sum((m - mean) ** 2 for m in branch_means) / n if n else 0.0
            rows.append({
                "target_victim": v,
                "target_condition": c.value,
                "n_branches": n,
                "branch_means": branch_means,
                "mean_asr": mean,
                "std_asr": var ** 0.5,
                "n_attacks_total": len(cell),
            })
    return rows


def aggregate_heatmap_matrix(
    results: Sequence[TransferResult],
    *,
    target_victims: Sequence[str],
    target_conditions: Sequence[Condition],
    metric: str,  # "mean_asr" or "ever_broken_fraction"
    source_condition_filter: str | None = None,
) -> list[list[float]]:
    if source_condition_filter is not None:
        results = [r for r in results if r.source_condition == source_condition_filter]
    matrix: list[list[float]] = []
    for v in target_victims:
        row: list[float] = []
        for c in target_conditions:
            cell = [r for r in results
                    if r.target_victim == v and r.target_condition == c.value]
            if not cell:
                row.append(float("nan"))
                continue
            if metric == "mean_asr":
                row.append(sum(r.asr for r in cell) / len(cell))
            elif metric == "ever_broken_fraction":
                n_items = len(cell[0].per_item) if cell[0].per_item else 0
                ever = set()
                for r in cell:
                    for p in r.per_item:
                        if p.attack_succeeded:
                            ever.add(p.item_id)
                row.append(len(ever) / n_items if n_items else 0.0)
            else:
                raise ValueError(metric)
        matrix.append(row)
    return matrix


def plot_heatmap(
    matrix: list[list[float]],
    *,
    target_victims: Sequence[str],
    target_conditions: Sequence[Condition],
    title: str,
    out_path: Path,
    cmap: str = "Reds",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.5 + 1.4 * len(target_conditions),
                                     1.5 + 0.8 * len(target_victims)))
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(target_conditions)))
    ax.set_xticklabels([c.value for c in target_conditions], rotation=30, ha="right")
    ax.set_yticks(range(len(target_victims)))
    ax.set_yticklabels(list(target_victims))
    ax.set_title(title, fontsize=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9)
            else:
                colour = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=colour)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def emit_per_attack_csv(
    results: Sequence[TransferResult],
    *,
    out_path: Path,
) -> None:
    """One row per (attack, target_victim, target_condition); useful for outliers."""
    fieldnames = [
        "attack_id", "source_condition", "source_branch", "source_score",
        "target_victim", "target_condition", "asr", "parse_ok_rate",
        "disqualified_rate", "n_items",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "attack_id": r.attack_id,
                "source_condition": r.source_condition,
                "source_branch": r.source_branch,
                "source_score": f"{r.source_score:.3f}",
                "target_victim": r.target_victim,
                "target_condition": r.target_condition,
                "asr": f"{r.asr:.3f}",
                "parse_ok_rate": f"{r.parse_ok_rate:.3f}",
                "disqualified_rate": f"{r.disqualified_rate:.3f}",
                "n_items": len(r.per_item),
            })


def emit_top_attacks_markdown(
    *,
    selected: Sequence[HarvestedAttack],
    results: Sequence[TransferResult],
    target_victims: Sequence[str],
    target_conditions: Sequence[Condition],
    source_dirs: Sequence[Path],
    target_sample_seed: str,
    out_path: Path,
) -> None:
    attacks_by_id = {a.attack_id: a for a in selected}
    results_by_attack: dict[str, list[TransferResult]] = {}
    for r in results:
        results_by_attack.setdefault(r.attack_id, []).append(r)

    rows = []
    for aid, cell_results in results_by_attack.items():
        if aid not in attacks_by_id:
            continue
        mean_asr = sum(r.asr for r in cell_results) / len(cell_results)
        max_asr = max(r.asr for r in cell_results)
        rows.append((aid, mean_asr, max_asr, cell_results))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)

    src_label = (
        source_dirs[0].name if len(source_dirs) == 1
        else f"{len(source_dirs)} sources"
    )
    lines: list[str] = [
        f"# Top attacks: transfer from `{src_label}`",
        "",
        f"- **Target victims**: {', '.join(target_victims)}",
        f"- **Target conditions**: {', '.join(c.value for c in target_conditions)}",
        f"- **Target item sample seed**: `{target_sample_seed}`",
        f"- **Selected attacks**: {len(selected)}",
        "",
        "Ranked by mean ASR across all target (victim × condition) cells.",
        "",
        "## Summary table",
        "",
        "| rank | mean_asr | max_asr | source_cond | src_score | attack_id | preview |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (aid, mean, maxa, _cells) in enumerate(rows, start=1):
        a = attacks_by_id[aid]
        preview = a.attack_string.replace("\n", "\\n").replace("|", "\\|")
        if len(preview) > 70:
            preview = preview[:70] + "..."
        lines.append(
            f"| {i} | {mean:.3f} | {maxa:.2f} | {a.source_condition} | "
            f"{a.source_score:.2f} | `{aid}` | `{preview}` |"
        )
    lines.append("")
    lines.append("## Per-attack detail")
    lines.append("")
    for i, (aid, mean, maxa, cell_results) in enumerate(rows, start=1):
        a = attacks_by_id[aid]
        lines.append(
            f"### Rank {i} — mean ASR {mean:.3f}, max ASR {maxa:.2f} (`{aid}`)"
        )
        lines.append("")
        lines.append(
            f"- **Source**: `{a.source_condition}` branch "
            f"({'nudge' if a.source_nudge_enabled else 'control'}), "
            f"`{a.source_dir_label}`, seed `{a.source_seed}`, "
            f"turn {a.source_turn_index}, source score {a.source_score:.2f}"
        )
        lines.append("")
        by_cell: dict[tuple[str, str], TransferResult] = {
            (r.target_victim, r.target_condition): r for r in cell_results
        }
        lines.append("| target victim | "
                     + " | ".join(c.value for c in target_conditions) + " |")
        lines.append("|" + "|".join("---" for _ in range(1 + len(target_conditions))) + "|")
        for v in target_victims:
            row = [v]
            for c in target_conditions:
                r = by_cell.get((v, c.value))
                if r is None:
                    row.append("n/a")
                else:
                    ever_count = sum(1 for p in r.per_item if p.attack_succeeded)
                    n_items = len(r.per_item)
                    row.append(f"{r.asr:.2f} ({ever_count}/{n_items})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("```")
        lines.append(a.attack_string)
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines))


# ── Argparse helpers ───────────────────────────────────────────────────


def add_transfer_common_args(parser: argparse.ArgumentParser) -> None:
    """Common --source / --target / --select flags shared by all task subparsers."""
    parser.add_argument(
        "--source-dir", type=Path, nargs="+", required=True,
        help="One or more PAIR-search output directories (each contains "
             "trajectory_*.json + summary.json).  Pass multiple paths to "
             "pool attacks across attacker arms.",
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Top-K per group (per the --attack-source strategy).",
    )
    parser.add_argument(
        "--attack-source", default="per_branch",
        choices=["pool_all", "per_condition", "per_branch"],
        help="How to group attacks before taking top-K.  Default per_branch "
             "(top-K per source_dir × seed × condition × nudge) — the matrix "
             "branch structure.",
    )
    parser.add_argument(
        "--diagonal-only", action="store_true",
        help="Only evaluate each attack against its source_condition.  "
             "Default for the 'within-condition transfer' paper number.",
    )
    parser.add_argument(
        "--exclude-source-items", action="store_true",
        help="Exclude items present in the source PAIR runs' "
             "sampled_item_keys_by_seed before sampling held-out items.  "
             "Required to make the held-out sample disjoint from search.",
    )
    parser.add_argument(
        "--per-seed-held-out", action="store_true",
        help="Compute target items per source-(arm, seed): each attack "
             "is evaluated on a 40-item set disjoint from THAT seed's 20 "
             "search items only (not the union across all seeds).  This "
             "lets each branch see a larger held-out pool than the "
             "global-disjoint case (which is the existing default). "
             "Implies --exclude-source-items semantically (per-seed); "
             "incompatible with passing --exclude-source-items at the "
             "same time.",
    )
    parser.add_argument(
        "--target-victim-models", nargs="+", required=True,
        help="One or more canonical target victim model names.",
    )
    parser.add_argument("--target-victim-provider", default="openrouter")
    parser.add_argument(
        "--target-conditions", nargs="+",
        default=[c.value for c in Condition],
        choices=[c.value for c in Condition],
    )
    parser.add_argument("--target-n-items", type=int, default=40)
    parser.add_argument(
        "--target-sample-seed", default="transfer-2026-04-27",
        help="Held-out item sampling seed.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=20.0,
        help="Batch polling interval in seconds (anthropic/openai targets).",
    )
    parser.add_argument(
        "--target-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Explicit reasoning_effort for OpenAI-compatible target victims.",
    )
    parser.add_argument(
        "--target-max-completion-tokens",
        type=int,
        default=None,
        help="Explicit max_completion_tokens for target victim calls.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)


def default_transfer_output_dir(
    *,
    task_name: str,
    source_dirs: Sequence[Path],
    args: argparse.Namespace,
    extra_tag: str = "",
) -> Path:
    """Standard naming under ``RESULTS_DIR/pair_<task>_transfer/``."""
    src_tag = (
        source_dirs[0].name if len(source_dirs) == 1
        else f"{len(source_dirs)}srcs_{stable_digest('+'.join(s.name for s in source_dirs), length=8)}"
    )
    leaf = (
        f"from__{src_tag}__"
        f"to_{len(args.target_victim_models)}victims_{len(args.target_conditions)}conds"
        f"_k{args.top_k}_{args.attack_source}_{args.target_sample_seed}"
    )
    if args.diagonal_only:
        leaf += "_diag"
    if extra_tag:
        leaf += f"_{extra_tag}"
    return RESULTS_DIR / f"pair_{task_name}_transfer" / leaf
