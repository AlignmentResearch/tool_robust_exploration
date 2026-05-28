"""Bootstrap-CI math for (condition_a − condition_b) ASR deltas.

Library module -- consumed by ``scripts/table_gen/_transfer_tables.py``
(which owns the actual paper-table emission) and a couple of variance-
decomposition scripts.  Does not write any artifacts itself.

Design:
- Outer resample: stratified by attacker.  Each (victim, condition) cell
  has 18 branches = 3 attackers x 6 seeds; per bootstrap iter we draw
  with-replacement *within each attacker*, preserving the family-spanning
  experimental design.
- Inner resample (shared-item path): items, with replacement, drawn ONCE
  per iter and applied to every (victim, condition) cell of that task.
  Shared item resampling means per-victim deltas are correlated by design
  (which is honest -- they are, in reality, evaluated on the same items).
- Inner resample (per-branch path): each branch has its own item set;
  resample items independently per branch within each iter.  Used when
  the transfer run is per-seed item-disjoint (the 2026-04-30 MT-Bench
  per-seed setup).
- Pairing: ASR(a) and ASR(b) are computed on the *same* resampled items
  within each iter (shared-item path) or within each matched
  ``(attacker, seed)`` branch pair (per-branch path), so item-level
  noise mostly cancels in the delta.

What this captures:
- Item-level uncertainty (would the delta look different if items had been
  drawn differently from each task's evaluation pool?)
- Search-realization uncertainty (would the delta look different if the
  attack search had used different seeds, holding the 3 attacker families
  fixed?)
What this does NOT claim:
- Generalization to a population of attackers (3 attackers is a fixed
  design; we don't pretend it samples a population).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _transfer_data import load_per_branch_transfer  # noqa: E402


# Display constants -- re-exported for consumers like
# scripts/decompose_transfer_ci_variance.py.  The canonical copies live
# in scripts/table_gen/_transfer_tables.py; we duplicate the minimum
# subset other scripts depend on here.
VICTIM_DISPLAY = {
    "claude-haiku-4-5": "Haiku-4.5",
    "qwen3-8b": "Qwen3-8b",
    "qwen/qwen3.5-flash-02-23": "Qwen3.5-flash",
    "google/gemma-4-26b-a4b-it": "Gemma-4-26b",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.4": "GPT-5.4",
}
VICTIM_ORDER = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-haiku-4-5",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3.5-flash-02-23",
    "qwen3-8b",
]
CONDITION_DISPLAY = {
    "baseline": "UserOnly",
    "multi_msg": "UserSys",
    "tool_wrapped_v2": "ToolWrapped",
    "system_distrust": "SystemDistrust",
    "tool_distrust_v2": "ToolDistrust",
}
# (a, b, label) -- delta = ASR(a) - ASR(b).  Positive = ``a`` more vulnerable.
DIFF_PAIRS = [
    ("tool_wrapped_v2", "baseline",
     r"\texttt{ToolWrapped}$-$\texttt{UserOnly}"),
    ("tool_wrapped_v2", "multi_msg",
     r"\texttt{ToolWrapped}$-$\texttt{UserSys}"),
    ("tool_distrust_v2", "system_distrust",
     r"\texttt{ToolDistrust}$-$\texttt{SystemDistrust}"),
]
TASK_DISPLAY = {
    "gsm8k": "GSM8K",
    "mtbench": "MT-Bench",
    "arena_hard": "Arena-Hard",
}


def load_task(
    task: str, results_dir: Path,
    *, attack_succeeded_fn=None, target_seed_tag: str | None = None,
) -> tuple[dict, list[str]]:
    """Re-shape the shared per-task loader's output to the legacy
    ``data[victim][condition] -> [branches]`` shape used here.  Branch
    records get an ``attacker`` key (short name) for backward compat
    with the bootstrap callers below.

    Picks the target_seed_tag via the per-task dispatch in
    :mod:`_transfer_data` (no explicit tag passed) unless overridden
    via ``target_seed_tag``.

    ``attack_succeeded_fn`` (optional) is forwarded to the loader to
    reparse per-item outcomes under an alternate parser; see
    :mod:`_parser_variants` for the registry.
    """
    flat, items = load_per_branch_transfer(
        task, results_dir, with_per_item=True,
        attack_succeeded_fn=attack_succeeded_fn,
        target_seed_tag=target_seed_tag,
    )
    data: dict = defaultdict(lambda: defaultdict(list))
    for (victim, cond), brs in flat.items():
        for br in brs:
            data[victim][cond].append({
                "attacker": br["attacker_short"],
                "seed": br["seed"],
                "outcomes": br["outcomes"],
                "src_arm": br["src_arm"],
                "src_branch": br["src_branch"],
            })
    return data, items


def dedupe_branches(branches: list[dict]) -> list[dict]:
    """No-op now: ``load_per_branch_transfer`` already dedupes by
    (attacker, seed, src_arm).  Kept as a stable function reference for
    callers (``scripts/table_gen/_transfer_tables.py``)."""
    return list(branches)


def build_matrix(branches: list[dict], items: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, attackers) where M is (n_branches, n_items) bool and
    attackers is (n_branches,) array of attacker short names."""
    n_b = len(branches)
    n_i = len(items)
    M = np.zeros((n_b, n_i), dtype=bool)
    atks = np.empty(n_b, dtype=object)
    for r, br in enumerate(branches):
        oc = br["outcomes"]
        for c, it in enumerate(items):
            M[r, c] = bool(oc.get(it, False))
        atks[r] = br["attacker"]
    return M, atks


def restrict_to_attackers(M: np.ndarray, atks: np.ndarray, keep: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return rows of M and atks whose attacker is in `keep`."""
    mask = np.array([a in keep for a in atks])
    return M[mask], atks[mask]


def stratified_indices(attackers: np.ndarray) -> dict[str, np.ndarray]:
    """Group branch indices by attacker."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(attackers):
        groups[a].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


_STAT_FUNCS = {
    "mean": lambda x: x.mean(),
    "p75": lambda x: float(np.percentile(x, 75)),
    "p90": lambda x: float(np.percentile(x, 90)),
    "max": lambda x: float(x.max()),
    "median": lambda x: float(np.percentile(x, 50)),
}


def _resolve_stat(statistic: str | callable):
    """Map a string name (mean/p75/p90/max/median) to a callable, or
    pass through a user-provided callable."""
    if callable(statistic):
        return statistic
    if statistic not in _STAT_FUNCS:
        raise ValueError(
            f"Unknown cell-level statistic {statistic!r}; "
            f"valid: {sorted(_STAT_FUNCS)} or a callable"
        )
    return _STAT_FUNCS[statistic]


def bootstrap_delta(
    M_a: np.ndarray, M_b: np.ndarray,
    atk_a: np.ndarray, atk_b: np.ndarray,
    B: int, rng: np.random.Generator,
    *, item_idx_per_iter: np.ndarray | None = None,
    statistic: str = "mean",
) -> np.ndarray:
    """Shared-item bootstrap: assumes ALL branches share the same item
    set (items are columns of M_a / M_b).  Returns (B,) delta samples.

    If ``item_idx_per_iter`` (shape ``(B, n_items)``) is provided, uses
    those item resamples (needed for "shared item resample across
    victims within a task").  Otherwise draws fresh per call.

    ``statistic`` controls the cell-level summary across branches.
    "mean" recovers the original behaviour. "p75" / "p90" / "max" /
    "median" report the corresponding percentile of per-branch ASR within
    each cell, then take the difference between conditions.  Order of
    operations: branch ASR := mean over (resampled) items per branch,
    then statistic over (resampled) branches.
    """
    n_items = M_a.shape[1]
    assert M_b.shape[1] == n_items
    stat_fn = _resolve_stat(statistic)
    deltas = np.empty(B)
    a_groups = stratified_indices(atk_a)
    b_groups = stratified_indices(atk_b)

    for k in range(B):
        if item_idx_per_iter is not None:
            ii = item_idx_per_iter[k]
        else:
            ii = rng.integers(0, n_items, size=n_items)
        ai = np.concatenate([
            rng.choice(idxs, size=len(idxs), replace=True)
            for idxs in a_groups.values()
        ])
        bi = np.concatenate([
            rng.choice(idxs, size=len(idxs), replace=True)
            for idxs in b_groups.values()
        ])
        per_branch_a = M_a[np.ix_(ai, ii)].mean(axis=1)
        per_branch_b = M_b[np.ix_(bi, ii)].mean(axis=1)
        deltas[k] = stat_fn(per_branch_a) - stat_fn(per_branch_b)
    return deltas


def make_branch_pairs(
    branches_a: list[dict], branches_b: list[dict],
) -> list[tuple[dict, dict]]:
    """Match cond_a / cond_b branches by ``(attacker, seed)``.  Each
    pair must have IDENTICAL item sets (same seed -> same items in the
    per-seed transfer mode); raises if they differ.

    Asymmetric branches (present in only one condition) are *warned*
    rather than raised: this happens during exploratory partial reruns
    where one condition got new seeds before the other.  The
    intersection-only behaviour still gives a valid (smaller) paired
    delta, but you usually want to know it happened.
    """
    by_key_a = {(b["attacker"], b["seed"]): b for b in branches_a}
    by_key_b = {(b["attacker"], b["seed"]): b for b in branches_b}
    a_only = sorted(set(by_key_a) - set(by_key_b))
    b_only = sorted(set(by_key_b) - set(by_key_a))
    if a_only or b_only:
        shared_n = len(set(by_key_a) & set(by_key_b))
        print(
            f"[warn] make_branch_pairs: asymmetric branches "
            f"(a_only={len(a_only)}, b_only={len(b_only)}); "
            f"using intersection of {shared_n} pairs. "
            f"Examples a_only={a_only[:3]}; b_only={b_only[:3]}."
        )
    shared = sorted(set(by_key_a) & set(by_key_b))
    pairs: list[tuple[dict, dict]] = []
    for key in shared:
        pa = by_key_a[key]
        pb = by_key_b[key]
        items_a = set(pa["outcomes"])
        items_b = set(pb["outcomes"])
        if items_a != items_b:
            raise ValueError(
                f"Branch {key}: items differ between conditions "
                f"(only in a: {len(items_a - items_b)}, "
                f"only in b: {len(items_b - items_a)})"
            )
        pairs.append((pa, pb))
    return pairs


def bootstrap_delta_per_branch(
    pairs: list[tuple[dict, dict]],
    B: int, rng: np.random.Generator,
    *, statistic: str = "mean",
) -> np.ndarray:
    """Per-branch-item bootstrap: each branch has its own (potentially
    different) item set.  Outer = stratified-by-attacker resample of
    pairs; inner = independent item resample within each pair.  Suitable
    when transfer items differ across branches (per-seed held-out
    setup).

    Each pair is ``(branch_a, branch_b)`` with the same ``(attacker,
    seed)``; items must match within the pair (paired across conditions
    naturally because the same seed produces the same items).

    ``statistic`` controls the cell-level summary across (resampled)
    branches.  Defaults to "mean" for backwards compatibility; pass
    "p75" / "p90" / "max" / "median" for worst-case reporting."""
    if not pairs:
        return np.array([])
    n_pairs = len(pairs)

    # Sanity: assume all pairs have the same n_items so we can pack into
    # a (n_pairs, n_items) matrix.  (Different seeds ARE allowed to have
    # different items, but each branch's n_items should match the rest --
    # 40 each in our setup.)
    n_items_per = [len(pa["outcomes"]) for pa, _ in pairs]
    n_items = n_items_per[0]
    if not all(n == n_items for n in n_items_per):
        raise ValueError(
            "bootstrap_delta_per_branch: branches have different n_items "
            f"(got {set(n_items_per)}); ragged shapes not supported."
        )

    OA = np.zeros((n_pairs, n_items))
    OB = np.zeros((n_pairs, n_items))
    attackers: list[str] = []
    for i, (pa, pb) in enumerate(pairs):
        items = sorted(pa["outcomes"])
        OA[i] = [1.0 if pa["outcomes"][it] else 0.0 for it in items]
        OB[i] = [1.0 if pb["outcomes"][it] else 0.0 for it in items]
        attackers.append(pa["attacker"])

    # ``dict.fromkeys(attackers)`` (not ``set``) so iteration order is
    # the deterministic first-occurrence order from ``attackers`` --
    # ``set`` iteration depends on PYTHONHASHSEED, which would make the
    # downstream ``rng.choice(idxs, ...)`` calls consume RNG state in a
    # nondeterministic order and shift CI bounds across processes.
    groups: dict[str, np.ndarray] = {
        a: np.array([i for i, x in enumerate(attackers) if x == a],
                    dtype=np.int64)
        for a in dict.fromkeys(attackers)
    }

    stat_fn = _resolve_stat(statistic)
    deltas = np.empty(B)
    for k in range(B):
        # Outer: stratified resample of pair indices.
        sampled = np.concatenate([
            rng.choice(idxs, size=len(idxs), replace=True)
            for idxs in groups.values()
        ])
        # Inner: per-pair independent item resample.
        item_idx = rng.integers(0, n_items, size=(len(sampled), n_items))
        oa_r = np.take_along_axis(OA[sampled], item_idx, axis=1)
        ob_r = np.take_along_axis(OB[sampled], item_idx, axis=1)
        per_branch_a = oa_r.mean(axis=1)
        per_branch_b = ob_r.mean(axis=1)
        deltas[k] = stat_fn(per_branch_a) - stat_fn(per_branch_b)
    return deltas


def items_are_shared_across_branches(branches: list[dict]) -> bool:
    """Returns True iff every branch's outcome dict has the same item
    set.  Used to dispatch between the shared-item and per-branch-item
    bootstrap variants."""
    if not branches:
        return True
    ref = set(branches[0]["outcomes"])
    return all(set(br["outcomes"]) == ref for br in branches[1:])

