"""Shared data utilities for PAIR transfer-eval analysis.

Single source of truth for:
    * attacker short/canonical mapping
    * parsing transfer source_branch keys
    * parsing matrix arm directory names
    * finding diagonal transfer run dirs
    * dedupe by (attacker, seed, src_arm)
    * loading per-branch transfer data (with or without per-item outcomes)

All scripts that consume PAIR matrix or transfer-eval outputs should
import from here so attacker-dict / arm-name / source_branch parsing is
fixed in one place when arm naming conventions change.

Used by:
    bootstrap_transfer_deltas.py
    gen_transfer_paper_tables.py
    gen_pair_swarm_alltasks.py
    gen_pair_paper_tables.py
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator


def _read_json_maybe_gz(path: Path) -> object:
    """Load JSON from ``path`` or its ``.gz`` sibling, whichever exists.

    The public release ships large transfer_results files gzipped; the
    monorepo keeps them uncompressed. This helper makes both consumable
    by the same code path.
    """
    if path.exists():
        return json.loads(path.read_text())
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rt") as fh:
            return json.load(fh)
    raise FileNotFoundError(f"neither {path} nor {gz_path} exists")


# ── Attacker identity ─────────────────────────────────────────────────

# Short suffix used in arm dir names / source_branch labels -> canonical
# model ID used in the underlying client calls.  When a new attacker is
# added, register it here ONCE.
ATTACKER_SHORT_TO_CANONICAL: dict[str, str] = {
    "kimi": "kimi-k2.5",
    "gemini3": "gemini-3-flash-preview",
    "v4pro": "deepseek-v4-pro",
}
ATTACKER_CANONICAL_TO_SHORT: dict[str, str] = {
    v: k for k, v in ATTACKER_SHORT_TO_CANONICAL.items()
}


# ── Victim identity ───────────────────────────────────────────────────

# Short suffix used in arm dir names -> canonical victim model ID.  Note
# that Haiku-4.5 is the "default" victim and its arm name OMITS the
# victim short -- ``parse_arm_dir`` handles that case specially.
VICTIM_SHORT_TO_CANONICAL: dict[str, str] = {
    "qwen3-8b": "qwen3-8b",
    "qwen35flash": "qwen/qwen3.5-flash-02-23",
    "gemma4": "google/gemma-4-26b-a4b-it",
    "gpt54mini": "gpt-5.4-mini",
    "gpt54": "gpt-5.4",
}


# (task, victim_canonical) cells whose delta columns should be reported
# as dashes in body tables and excluded from delta-counting in paper
# variables (e.g. nMtBenchLastMatchPositiveDeltaCells).  See appendix
# discussion of parser-format collapse for the Haiku-MT-Bench case:
# under tool_wrapped_v2 + attack, ~34% of Haiku outputs emit no
# parseable rating and silently get attack_succeeded=False, so the
# strict-parse-derived delta is partly format-collapse, not active
# resistance.  Other models do not have this issue.
#
# Single source of truth: imported by both gen_transfer_paper_tables
# (table rendering) and gen_paper_vars (count adjustments) so the
# rendered table and the prose macros stay in sync.
DASH_DELTA_CELLS: set[tuple[str, str]] = {
    ("mtbench", "claude-haiku-4-5"),
}


# ── Source-branch / arm-name parsing ──────────────────────────────────

_ARM_VARIANT_SUFFIXES = ("_baseline", "_seeds_def")


def _strip_baseline_suffix(s: str) -> str:
    """Strip routing-variant suffixes from arm names so attacker
    parsing sees the canonical ``..._<attacker>`` ending.

    Variants:
      * ``_baseline`` -- baseline-only arms (2026-04-29) carry this
        when the matrix run was for the baseline condition only.
      * ``_seeds_def`` -- additional-seeds arms (2026-04-30, MT-Bench
        only) carry this when matrix-d/e/f were added in a separate
        directory to avoid clobbering the existing summary.json.

    Order matters only for nested cases (none currently exist); we just
    iterate.
    """
    for suffix in _ARM_VARIANT_SUFFIXES:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def parse_source_branch(source_branch: str) -> tuple[str, str, str, str]:
    """Decode a transfer ``source_branch`` key.

    Format from ``transfer.py``:
        ``{src_arm}::{seed}::{condition}::{nudge}``
    where ``src_arm`` is the matrix arm dir name, ending in
    ``_<attacker_short>`` (optionally followed by ``_baseline`` for the
    baseline-only arms).

    Returns ``(src_arm_clean, attacker_short, attacker_canonical, seed)``.

    Raises ``ValueError`` if the branch is malformed (less than two
    ``::``-separated parts) or ends in an unknown attacker suffix --
    typically a config-drift bug where a new attacker was added to a
    launcher but not registered in ``ATTACKER_SHORT_TO_CANONICAL``.
    Silently skipping would drop a whole attacker family from the
    tables with no indication; raising surfaces it at the source.

    ``src_arm_clean`` is the arm prefix with the ``_baseline`` infix
    stripped (so two arms that differ only in their baseline-vs-other
    routing produce the same arm key for dedupe purposes).
    """
    parts = source_branch.split("::")
    if len(parts) < 2:
        raise ValueError(
            f"malformed source_branch {source_branch!r}: expected at "
            f"least 'src_arm::seed', got {len(parts)} parts"
        )
    src_arm = _strip_baseline_suffix(parts[0])
    seed = parts[1]
    for short, canonical in ATTACKER_SHORT_TO_CANONICAL.items():
        if src_arm.endswith(f"_{short}"):
            return src_arm, short, canonical, seed
    raise ValueError(
        f"unknown attacker suffix in src_arm {src_arm!r} (from "
        f"source_branch {source_branch!r}); registered attackers: "
        f"{sorted(ATTACKER_SHORT_TO_CANONICAL)}.  Add the new attacker "
        f"to ATTACKER_SHORT_TO_CANONICAL in _transfer_data.py."
    )


def parse_arm_dir(arm_dir: Path | str) -> tuple[str, str, str] | None:
    """Decode a matrix arm dir name into (task, victim_canonical, attacker_canonical).

    Naming conventions (see also each launcher script):
        Haiku (default victim, omits victim short from the name):
            ``{task}_{attacker}``                          (4-condition arm)
            ``{task}_baseline_{attacker}``                 (baseline-only)
            ``{task}_{attacker}_baseline``                 (baseline-only, alt)
        Other victims:
            ``{task}_{victim_short}_{attacker}``           (4-condition arm)
            ``{task}_{victim_short}_baseline_{attacker}``  (baseline-only)
            ``{task}_{victim_short}_{attacker}_baseline``  (baseline-only, alt)

    Returns ``None`` if the name doesn't match a known arm pattern.
    """
    name = arm_dir.name if isinstance(arm_dir, Path) else arm_dir
    # Strip any ``_baseline`` / ``_seeds_def`` routing suffix so that e.g.
    # ``mtbench_v4pro_baseline`` parses as the Haiku/v4pro baseline-only
    # arm rather than as some unknown arm with a baseline-named attacker.
    stem = _strip_baseline_suffix(name)
    # Now stem ends in ``_<attacker_short>`` (possibly with a
    # ``_baseline`` infix earlier in the stem -- strip that too).
    for suf, attacker_canonical in ATTACKER_SHORT_TO_CANONICAL.items():
        if stem.endswith(f"_{suf}"):
            arm_minus_atk = stem[: -len(f"_{suf}")]
            break
    else:
        return None

    # Strip a baseline INFIX (legacy ``{task}_..._baseline_{attacker}``
    # naming).  Today's launchers use the suffix form, but keep
    # compatibility just in case.
    if arm_minus_atk.endswith("_baseline"):
        arm_minus_atk = arm_minus_atk[: -len("_baseline")]

    # arm_minus_atk now is either ``{task}`` (Haiku) or ``{task}_{victim_short}``.
    from_pair_paper = ("gsm8k", "mtbench", "arena_hard")
    for task in from_pair_paper:
        if arm_minus_atk == task:
            return task, "claude-haiku-4-5", attacker_canonical
        if arm_minus_atk.startswith(f"{task}_"):
            victim_short = arm_minus_atk[len(task) + 1 :]
            victim_canonical = VICTIM_SHORT_TO_CANONICAL.get(victim_short)
            if victim_canonical is None:
                return None
            return task, victim_canonical, attacker_canonical
    return None


# ── Transfer run discovery ────────────────────────────────────────────

DEFAULT_TARGET_SEED_TAG = "transfer-2026-04-28"

# Per-task target-seed tag dispatch.  Each task may have ONE tag (string)
# or MULTIPLE tags (tuple); the latter is used to union multiple
# transfer-eval runs (e.g. existing Haiku transfer + a new
# Sonnet transfer in the same body table).  MT-Bench currently
# unions Haiku's per-seed eval with Sonnet's 2026-05-09 eval.
# load_per_branch_transfer() falls back to DEFAULT_TARGET_SEED_TAG if
# the task isn't here.
TARGET_SEED_TAG_PER_TASK: dict[str, str | tuple[str, ...]] = {
    # GSM8K: original 2026-05-02 6-victim eval + 2026-05-09 Sonnet eval.
    "gsm8k": ("transfer-2026-05-02-lastmatch", "transfer-2026-05-09-sonnet-gsm8k"),
    # MT-Bench: original Haiku per-seed eval + 2026-05-09 Sonnet eval.
    "mtbench": ("transfer-2026-04-30-perseed", "transfer-2026-05-09-sonnet"),
    # Arena-Hard: original 2026-05-02 + 2026-05-09 Sonnet eval.
    "arena_hard": ("transfer-2026-05-02", "transfer-2026-05-09-sonnet-arena"),
}


def find_diag_transfer_runs(
    task_dir: Path,
    *,
    target_seed_tag: str | tuple[str, ...] = DEFAULT_TARGET_SEED_TAG,
) -> list[Path]:
    """Return all diagonal-only transfer run dirs under ``task_dir``.

    ``target_seed_tag`` may be a single tag string or a tuple of tags;
    the union of matching dirs is returned (sorted).
    """
    if not task_dir.exists():
        return []
    tags = (target_seed_tag,) if isinstance(target_seed_tag, str) else tuple(target_seed_tag)
    seen: set[Path] = set()
    out: list[Path] = []
    for d in sorted(task_dir.iterdir()):
        if not d.is_dir():
            continue
        # Match dirs whose name contains "<tag>_diag" exactly -- this
        # avoids picking up suffix variants like
        # "<tag>-lastmatch_diag_single_v1_last" when the caller passes
        # the prefix-only tag.
        if any(f"{t}_diag" in d.name for t in tags) and d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ── Loading per-branch transfer data ──────────────────────────────────

def _iter_transfer_rows(
    run_dir: Path,
) -> Iterator[tuple[str, dict]]:
    """Yield ``(target_victim, row)`` from a transfer run dir, diagonal only.

    Raises ``RuntimeError`` if the run dir is missing required files or
    has malformed JSON: ``find_diag_transfer_runs`` already filtered to
    dirs that match a tracked ``target_seed_tag``, so any failure here
    means a tracked dir is corrupt or partially written -- preferable to
    fail loudly than to silently exclude its branches from the tables.
    """
    args_path = run_dir / "args.json"
    res_path = run_dir / "transfer_results.json"

    def _exists_either(p: Path) -> bool:
        return p.exists() or p.with_suffix(p.suffix + ".gz").exists()

    missing = [p.name for p in (args_path, res_path) if not _exists_either(p)]
    if missing:
        raise RuntimeError(
            f"transfer run {run_dir} missing required file(s): "
            f"{', '.join(missing)}"
        )
    try:
        args = _read_json_maybe_gz(args_path)
        victim = args["target_victim_models"][0]
        rows = _read_json_maybe_gz(res_path)
    except (OSError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError(
            f"transfer run {run_dir} could not be parsed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for r in rows:
        if r.get("source_condition") != r.get("target_condition"):
            continue  # diagonal only
        yield victim, r


def load_per_branch_transfer(
    task: str, results_dir: Path, *,
    with_per_item: bool = False,
    target_seed_tag: str | None = None,
    attack_succeeded_fn: "Callable[[dict], bool] | None" = None,
) -> tuple[dict[tuple[str, str], list[dict]], list[str]]:
    """Load per-branch transfer ASR for one task across all victims.

    Returns:
        ``(data, items)`` where:
        - ``data[(victim, condition)] = [branch_record, ...]``
        - ``branch_record`` keys: ``attacker_short``, ``attacker_canonical``,
          ``seed``, ``src_arm``, ``src_branch``, ``asr`` (and ``outcomes``
          dict ``{item_id: bool}`` if ``with_per_item=True``)
        - ``items`` is the canonical item-id list (shared across all
          branches of this task; empty if ``with_per_item=False`` or no
          data found)

    Dedupes by ``(attacker_short, seed, src_arm)`` per cell, since
    overlapping transfer runs (e.g. ``2srcs_*`` and ``3srcs_*``) include
    the same kimi/gemini3 branches.

    ``target_seed_tag`` defaults to the per-task entry in
    :data:`TARGET_SEED_TAG_PER_TASK` if available, else
    :data:`DEFAULT_TARGET_SEED_TAG`.

    ``attack_succeeded_fn``, if provided, is applied to each ``per_item``
    dict to override the run-time ``attack_succeeded``; ASR and
    ``outcomes`` are recomputed from those derived booleans.  This is
    used by parser-sensitivity analyses to swap (e.g.) MT-Bench
    first-match for last-match without rerunning the judge.
    See :mod:`_parser_variants` for the variant registry.  When
    overriding, per-item outcomes are always materialised even if
    ``with_per_item=False``, since the override implies recomputing ASR.
    """
    if target_seed_tag is None:
        target_seed_tag = TARGET_SEED_TAG_PER_TASK.get(
            task, DEFAULT_TARGET_SEED_TAG,
        )
    materialise_per_item = with_per_item or attack_succeeded_fn is not None

    data: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, str], set[tuple]] = defaultdict(set)
    items: list[str] | None = None

    task_dir = results_dir / f"pair_{task}_transfer"
    for run_dir in find_diag_transfer_runs(task_dir, target_seed_tag=target_seed_tag):
        for victim, row in _iter_transfer_rows(run_dir):
            src_arm, atk_short, atk_canonical, seed = parse_source_branch(
                row["source_branch"],
            )
            cond = row["target_condition"]
            cell = (victim, cond)
            dedupe_key = (atk_short, seed, src_arm)
            if dedupe_key in seen[cell]:
                continue
            seen[cell].add(dedupe_key)
            rec = {
                "attacker_short": atk_short,
                "attacker_canonical": atk_canonical,
                "seed": seed,
                "src_arm": src_arm,
                "src_branch": row["source_branch"],
                "asr": float(row.get("asr", 0.0)),
            }
            if materialise_per_item:
                if attack_succeeded_fn is not None:
                    outcomes = {
                        pi["item_id"]: bool(attack_succeeded_fn(pi))
                        for pi in row.get("per_item") or []
                    }
                    # Recompute ASR under the override; outcomes is the
                    # source of truth now.
                    rec["asr"] = (
                        sum(outcomes.values()) / len(outcomes)
                        if outcomes else 0.0
                    )
                else:
                    outcomes = {
                        pi["item_id"]: bool(pi["attack_succeeded"])
                        for pi in row.get("per_item") or []
                    }
                if with_per_item:
                    rec["outcomes"] = outcomes
                    if items is None:
                        items = list(outcomes.keys())
                    elif set(outcomes.keys()) != set(items):
                        shared = [i for i in items if i in outcomes]
                        if len(shared) != len(items):
                            # Keep the smaller intersection so all branches
                            # of this task can be evaluated on the same
                            # items.  Caller may want to log this.
                            items = shared
            data[cell].append(rec)

    return data, (items or [])
