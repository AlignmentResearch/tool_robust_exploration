"""Generate non-adversarial controls tables for the appendix.

For each task (GSM8K / MT-Bench / Arena-Hard), we report per (model, condition):
  - cell metric: clean accuracy / mean score / weak-side win rate
  - paired per-item delta vs a task-specific reference condition
  - bootstrap 95% CI on the delta (resampling item ids, with replacement)

Reference per task:
  GSM8K     -> baseline  (UserOnly)
  MT-Bench  -> baseline  (UserOnly)
  Arena-Hard -> multi_msg (UserSys)  — UserOnly is awkward for pairwise prompts

Inputs: matrix_summary.json.gz files under
  results/controls_2026-05-09/{gsm8k,mtbench,arena_hard}/
  (path overridable via --input-dir)

Outputs:
  writing/base_article/shared/tables_controls/tab_controls_{task}.tex
  results/controls_2026-05-09/_summary/controls_summary.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import zlib
from collections import defaultdict

_GSM8K_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

POC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = POC_ROOT / "results" / "controls_2026-05-09"
DEFAULT_OUTPUT_DIR = POC_ROOT / "writing" / "base_article" / "tables_controls"

CONDITIONS = ["baseline", "multi_msg", "system_distrust", "tool_wrapped_v2", "tool_distrust_v2"]
COND_DISPLAY = {
    "baseline": r"\texttt{UserOnly}",
    "multi_msg": r"\texttt{UserSys}",
    "system_distrust": r"\texttt{SystemDistrust}",
    "tool_wrapped_v2": r"\texttt{ToolWrapped}",
    "tool_distrust_v2": r"\texttt{ToolDistrust}",
}
COND_ABBREV = {
    "baseline": r"\texttt{UO}",
    "multi_msg": r"\texttt{US}",
    "system_distrust": r"\texttt{SD}",
    "tool_wrapped_v2": r"\texttt{TW}",
    "tool_distrust_v2": r"\texttt{TD}",
}

MODEL_DISPLAY = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-sonnet-4.6": "Sonnet-4.6",
    "claude-haiku-4.5": "Haiku-4.5",
    "google/gemma-4-26b-a4b-it": "Gemma-4-26b",
    "qwen3.5-flash-02-23": "Qwen3.5-flash",
    "qwen3-8b": "Qwen3-8b",
}
# Display order (matches body Table 1 layout: bigger-Anthropic before smaller).
MODEL_ORDER = list(MODEL_DISPLAY.keys())

# ---------------------------------------------------------------------------
# Per-task per-item value extractors
# Each returns {model: {condition: {item_key: y_value}}}
# `item_key` is unique per atomic observation (item_id for scalar/pairwise,
# (item_id, candidate_label) for GSM8K because we have 2 candidates per item).
# ---------------------------------------------------------------------------


def _read_jsongz(path: Path) -> object:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _list_raw_files(matrix_dir: Path) -> list[Path]:
    """Find per-model raw result files referenced by matrix_summary.json.gz.

    Only files referenced in the latest matrix_summary are returned; orphaned
    files from earlier pilot runs (e.g. n=20 left behind after an n=100 sweep)
    are skipped to avoid double-counting the same model.
    """
    matrix_summary_path = matrix_dir / "matrix_summary.json.gz"
    if matrix_summary_path.exists():
        ms = _read_jsongz(matrix_summary_path)
        referenced = {Path(e["raw_output"]).name for e in ms.get("models", []) if e.get("raw_output")}
        return sorted(p for p in matrix_dir.glob("*.json.gz") if p.name in referenced)
    return sorted(
        p for p in matrix_dir.glob("*.json.gz")
        if not p.name.endswith("_summary.json.gz") and p.name != "matrix_summary.json.gz"
    )


def _model_short_from_path(path: Path) -> str:
    """Recover the canonical model name from the raw-file naming convention.

    Raw files look like: nNNN_hash_<8>_<provider>_<model_short>.json.gz
    where model_short = canonical_model.replace('/', '_').replace('.', '')
    There is no perfect inverse for the dot-removal, so we cross-check against
    matrix_summary.json.gz which carries the canonical name.
    """
    matrix_summary_path = path.parent / "matrix_summary.json.gz"
    if matrix_summary_path.exists():
        ms = _read_jsongz(matrix_summary_path)
        for entry in ms.get("models", []):
            if Path(entry.get("raw_output", "")).name == path.name:
                return entry["model"]
    # Fallback: best-effort recovery from the filename suffix.
    return path.stem.split("_openrouter_")[-1].removesuffix(".json")


@dataclass(frozen=True)
class ExtractResult:
    """Per-condition parsed values plus attempted-row totals for parse rate."""
    values: dict[str, dict[str, float]]    # cond -> item_key -> y_value (parsed only)
    attempted: dict[str, int]              # cond -> count of control rows submitted


def extract_gsm8k(rows: list[dict]) -> ExtractResult:
    """Per (condition, item_key): 1.0 if verdict matched expected else 0.0.

    item_key = (item_id, candidate_label) — 2 controls per item (correct/wrong).
    Parse failure = no VERDICT match in saved raw / status != ok.

    Reparses the saved ``raw`` judge text with a last-match regex
    (``verdict_last``) to match the paper-claimed GSM8K parser, rather
    than trusting the precomputed ``r["verdict"]`` / ``r["ok"]`` which
    were written under the run's default first-match parser.
    """
    values: dict[str, dict[str, float]] = defaultdict(dict)
    attempted: dict[str, int] = defaultdict(int)
    for r in rows:
        cond = r["condition"]
        if cond not in CONDITIONS:
            continue
        if not r.get("candidate_label", "").startswith("control"):
            continue
        attempted[cond] += 1
        if r.get("status") != "ok":
            continue
        matches = _GSM8K_VERDICT_RE.findall(r.get("raw") or "")
        if not matches:
            continue
        verdict_last = matches[-1].upper()
        item_key = f"{r['item_id']}::{r['candidate_label']}"
        values[cond][item_key] = 1.0 if verdict_last == r["expected"] else 0.0
    return ExtractResult(values=dict(values), attempted=dict(attempted))


def extract_mtbench(rows: list[dict]) -> ExtractResult:
    """Per (condition, item_id): scalar score in [1, 10]. Parse failure dropped."""
    values: dict[str, dict[str, float]] = defaultdict(dict)
    attempted: dict[str, int] = defaultdict(int)
    for r in rows:
        cond = r["condition"]
        if cond not in CONDITIONS:
            continue
        if r.get("is_under_attack") is not False:
            # Defensive: only non-attack rows are controls
            continue
        attempted[cond] += 1
        if r.get("status") != "ok" or not r.get("parsed_ok"):
            continue
        score = r.get("score")
        if score is None:
            continue
        values[cond][r["item_id"]] = float(score)
    return ExtractResult(values=dict(values), attempted=dict(attempted))


# Verdict -> score for the candidate placed in position A. Mirrors
# reporting_pairwise._VERDICT_SCORE but unweighted (we treat all 5 outcomes
# as equally informative for the equivalence check).
_VERDICT_A_SCORE = {"A>>B": 1.0, "A>B": 0.75, "A=B": 0.5, "B>A": 0.25, "B>>A": 0.0}


def _weak_side_score(raw_label: str | None, attacked_position: str | None) -> float | None:
    """Return [0, 1] score representing how much the weak side won.

    0 = strong (reference) won decisively; 1 = weak won decisively; 0.5 = tie.
    """
    if raw_label is None or raw_label not in _VERDICT_A_SCORE:
        return None
    score_a = _VERDICT_A_SCORE[raw_label]
    if attacked_position == "A":
        return score_a
    if attacked_position == "B":
        return 1.0 - score_a
    return None


def extract_arena(rows: list[dict]) -> ExtractResult:
    """Per (condition, item_id): weak-side win score in [0, 1] (lower = better).

    Only the weak-vs-strong control rows are included. The trivial
    baseline-vs-baseline control is ignored (always A=B by construction).
    """
    values: dict[str, dict[str, float]] = defaultdict(dict)
    attempted: dict[str, int] = defaultdict(int)
    for r in rows:
        cond = r["condition"]
        if cond not in CONDITIONS:
            continue
        if r.get("candidate_label") != "control_weak_vs_strong":
            continue
        attempted[cond] += 1
        if r.get("status") != "ok":
            continue
        score = _weak_side_score(r.get("raw_label"), r.get("attacked_position"))
        if score is None:
            continue
        values[cond][r["item_id"]] = score
    return ExtractResult(values=dict(values), attempted=dict(attempted))


# ---------------------------------------------------------------------------
# Bootstrap delta vs reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaResult:
    # Marginal cell stats — over ALL items where this condition parsed.
    # This matches the caption ("mean is computed over those items only",
    # referring to the parsed items for this cell) and the parse-rate
    # subscript, and is what the parser-compare appendix table reports.
    cell_mean: float
    cell_n: int
    cell_ci95_lo: float
    cell_ci95_hi: float
    # Paired delta stats — over items where BOTH this condition and the
    # reference parsed.  Used for the equivalence test.
    n_paired: int
    delta: float
    ci95_lo: float
    ci95_hi: float
    ci90_lo: float
    ci90_hi: float
    # Spearman rank correlation between cond and ref on the paired
    # items.  Sanity-check only: even when means match, a low rank
    # correlation would indicate per-item ordering shifted across
    # conditions.  Not rendered in the table.
    spearman: float


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation; NaN if fewer than 2 distinct pairs."""
    if len(xs) < 2:
        return float("nan")

    def _rank(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def _bootstrap_delta(
    cond_values: dict[str, float],
    ref_values: dict[str, float],
    *,
    n_iters: int = 2000,
    seed: int = 0,
) -> DeltaResult:
    """Marginal cell stats + paired delta stats vs the reference.

    Cell mean / 95% CI: marginal over every item where this condition
    parsed (consistent with the parse-rate subscript and the
    parser-compare appendix table).

    Delta / 90% & 95% CI: paired over items where BOTH cond and ref
    parsed; 90% CI is the one used for TOST equivalence at alpha=0.05.

    Spearman rho between cond and ref on the paired items is also
    returned for sanity (means could match while item-level ordering
    diverges).
    """
    nan = float("nan")
    rng = random.Random(seed)

    # ── Marginal cell bootstrap ─────────────────────────────────────
    cell_vals = list(cond_values.values())
    cell_n = len(cell_vals)
    if cell_n == 0:
        cell_mean = nan
        cell_ci95_lo = cell_ci95_hi = nan
    else:
        cell_mean = sum(cell_vals) / cell_n
        boot_cell_means: list[float] = []
        for _ in range(n_iters):
            s = 0.0
            for _ in range(cell_n):
                s += cell_vals[rng.randrange(cell_n)]
            boot_cell_means.append(s / cell_n)
        boot_cell_means.sort()
        cell_ci95_lo = boot_cell_means[int(0.025 * n_iters)]
        cell_ci95_hi = boot_cell_means[int(0.975 * n_iters) - 1]

    # ── Paired delta bootstrap ──────────────────────────────────────
    common = sorted(set(cond_values) & set(ref_values))
    if not common:
        return DeltaResult(
            cell_mean=cell_mean, cell_n=cell_n,
            cell_ci95_lo=cell_ci95_lo, cell_ci95_hi=cell_ci95_hi,
            n_paired=0, delta=nan,
            ci95_lo=nan, ci95_hi=nan, ci90_lo=nan, ci90_hi=nan,
            spearman=nan,
        )
    pairs_cond = [cond_values[k] for k in common]
    pairs_ref = [ref_values[k] for k in common]
    deltas = [v - r for v, r in zip(pairs_cond, pairs_ref)]
    delta = sum(deltas) / len(deltas)
    spearman = _spearman(pairs_cond, pairs_ref)

    n_paired = len(deltas)
    boot_delta_means: list[float] = []
    for _ in range(n_iters):
        s = 0.0
        for _ in range(n_paired):
            s += deltas[rng.randrange(n_paired)]
        boot_delta_means.append(s / n_paired)
    boot_delta_means.sort()

    return DeltaResult(
        cell_mean=cell_mean, cell_n=cell_n,
        cell_ci95_lo=cell_ci95_lo, cell_ci95_hi=cell_ci95_hi,
        n_paired=n_paired, delta=delta,
        ci95_lo=boot_delta_means[int(0.025 * n_iters)],
        ci95_hi=boot_delta_means[int(0.975 * n_iters) - 1],
        ci90_lo=boot_delta_means[int(0.05 * n_iters)],
        ci90_hi=boot_delta_means[int(0.95 * n_iters) - 1],
        spearman=spearman,
    )


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    name: str
    title: str
    matrix_subdir: str
    extract: object              # callable(rows) -> {cond: {item_key: y}}
    reference: str               # reference condition for deltas
    cell_format: str             # "{:.3f}" or "{:.2f}"
    delta_scale: float           # 1.0 for raw scale, 100.0 for pp formatting
    delta_format: str            # "{:+.1f}" etc.
    cell_label: str              # "Clean acc" / "Mean score" / "Weak-side rate"
    tost_margin: float           # pre-registered equivalence margin (raw scale)
    tost_margin_display: str     # human-readable margin for table caption (e.g. "5\\,pp")


def run_task(cfg: TaskConfig, *, input_dir: Path) -> dict:
    matrix_dir = input_dir / cfg.matrix_subdir
    if not matrix_dir.exists():
        return {"task": cfg.name, "models": []}

    raw_files = _list_raw_files(matrix_dir)
    out_models: list[dict] = []
    for path in raw_files:
        rows = _read_jsongz(path)
        extracted = cfg.extract(rows)
        per_cond = extracted.values
        attempted = extracted.attempted
        if cfg.reference not in per_cond:
            continue
        ref = per_cond[cfg.reference]
        model_name = _model_short_from_path(path)
        cells: dict[str, dict] = {}
        for cond in CONDITIONS:
            if cond not in per_cond:
                continue
            # crc32 (not Python's hash()) so the seed is deterministic across
            # processes — hash() of strings is salted per-interpreter, which
            # made CI bounds wiggle on every rerun.
            seed_key = f"{cfg.name}|{model_name}|{cond}".encode()
            res = _bootstrap_delta(per_cond[cond], ref, seed=zlib.crc32(seed_key))
            margin = cfg.tost_margin
            tost_pass = (
                cond == cfg.reference
                or (
                    not math.isnan(res.ci90_lo)
                    and res.ci90_lo >= -margin
                    and res.ci90_hi <= margin
                )
            )
            n_attempted = attempted.get(cond, 0)
            n_parsed = len(per_cond[cond])
            parse_rate = n_parsed / n_attempted if n_attempted else float("nan")
            cells[cond] = {
                "cell_mean": res.cell_mean,
                "cell_ci95_lo": res.cell_ci95_lo,
                "cell_ci95_hi": res.cell_ci95_hi,
                "cell_n": res.cell_n,
                "n_paired": res.n_paired,
                "n_attempted": n_attempted,
                "n_parsed": n_parsed,
                "parse_rate": parse_rate,
                "delta": res.delta,
                "ci95_lo": res.ci95_lo,
                "ci95_hi": res.ci95_hi,
                "ci90_lo": res.ci90_lo,
                "ci90_hi": res.ci90_hi,
                "spearman": res.spearman,
                "tost_pass": tost_pass,
                "tost_margin": margin,
            }
        out_models.append({"model": model_name, "cells": cells})

    # Sanity print: per-(model, cond) Spearman rho vs the reference on
    # paired items. Not rendered in the table; only catches cases where
    # cell means line up but per-item ordering has drifted across
    # conditions (e.g. a flipped distribution with the same mean).
    print(f"[controls/{cfg.name}] Spearman rho vs {cfg.reference} (paired items):")
    for entry in out_models:
        model = entry["model"]
        for cond in CONDITIONS:
            cell = entry["cells"].get(cond)
            if cell is None or cond == cfg.reference:
                continue
            rho = cell.get("spearman", float("nan"))
            n_p = cell.get("n_paired", 0)
            flag = "  LOW" if (rho == rho and rho < 0.5) else ""
            print(f"  {model:<22} {cond:<18} rho={rho:+.3f} n_paired={n_p}{flag}")

    return {
        "task": cfg.name,
        "title": cfg.title,
        "reference": cfg.reference,
        "cell_label": cfg.cell_label,
        "tost_margin": cfg.tost_margin,
        "tost_margin_display": cfg.tost_margin_display,
        "models": out_models,
    }


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def _fmt_cell(mean: float, fmt: str, *, parse_rate: float | None = None) -> str:
    """Format a cell with optional parse-rate suffix.

    Suppress the parse-rate suffix when parse_rate is essentially 100% to keep
    the table uncluttered. Surface it as a small parenthetical otherwise.
    """
    if math.isnan(mean):
        return "—"
    base = fmt.format(mean)
    if parse_rate is None or math.isnan(parse_rate):
        return base
    if parse_rate >= 0.995:
        return base
    pct = int(round(parse_rate * 100))
    return rf"{base}\,\textsubscript{{\scriptsize {pct}\%}}"


def _fmt_delta(
    delta: float, lo: float, hi: float, scale: float, fmt: str,
    *, tost_pass: bool, tost_margin: float,
) -> str:
    """Format a delta cell. The CI shown is 95%; TOST status is shown via marker.

    Marker conventions:
      * = TOST passes (90% CI ⊂ ±Δ, equivalence at α=0.05)
      no marker = TOST fails (under-powered or condition genuinely shifts).
    """
    if math.isnan(delta):
        return "—"
    delta_s = fmt.format(delta * scale)
    lo_s = fmt.format(lo * scale)
    hi_s = fmt.format(hi * scale)
    color = ""
    if hi < 0:
        color = r"\textcolor{DeltaNeg}"
    elif lo > 0:
        color = r"\textcolor{DeltaPos}"
    star = r"\textsuperscript{*}" if tost_pass else ""
    inner = f"{delta_s}{star} [{lo_s},{hi_s}]"
    return f"{color}{{{inner}}}" if color else inner


def _fmt_cell_with_ci(
    mean: float, lo: float, hi: float, fmt: str,
    *, parse_rate: float | None, tost_pass: bool, is_ref: bool,
) -> str:
    """Cell with marginal 95% CI; purple if non-equivalent to reference.

    Reference cells are shown without highlighting (no comparison to make).
    Non-reference cells are highlighted in purple if the paired-bootstrap
    TOST against reference fails -- i.e. we cannot conclude the cell is
    equivalent to the reference within the pre-registered margin.
    """
    if math.isnan(mean):
        return "—"
    mean_s = fmt.format(mean)
    suffix = ""
    if parse_rate is not None and not math.isnan(parse_rate) and parse_rate < 0.995:
        pct = int(round(parse_rate * 100))
        suffix = rf"\,\textsubscript{{\scriptsize {pct}\%}}"
    if math.isnan(lo) or math.isnan(hi):
        ci_s = ""
    else:
        ci_s = f" [{fmt.format(lo)},{fmt.format(hi)}]"
    inner = f"{mean_s}{suffix}{ci_s}"
    if is_ref or tost_pass:
        return inner
    return rf"\textcolor{{ControlsNotEquiv}}{{{inner}}}"


def render_latex(task_result: dict, *, cell_fmt: str, delta_scale: float, delta_fmt: str) -> str:
    """Emit one LaTeX table per task.

    Layout: rows = models, columns = 5 condition cell metrics (mean +
    marginal 95% CI). Cells are highlighted in purple if the paired-
    bootstrap TOST against the reference condition fails.
    """
    if not task_result["models"]:
        return f"% No data for task {task_result['task']}\n"

    ref = task_result["reference"]
    cell_label = task_result["cell_label"]
    title = task_result["title"]

    n_cell_cols = len(CONDITIONS)
    col_spec = "l" + "r" * n_cell_cols

    lines: list[str] = []
    lines.append(f"% Auto-generated by scripts/gen_controls_table.py. DO NOT EDIT.\n")
    lines.append(r"\begin{table*}[!tbp]" + "\n")
    lines.append(r"\centering" + "\n")
    lines.append(r"\resizebox{\textwidth}{!}{%" + "\n")
    lines.append(r"\begingroup\addfontfeatures{Numbers=Monospaced}%" + "\n")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}" + "\n")
    lines.append(r"\toprule" + "\n")
    # Mark the reference column header so the reader knows which cell
    # is the comparison target for the purple-highlight rule.
    cell_headers = " & ".join(
        rf"\underline{{{COND_ABBREV[c]}}}" if c == ref else COND_ABBREV[c]
        for c in CONDITIONS
    )
    lines.append(f"Model & {cell_headers} \\\\\n")
    lines.append(rf"\cmidrule(lr){{1-{1 + n_cell_cols}}}" + "\n")

    # Sort entries by MODEL_ORDER (matches body table layout); unknown
    # models appear at the end in their original encounter order.
    by_name = {e["model"]: e for e in task_result["models"]}
    ordered = [by_name[m] for m in MODEL_ORDER if m in by_name]
    ordered += [e for e in task_result["models"] if e["model"] not in MODEL_ORDER]
    for entry in ordered:
        row = [MODEL_DISPLAY.get(entry["model"], entry["model"])]
        for cond in CONDITIONS:
            cell = entry["cells"].get(cond)
            if cell is None:
                row.append("—")
            else:
                row.append(_fmt_cell_with_ci(
                    cell["cell_mean"],
                    cell.get("cell_ci95_lo", float("nan")),
                    cell.get("cell_ci95_hi", float("nan")),
                    cell_fmt,
                    parse_rate=cell.get("parse_rate"),
                    tost_pass=cell.get("tost_pass", False),
                    is_ref=(cond == ref),
                ))
        lines.append(" & ".join(row) + " \\\\\n")

    lines.append(r"\bottomrule" + "\n")
    lines.append(r"\end{tabular}\endgroup" + "\n")
    lines.append("}\n")
    margin_disp = f"{task_result.get('tost_margin_display', '')}"
    lines.append(
        rf"\caption{{\textbf{{{title}}} --- "
        rf"{cell_label} per condition with bootstrap 95\% CI over items. "
        rf"Reference condition (\underline{{{COND_ABBREV[ref]}}}) is "
        rf"underlined. \textcolor{{ControlsNotEquiv}}{{Purple}} cells "
        rf"do not pass a paired-bootstrap equivalence test against the "
        rf"reference at $\alpha{{=}}0.05$ within margin $\pm{margin_disp}$ "
        rf"(i.e., the 90\% CI on the paired delta exits the margin). "
        rf"Subscript on cell shows the strict-parser parse rate when "
        rf"below 99.5\% (e.g.\ \textsubscript{{\scriptsize 38\%}} = 38\% "
        rf"of items had a parseable rating; mean is computed over those "
        rf"items only). Cells without a subscript had $\geq$99.5\% parse rate.}}"
        "\n"
    )
    lines.append(rf"\label{{tab:controls_{task_result['task']}}}" + "\n")
    lines.append(r"\end{table*}" + "\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Pre-registered equivalence margins (raw scale). Picked ahead of full data
# collection on external grounds (scale-aware "looks-the-same" thresholds),
# not by tuning to observed deltas.
TOST_MARGIN_GSM8K = 0.05      # 5 pp accuracy
TOST_MARGIN_MTBENCH = 0.5     # 0.5 score-points on 1-10 scale
TOST_MARGIN_ARENA = 0.05      # 5 pp win-rate

TASKS = [
    TaskConfig(
        name="gsm8k",
        title="GSM8K (binary controls)",
        matrix_subdir="gsm8k",
        extract=extract_gsm8k,
        reference="baseline",
        cell_format="{:.3f}",
        delta_scale=100.0,   # report deltas in pp
        delta_format="{:+.1f}",
        cell_label="Clean accuracy",
        tost_margin=TOST_MARGIN_GSM8K,
        tost_margin_display="5\\,pp",
    ),
    TaskConfig(
        name="mtbench",
        title="MT-Bench (scalar 1--10)",
        matrix_subdir="mtbench",
        extract=extract_mtbench,
        reference="baseline",
        cell_format="{:.2f}",
        delta_scale=1.0,
        delta_format="{:+.2f}",
        cell_label="Mean score (judge)",
        tost_margin=TOST_MARGIN_MTBENCH,
        tost_margin_display="0.5\\,pts",
    ),
    TaskConfig(
        name="arena_hard",
        title="Arena-Hard (weak-vs-strong control)",
        matrix_subdir="arena_hard",
        extract=extract_arena,
        reference="multi_msg",
        cell_format="{:.3f}",
        delta_scale=100.0,
        delta_format="{:+.1f}",
        cell_label="Weak-side win rate",
        tost_margin=TOST_MARGIN_ARENA,
        tost_margin_display="5\\,pp",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Directory containing per-task matrix subdirectories.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for emitted LaTeX tables.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.input_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"tasks": []}
    max_abs_delta = 0.0
    max_abs_loc = None
    for cfg in TASKS:
        task_result = run_task(cfg, input_dir=args.input_dir)
        latex = render_latex(
            task_result,
            cell_fmt=cfg.cell_format,
            delta_scale=cfg.delta_scale,
            delta_fmt=cfg.delta_format,
        )
        out_path = args.output_dir / f"tab_controls_{cfg.name}.tex"
        out_path.write_text(latex)
        print(f"wrote {out_path}")
        summary["tasks"].append(task_result)

        # Track the worst-case absolute delta on the natural scale (no scale mult)
        for entry in task_result.get("models", []):
            for cond, cell in entry["cells"].items():
                if cond == cfg.reference:
                    continue
                d = abs(cell["delta"])
                if not math.isnan(d) and d > max_abs_delta:
                    max_abs_delta = d
                    max_abs_loc = (cfg.name, entry["model"], cond)

    summary["max_abs_delta"] = max_abs_delta
    summary["max_abs_delta_location"] = max_abs_loc

    # TOST tally across all (task, model, non-ref-condition) cells
    tost_pass = 0
    tost_total = 0
    failing_cells: list[tuple[str, str, str, float, float, float]] = []
    for task_result in summary["tasks"]:
        ref = task_result["reference"]
        for entry in task_result["models"]:
            for cond, cell in entry["cells"].items():
                if cond == ref:
                    continue
                tost_total += 1
                if cell["tost_pass"]:
                    tost_pass += 1
                else:
                    failing_cells.append((
                        task_result["task"], entry["model"], cond,
                        cell["delta"], cell["ci90_lo"], cell["ci90_hi"],
                    ))
    summary["tost"] = {
        "pass": tost_pass,
        "total": tost_total,
        "failing_cells": failing_cells,
    }
    out_summary = summary_dir / "controls_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_summary}")

    if max_abs_loc is not None:
        task, model, cond = max_abs_loc
        print(f"Largest |delta|: {max_abs_delta:.4f} at task={task}, model={model}, cond={cond}")
    print(f"TOST pass: {tost_pass}/{tost_total}")
    if failing_cells:
        print("Failing cells:")
        for task, model, cond, d, lo, hi in failing_cells:
            print(f"  {task} {model} {cond}: delta={d:+.3f} 90% CI [{lo:+.3f}, {hi:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
