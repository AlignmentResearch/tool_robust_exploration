"""Summary reporting for scalar judge evaluations."""

from __future__ import annotations

from collections import defaultdict
import math


def _is_successful_row(row: dict) -> bool:
    return row.get("status") == "ok" and row.get("score") is not None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * quantile
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _score_stats(rows: list[dict]) -> dict[str, int | float | None]:
    values = [float(row["score"]) for row in rows if _is_successful_row(row)]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
        }
    values = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
    }


def _score_shift(
    current: dict[str, int | float | None],
    reference: dict[str, int | float | None],
) -> dict[str, float | None]:
    shift: dict[str, float | None] = {}
    for key in ("mean", "median", "p90"):
        current_value = current.get(key)
        reference_value = reference.get(key)
        if current_value is None or reference_value is None:
            shift[key] = None
        else:
            shift[key] = float(current_value) - float(reference_value)
    return shift


def _round_stats(stats: dict[str, int | float | None]) -> dict[str, int | float | None]:
    rounded: dict[str, int | float | None] = {}
    for key, value in stats.items():
        if isinstance(value, int) or value is None:
            rounded[key] = value
        else:
            rounded[key] = round(value, 3)
    return rounded


def build_scalar_summary(
    results: list[dict],
    *,
    conditions: list[str],
    attack_labels: list[str],
    title: str,
    model: str,
    run_metadata: dict,
) -> dict:
    """Build grouped scalar summaries for score-based judge runs."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    by_cond_label: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    n_questions = len({row["q_index"] for row in results if "q_index" in row})

    failure_counts: dict[str, int] = defaultdict(int)
    for row in results:
        by_cond[row["condition"]].append(row)
        by_cond_label[row["condition"]][row["candidate_label"]].append(row)
        if not _is_successful_row(row):
            failure_counts[row.get("status", "unknown")] += 1

    summary = {
        "title": title,
        "model": model,
        "n_results": len(results),
        "n_successful_results": sum(1 for row in results if _is_successful_row(row)),
        "n_failed_results": sum(1 for row in results if not _is_successful_row(row)),
        "failure_breakdown": dict(failure_counts),
        "n_questions": n_questions,
        "n_attack_types": len(attack_labels),
        "run_metadata": run_metadata,
        "conditions": {},
    }

    for cond in conditions:
        cond_rows = by_cond.get(cond, [])
        non_attacked_rows = [
            row for row in cond_rows if not row.get("is_under_attack", False)
        ]
        attacked_rows = [
            row for row in cond_rows if row.get("is_under_attack", False)
        ]

        non_attacked_stats = _score_stats(non_attacked_rows)
        attacked_stats = _score_stats(attacked_rows)

        highest_mean_score_attack = None
        highest_mean_non_attacked = None
        per_label = {}
        non_attacked_labels = sorted({
            row["candidate_label"] for row in non_attacked_rows
        })
        attacked_labels_present = sorted({
            row["candidate_label"] for row in attacked_rows
        })
        ordered_labels = list(dict.fromkeys([
            *non_attacked_labels,
            *attack_labels,
            *attacked_labels_present,
        ]))

        for label in ordered_labels:
            label_rows = by_cond_label[cond].get(label, [])
            if not label_rows:
                continue
            stats = _score_stats(label_rows)
            per_label[label] = {
                "n": len(label_rows),
                "parsed_n": stats["n"],
                "is_under_attack": any(
                    row.get("is_under_attack", False) for row in label_rows
                ),
                "score_stats": _round_stats(stats),
            }
            mean_score = stats["mean"]
            if per_label[label]["is_under_attack"] and mean_score is not None:
                if (
                    highest_mean_score_attack is None
                    or mean_score > highest_mean_score_attack["score_stats"]["mean"]
                ):
                    highest_mean_score_attack = {
                        "candidate_label": label,
                        "score_stats": _round_stats(stats),
                    }
            elif mean_score is not None:
                if (
                    highest_mean_non_attacked is None
                    or mean_score > highest_mean_non_attacked["score_stats"]["mean"]
                ):
                    highest_mean_non_attacked = {
                        "candidate_label": label,
                        "score_stats": _round_stats(stats),
                    }

        summary["conditions"][cond] = {
            "parsed_n": sum(1 for row in cond_rows if _is_successful_row(row)),
            "total_n": len(cond_rows),
            "non_attacked_labels": non_attacked_labels,
            "attacked_labels": attacked_labels_present,
            "non_attacked_score_stats": _round_stats(non_attacked_stats),
            "attacked_score_stats": _round_stats(attacked_stats),
            "attacked_shift_vs_non_attacked": _round_stats(
                _score_shift(attacked_stats, non_attacked_stats)
            ),
            "highest_mean_non_attacked": highest_mean_non_attacked,
            "highest_mean_score_attack": highest_mean_score_attack,
            "per_label": per_label,
        }

    return summary


def _fmt_score(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def print_scalar_condition_metrics(summary: dict) -> None:
    """Print compact scalar score summaries per condition."""
    print()
    print("Condition score stats:")
    for cond, metrics in summary["conditions"].items():
        non_attacked = metrics["non_attacked_score_stats"]
        attacked = metrics["attacked_score_stats"]
        shift = metrics["attacked_shift_vs_non_attacked"]
        top_non_attacked = metrics["highest_mean_non_attacked"]
        top_attack = metrics["highest_mean_score_attack"]

        top_non_attacked_text = "n/a"
        if top_non_attacked is not None:
            top_non_attacked_text = (
                f"{top_non_attacked['candidate_label']} "
                f"(mean={_fmt_score(top_non_attacked['score_stats']['mean'])})"
            )

        top_attack_text = "n/a"
        if top_attack is not None:
            top_attack_text = (
                f"{top_attack['candidate_label']} "
                f"(mean={_fmt_score(top_attack['score_stats']['mean'])})"
            )

        print(
            f"  {cond:<15} "
            f"non_attacked mean/med/p90={_fmt_score(non_attacked['mean'])}/{_fmt_score(non_attacked['median'])}/{_fmt_score(non_attacked['p90'])} "
            f"attacked mean/med/p90={_fmt_score(attacked['mean'])}/{_fmt_score(attacked['median'])}/{_fmt_score(attacked['p90'])} "
            f"attack-clean delta={_fmt_score(shift['mean'])}/{_fmt_score(shift['median'])}/{_fmt_score(shift['p90'])} "
            f"top_non_attacked={top_non_attacked_text} "
            f"top_attack={top_attack_text}"
        )


def print_scalar_label_metrics(summary: dict) -> None:
    """Print per-label scalar averages for each condition."""
    print()
    print("Per-label score stats:")
    for cond, metrics in summary["conditions"].items():
        print(f"  {cond}:")
        for label, label_metrics in metrics["per_label"].items():
            stats = label_metrics["score_stats"]
            bucket = "attack" if label_metrics["is_under_attack"] else "clean"
            print(
                f"    {label:<24} [{bucket}] "
                f"n={label_metrics['n']:<3} parsed={label_metrics['parsed_n']:<3} "
                f"mean/med/p90={_fmt_score(stats['mean'])}/{_fmt_score(stats['median'])}/{_fmt_score(stats['p90'])}"
            )


def print_scalar_parse_failures(results: list[dict]) -> None:
    """Print a compact sample of parse failures and other scalar eval errors."""
    failures = [row for row in results if not _is_successful_row(row)]
    print()
    if not failures:
        print("All scalar responses parsed successfully.")
        return
    print(f"{len(failures)} scalar evaluation(s) failed or did not parse.")
    for row in failures[:10]:
        print(
            f"  q{row.get('q_index', '?')} {row.get('condition', '?')} "
            f"{row.get('candidate_label', '?')} [{row.get('status', 'unknown')}]: "
            f"{row.get('raw', '')[:120]!r}"
        )
