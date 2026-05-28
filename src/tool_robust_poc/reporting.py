"""Summary reporting for judge robustness experiments.

Prints condition-comparison tables and exposes structured summary metrics for
single runs and model sweeps. Task-agnostic: works with any result dicts that
have the standard fields from runner_utils.eval_one.
"""

from __future__ import annotations

from collections import defaultdict


def _is_control_label(label: str, *, control_prefix: str = "control") -> bool:
    return label.startswith(control_prefix)


def _make_fraction(num: int, den: int) -> dict[str, int | float | None]:
    return {
        "num": num,
        "den": den,
        "rate": (num / den) if den else None,
    }


def _is_successful_row(row: dict) -> bool:
    return row.get("status", "ok") == "ok"


def _get_item_field(item: object, field_name: str) -> object | None:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def build_experiment_summary(
    results: list[dict],
    *,
    conditions: list[str],
    attack_labels: list[str] | None = None,
    title: str | None = None,
    model: str | None = None,
    control_prefix: str = "control",
    run_metadata: dict | None = None,
) -> dict:
    """Compute reusable summary metrics for an experiment run."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    by_cond_label: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_questions = len({r["q_index"] for r in results if "q_index" in r})

    for row in results:
        cond = row["condition"]
        label = row["candidate_label"]
        by_cond[cond].append(row)
        by_cond_label[cond][label].append(row)

    if attack_labels is None:
        attack_labels = sorted({
            r["candidate_label"]
            for r in results
            if not _is_control_label(r["candidate_label"], control_prefix=control_prefix)
        })
    else:
        attack_labels = list(attack_labels)

    # Breakdown of failure types
    failure_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if not _is_successful_row(r):
            failure_counts[r.get("status", "unknown")] += 1

    summary = {
        "title": title,
        "model": model,
        "n_results": len(results),
        "n_successful_results": sum(1 for r in results if _is_successful_row(r)),
        "n_failed_results": sum(1 for r in results if not _is_successful_row(r)),
        "failure_breakdown": dict(failure_counts),
        "n_questions": n_questions,
        "n_attack_types": len(attack_labels),
        "run_metadata": run_metadata or {},
        "conditions": {},
    }

    baseline_clean_rate = None
    if "baseline" in by_cond:
        baseline_controls = [
            r for r in by_cond["baseline"]
            if _is_successful_row(r)
            if _is_control_label(r["candidate_label"], control_prefix=control_prefix)
        ]
        if baseline_controls:
            baseline_clean_rate = sum(r["ok"] for r in baseline_controls) / len(
                baseline_controls
            )

    for cond in conditions:
        cond_rows = by_cond.get(cond, [])
        successful_cond_rows = [r for r in cond_rows if _is_successful_row(r)]
        failed_cond_rows = [r for r in cond_rows if not _is_successful_row(r)]
        control_rows = [
            r for r in successful_cond_rows
            if _is_control_label(r["candidate_label"], control_prefix=control_prefix)
        ]
        adv_rows = [
            r for r in successful_cond_rows
            if not _is_control_label(r["candidate_label"], control_prefix=control_prefix)
        ]

        clean_ok = sum(r["ok"] for r in control_rows)
        adv_ok = sum(r["ok"] for r in adv_rows)
        adv_attack_success = sum(r["attack_success"] for r in adv_rows)

        attack_success_count = 0
        worst_case = None
        attacks_by_label = {}
        any_attack_q_indices: set[int] = set()

        for label in attack_labels:
            all_rows = by_cond_label[cond].get(label, [])
            rows = [r for r in all_rows if _is_successful_row(r)]
            if not all_rows:
                continue
            success_num = sum(r["attack_success"] for r in rows)
            acc_num = sum(r["ok"] for r in rows)
            fraction = _make_fraction(success_num, len(rows))
            attacks_by_label[label] = {
                "accuracy": _make_fraction(acc_num, len(rows)),
                "attack_success": fraction,
                "failed_evaluations": _make_fraction(
                    len(all_rows) - len(rows),
                    len(all_rows),
                ),
            }
            if success_num > 0:
                attack_success_count += 1
                any_attack_q_indices.update(
                    r["q_index"] for r in rows if r["attack_success"] and "q_index" in r
                )
            if worst_case is None or (fraction["rate"] or 0.0) > (
                worst_case["attack_success"]["rate"] or 0.0
            ):
                worst_case = {
                    "candidate_label": label,
                    "attack_success": fraction,
                }

        clean_rate = (clean_ok / len(control_rows)) if control_rows else None
        clean_delta = (
            clean_rate - baseline_clean_rate
            if clean_rate is not None and baseline_clean_rate is not None
            else None
        )

        summary["conditions"][cond] = {
            "successful_evaluations": _make_fraction(
                len(successful_cond_rows),
                len(cond_rows),
            ),
            "failed_evaluations": _make_fraction(
                len(failed_cond_rows),
                len(cond_rows),
            ),
            "clean_accuracy": _make_fraction(clean_ok, len(control_rows)),
            "clean_accuracy_delta_vs_baseline": clean_delta,
            "adversarial_accuracy": _make_fraction(adv_ok, len(adv_rows)),
            "attack_success": _make_fraction(adv_attack_success, len(adv_rows)),
            "attack_success_count": {
                "num": attack_success_count,
                "den": len(attack_labels),
                "rate": (
                    attack_success_count / len(attack_labels)
                    if attack_labels else None
                ),
            },
            "worst_case_attack_success": worst_case,
            "any_attack_success_per_question": _make_fraction(
                len(any_attack_q_indices),
                n_questions,
            ),
            "per_attack": attacks_by_label,
        }

    return summary


def print_summary_table(
    results: list[dict],
    *,
    conditions: list[str],
    candidate_labels: list[str],
    control_labels: list[str] = ("control_correct", "control_wrong"),
    n_questions: int,
    model: str,
    title: str = "GSM8K",
    attack_col_header: str = "Atk",
) -> None:
    """Print a per-candidate accuracy table across conditions."""
    by_cond_label: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        by_cond_label[r["condition"]][r["candidate_label"]].append(r)

    failed_total = sum(1 for r in results if not _is_successful_row(r))

    all_labels = list(control_labels) + list(candidate_labels)

    header = f"{'Candidate':<25} " + "  ".join(
        f"{'Acc':>7} {attack_col_header:>5}" for _ in conditions
    )
    cond_header = f"{'':25} " + "  ".join(f"{c:>13}" for c in conditions)

    print(f"{title}: {n_questions} questions, model: {model}")
    if failed_total:
        print(
            f"Warning: {failed_total} evaluation(s) failed and are excluded from "
            "accuracy / attack denominators."
        )
    print()
    print(cond_header)
    print(header)
    print("-" * len(header))

    for label in all_labels:
        parts = []
        for cond in conditions:
            rows = [r for r in by_cond_label[cond][label] if _is_successful_row(r)]
            n = len(rows)
            if n == 0:
                parts.append(f"{'':>13}")
                continue
            acc = sum(r["ok"] for r in rows)
            if label in control_labels:
                parts.append(f"{acc:>3}/{n}   -  ")
            else:
                atk = sum(r["attack_success"] for r in rows)
                parts.append(f"{acc:>3}/{n} {atk:>3}/{n}")
        print(f"{label:<25} " + "  ".join(parts))


def print_condition_metrics(summary: dict) -> None:
    """Print headline condition-level metrics used for comparison."""
    print()
    print("Condition metrics:")
    for cond, metrics in summary["conditions"].items():
        failed = metrics["failed_evaluations"]
        clean = metrics["clean_accuracy"]
        delta = metrics["clean_accuracy_delta_vs_baseline"]
        asc = metrics["attack_success_count"]
        worst = metrics["worst_case_attack_success"]
        any_attack = metrics["any_attack_success_per_question"]

        clean_text = (
            f"{clean['num']}/{clean['den']} ({100 * clean['rate']:.1f}%)"
            if clean["rate"] is not None else "n/a"
        )
        failed_text = (
            f"{failed['num']}/{failed['den']} ({100 * failed['rate']:.1f}%)"
            if failed["rate"] is not None else "n/a"
        )
        delta_text = f"{delta:+.1%}" if delta is not None else "n/a"
        asc_text = (
            f"{asc['num']}/{asc['den']} ({100 * asc['rate']:.1f}%)"
            if asc["rate"] is not None else "n/a"
        )
        any_attack_text = (
            f"{any_attack['num']}/{any_attack['den']} ({100 * any_attack['rate']:.1f}%)"
            if any_attack["rate"] is not None else "n/a"
        )
        if worst is None or worst["attack_success"]["rate"] is None:
            worst_text = "n/a"
        else:
            worst_rate = worst["attack_success"]["rate"]
            worst_text = (
                f"{worst['candidate_label']} "
                f"({worst['attack_success']['num']}/{worst['attack_success']['den']}, "
                f"{100 * worst_rate:.1f}%)"
            )

        print(
            f"  {cond:<15} failed={failed_text}  "
            f"clean={clean_text}  "
            f"delta_vs_baseline={delta_text}  "
            f"ASC={asc_text}  "
            f"worst_case={worst_text}  "
            f"any_attack_per_q={any_attack_text}"
        )


def print_adversarial_totals(
    results: list[dict],
    *,
    conditions: list[str],
    attack_label: str = "Attack success",
    control_prefix: str = "control",
) -> None:
    """Print aggregate attack success rates for adversarial candidates."""
    print()
    print("Adversarial totals:")
    for cond in conditions:
        adv_rows = [
            r for r in results
            if _is_successful_row(r)
            if r["condition"] == cond
            and not _is_control_label(r["candidate_label"], control_prefix=control_prefix)
        ]
        failed_rows = [
            r for r in results
            if (not _is_successful_row(r))
            and r["condition"] == cond
            and not _is_control_label(r["candidate_label"], control_prefix=control_prefix)
        ]
        total = len(adv_rows)
        if total == 0 and not failed_rows:
            continue
        atk = sum(r["attack_success"] for r in adv_rows)
        acc = sum(r["ok"] for r in adv_rows)
        failed_text = (
            f"  failed={len(failed_rows)}/{len(adv_rows) + len(failed_rows)}"
            if failed_rows else ""
        )
        if total == 0:
            print(f"  {cond:<15} acc=n/a  {attack_label}=n/a{failed_text}")
            continue
        print(
            f"  {cond:<15} acc={acc}/{total} ({100*acc/total:.1f}%)  "
            f"{attack_label}={atk}/{total} ({100*atk/total:.1f}%)"
            f"{failed_text}"
        )


def print_attack_details(
    results: list[dict],
    *,
    questions: list[object],
    question_key: str = "question",
    reference_key: str | None = "answer",
) -> None:
    """Print detailed info for each attack success."""
    print()
    successes = [r for r in results if r["attack_success"]]
    if not successes:
        print("No attack successes.")
        return

    print(f"=== {len(successes)} total attack success(es) ===")
    by_q: dict[tuple, list[dict]] = defaultdict(list)
    for r in successes:
        by_q[(r["q_index"], r["candidate_label"])].append(r)

    for (qi, label), group in sorted(by_q.items()):
        question_text = _get_item_field(questions[qi], question_key)
        q_text = str(question_text)[:80] if question_text is not None else "<missing question>"
        cand = group[0]["candidate"]
        reference_value = (
            _get_item_field(questions[qi], reference_key)
            if reference_key is not None
            else None
        )
        if reference_value is not None:
            ref_text = f" (ref={reference_value})"
        else:
            ref_text = ""
        print(f"\n--- q{qi} / {label}{ref_text} ---")
        print(f"Q: {q_text}...")
        print(f"Candidate: {cand!r}")
        for r in group:
            print(f"  {r['condition']:<15} [{r['verdict']}]: {r['raw'][:100]}")
