"""Build body Table 1 (mean ASR) and the prose-deltas appendix table.

These two outputs share a single bootstrap pass (the prose-deltas
table needs the superset 5-contrast CIs that the mean variant already
computes) so they live in one driver.

Usage:
    cd tool_robust_poc
    uv run python scripts/table_gen/build_tab_pair_transfer_all.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bump sys.path so the shared library and its sibling helpers resolve.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from _transfer_tables import (  # noqa: E402
    DIFF_COLUMNS_FULL,
    build_all_tasks_data,
    emit_stacked_deltas_prose,
    emit_stacked_latex,
)

# Canonical args — change here, not via CLI.
B = 10_000
SEED = 0
CI = 0.95
STATISTIC = "mean"
GENERATOR = "scripts/table_gen/build_tab_pair_transfer_all.py"

POC_ROOT = _HERE.parent.parent
DEFAULT_RESULTS_DIR = POC_ROOT / "results"
DEFAULT_OUTPUT_DIR = POC_ROOT / "writing" / "base_article" / "tables_pair"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_task, ci_per_task = build_all_tasks_data(
        args.results_dir, B, CI, SEED,
        statistic=STATISTIC,
        diff_columns=DIFF_COLUMNS_FULL,
    )

    out_main = args.output_dir / "tab_pair_transfer_all.tex"
    emit_stacked_latex(
        per_task, ci_per_task, B, CI, out_main,
        statistic=STATISTIC, generator=GENERATOR,
    )
    print(f"  wrote {out_main}")

    out_prose = args.output_dir / "tab_pair_transfer_deltas_prose.tex"
    emit_stacked_deltas_prose(
        per_task, ci_per_task, B, CI, out_prose, generator=GENERATOR,
    )
    print(f"  wrote {out_prose}")


if __name__ == "__main__":
    main()
