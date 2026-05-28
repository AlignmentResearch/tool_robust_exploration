"""Build the p75 (worst-case-attacker) variant of Table 1.

Same layout as the body table but per-cell statistic is the 75th
percentile of per-branch ASR instead of the mean.  This captures threat
under a more-persistent attacker who tries multiple search seeds and
picks among the better ones.

Usage:
    cd tool_robust_poc
    uv run python scripts/table_gen/build_tab_pair_transfer_all_p75.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from _transfer_tables import (  # noqa: E402
    DIFF_COLUMNS,
    build_all_tasks_data,
    emit_stacked_latex,
)

# Canonical args — change here, not via CLI.
B = 10_000
SEED = 0
CI = 0.95
STATISTIC = "p75"
GENERATOR = "scripts/table_gen/build_tab_pair_transfer_all_p75.py"

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
        diff_columns=DIFF_COLUMNS,
    )

    out_p75 = args.output_dir / "tab_pair_transfer_all_p75.tex"
    emit_stacked_latex(
        per_task, ci_per_task, B, CI, out_p75,
        statistic=STATISTIC, generator=GENERATOR,
    )
    print(f"  wrote {out_p75}")


if __name__ == "__main__":
    main()
