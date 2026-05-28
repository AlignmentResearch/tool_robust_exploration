"""Build the MT-Bench parser-sensitivity appendix table.

Two-panel table:
  * Top: first-match parser (identical to the MT-Bench panel of Table 1)
  * Bottom: last-match parser, sourced from a *separate* PAIR sweep
    whose optimizer scored against last-match ASR. The rerun-mode tags
    below point at those independent transfer dirs.

Rerun-mode is mandatory in this driver: the alternative (reparsing the
first-match-optimized attacks under a last-match scorer) was a known
caveat in earlier drafts; the body now claims an "independent PAIR
rerun" so we must source from the rerun dirs to match.

Usage:
    cd tool_robust_poc
    uv run python scripts/table_gen/build_tab_pair_transfer_mtbench_parser_compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from _transfer_tables import (  # noqa: E402
    emit_mtbench_parser_compare,
    task_rngs_from_seed,
)

# Canonical args — change here, not via CLI.
B = 10_000
SEED = 0
CI = 0.95
GENERATOR = "scripts/table_gen/build_tab_pair_transfer_mtbench_parser_compare.py"

# Tags for the bottom-panel PAIR rerun. The non-Sonnet victims live
# under transfer-2026-05-06-lastmatch; Sonnet was added later under
# transfer-2026-05-09-sonnet-lastmatch. Both used prompt_variant
# single_v1_last so the attacker optimized against last-match ASR.
RERUN_TAGS = (
    "transfer-2026-05-06-lastmatch",
    "transfer-2026-05-09-sonnet-lastmatch",
)

POC_ROOT = _HERE.parent.parent
DEFAULT_RESULTS_DIR = POC_ROOT / "results"
DEFAULT_OUTPUT_DIR = POC_ROOT / "writing" / "base_article" / "tables_pair"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out = args.output_dir / "tab_pair_transfer_mtbench_parser_compare.tex"
    body_table = args.output_dir / "tab_pair_transfer_all.tex"
    # Top panel is copied verbatim from the body Table 1 file when it
    # exists; ``top_fallback_rng`` matches Table 1's per-task mtbench
    # RNG so a fallback regen would also byte-match.  Bottom panel
    # reads a different transfer tag and bootstraps independently.
    top_fallback_rng = task_rngs_from_seed(SEED)["mtbench"]
    bottom_rng = np.random.default_rng(SEED + 1)
    emit_mtbench_parser_compare(
        args.results_dir, B, CI, top_fallback_rng, bottom_rng, out,
        bottom_panel_seed_tag=RERUN_TAGS,
        body_table_path=body_table,
        generator=GENERATOR,
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
