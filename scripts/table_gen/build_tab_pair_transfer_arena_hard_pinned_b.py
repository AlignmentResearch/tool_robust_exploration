"""Build the Arena-Hard pinned-B appendix table.

Two-panel ablation: the slot-unaware Arena-Hard rows from Table 1 on
top, and an independent PAIR rerun on the bottom where the attacked
response is always placed in Assistant B (and the attacker is told
this fact). GPT models only.

The top panel is copied verbatim from
``tab_pair_transfer_all.tex`` (if present) so the two tables agree
byte-for-byte; otherwise a fresh bootstrap reproduces it.

Usage:
    cd tool_robust_poc
    uv run python scripts/table_gen/build_tab_pair_transfer_arena_hard_pinned_b.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

from _transfer_tables import (  # noqa: E402
    emit_arena_hard_pinned_b_compare,
    task_rngs_from_seed,
)

# Canonical args — change here, not via CLI.
B = 10_000
SEED = 0
CI = 0.95
PINNED_B_TAG = "transfer-2026-05-09-pinned-b"
GENERATOR = "scripts/table_gen/build_tab_pair_transfer_arena_hard_pinned_b.py"

POC_ROOT = _HERE.parent.parent
DEFAULT_RESULTS_DIR = POC_ROOT / "results"
DEFAULT_OUTPUT_DIR = POC_ROOT / "writing" / "base_article" / "tables_pair"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out = args.output_dir / "tab_pair_transfer_arena_hard_pinned_b_gpt54mini.tex"
    body_table = args.output_dir / "tab_pair_transfer_all.tex"
    # Top-panel fallback (only fires if the body .tex is missing): match
    # Table 1's per-task arena_hard RNG so a regen produces byte-
    # identical numbers.  Bottom panel reads the pinned-B transfer tag,
    # so its RNG is independent.
    top_fallback_rng = task_rngs_from_seed(SEED)["arena_hard"]
    bottom_rng = np.random.default_rng(SEED + 1)
    emit_arena_hard_pinned_b_compare(
        args.results_dir, B, CI, top_fallback_rng, bottom_rng, out,
        pinned_b_tag=PINNED_B_TAG,
        body_table_path=body_table,
        generator=GENERATOR,
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
