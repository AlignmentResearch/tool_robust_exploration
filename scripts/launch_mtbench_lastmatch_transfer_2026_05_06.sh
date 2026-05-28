#!/bin/bash
# Transfer eval for the MT-Bench last-match rerun. Sources from the new
# pair_matrix_2026-05-06_lastmatch/mtbench_* arms (trained against
# `--prompt-variant single_v1_last`) and evaluates under last-match
# parsing.
#
# Per-seed held-out: each (attacker, seed) branch is evaluated on a
# 40-item set deterministically derived from the seed string and disjoint
# from THAT seed's 20 search items (mirrors the existing first-match
# perseed transfer setup at `transfer-2026-04-30-perseed`).
#
# Run AFTER launch_mtbench_lastmatch_2026_05_06.sh completes.
#
# Output dirs use --target-sample-seed=transfer-2026-05-06-lastmatch.
# Update _transfer_data.TARGET_SEED_TAG_PER_TASK["mtbench"] to this value
# before regenerating the appendix table; the body table can keep
# pointing at first-match (transfer-2026-04-30-perseed).

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

VICTIMS=(
    "haiku|claude-haiku-4-5|anthropic|"
    "gpt54|gpt-5.4|openai|gpt54_"
    "gpt54mini|gpt-5.4-mini|openai|gpt54mini_"
    "gemma4|google/gemma-4-26b-a4b-it|openrouter|gemma4_"
    "qwen35flash|qwen/qwen3.5-flash-02-23|openrouter|qwen35flash_"
    "qwen3-8b|qwen3-8b|openrouter|qwen3-8b_"
)

PIDS=()
for vic_spec in "${VICTIMS[@]}"; do
    IFS='|' read -r vshort victim provider srcprefix <<< "$vic_spec"
    log="logs/transfer_mtbench_${vshort}_lastmatch.log"

    src_kimi="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}kimi"
    src_gemini3="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}gemini3"
    src_v4pro="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}v4pro"

    echo "Launching mtbench / $vshort / lastmatch transfer (per-seed) -> $log"
    ( uv run python scripts/run_attack_transfer.py mtbench \
        --source-dir "$src_kimi" "$src_gemini3" "$src_v4pro" \
        --attack-source per_branch --top-k 1 --diagonal-only \
        --per-seed-held-out \
        --target-victim-models "$victim" --target-victim-provider "$provider" \
        --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
        --target-n-items 40 \
        --target-sample-seed transfer-2026-05-06-lastmatch \
        --prompt-variant single_v1_last \
        > "$log" 2>&1 ) &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} mtbench last-match transfer runs; waiting..."
wait "${PIDS[@]}"
echo "Done."
