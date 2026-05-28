#!/bin/bash
# Full MT-Bench PAIR matrix rerun under the last-match parser.
#
# Reruns the PAIR optimization with the judge scored by `--prompt-variant
# single_v1_last` (parses the LAST "[[N]]" token in the judge's output,
# not the first). Motivation: the FastChat first-match parser is fooled
# by careful judges (e.g. Haiku-4.5 / Qwen3.5-flash UO baseline) that
# quote the candidate's injected "[[10]]" while writing their own
# different rating later. Last-match captures the judge's actual final
# rating; under that parser the IH-inversion picture sharpens (see
# Appendix B / `mtbench_alt_parser_table.md`).
#
# Distinct from the cell-level reparse already in
# `tab_pair_transfer_mtbench_parser_compare.tex`: that was the SAME
# attacks reparsed under last-match. This rerun lets the PAIR optimizer
# see last-match ASR as its reward signal during search, so it
# discovers attacks that work AGAINST last-match (e.g., format
# truncation tricks that prevent the judge from emitting its own
# rating) rather than first-match injection-only attacks the parser
# would now ignore.
#
# 6 victims × 3 attackers = 18 arms.  Each arm: 6 seeds × 5 conditions
# × 7 turns × 20 search items.
#
# Run AFTER this completes:
#   bash scripts/launch_mtbench_lastmatch_transfer_2026_05_06.sh

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

# victim_short | victim_canonical | provider | output_prefix
VICTIMS=(
    "haiku|claude-haiku-4-5|anthropic|"
    "gpt54|gpt-5.4|openai|gpt54_"
    "gpt54mini|gpt-5.4-mini|openai|gpt54mini_"
    "gemma4|google/gemma-4-26b-a4b-it|openrouter|gemma4_"
    "qwen35flash|qwen/qwen3.5-flash-02-23|openrouter|qwen35flash_"
    "qwen3-8b|qwen3-8b|openrouter|qwen3-8b_"
)
ATTACKERS=(
    "kimi|moonshotai/kimi-k2.5|openrouter"
    "gemini3|google/gemini-3-flash-preview|openrouter"
    "v4pro|deepseek/deepseek-v4-pro|openrouter"
)

PIDS=()
for vic_spec in "${VICTIMS[@]}"; do
    IFS='|' read -r vshort victim provider srcprefix <<< "$vic_spec"
    for atk_spec in "${ATTACKERS[@]}"; do
        IFS='|' read -r atkshort attacker atkprov <<< "$atk_spec"
        outdir="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}${atkshort}"
        log="logs/matrix_mtbench_${vshort}_${atkshort}_lastmatch.log"
        echo "Launching mtbench / $vshort / $atkshort / matrix-a..f -> $outdir"
        # shellcheck disable=SC2086
        ( uv run python scripts/run_pair_matrix.py mtbench \
            --victim-model "$victim" --victim-provider "$provider" \
            --attacker-model "$attacker" --attacker-provider "$atkprov" \
            --conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
            --sample-seeds matrix-a matrix-b matrix-c matrix-d matrix-e matrix-f \
            --n-items 20 --n-turns 7 \
            --prompt-variant single_v1_last \
            --output-dir "$outdir" \
            > "$log" 2>&1 ) &
        PIDS+=($!)
    done
done

echo "Launched ${#PIDS[@]} mtbench last-match matrix arms; waiting..."
wait "${PIDS[@]}"
echo "Done."
