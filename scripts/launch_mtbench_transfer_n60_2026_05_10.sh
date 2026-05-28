#!/bin/bash
# MT-Bench transfer rerun at n=60 (the per-seed held-out cap given the
# 80-item question pool minus 20 search items per seed).  Goal: tighter
# per-cell CIs in the body table.
#
# Cache reuse strategy: keep the same target_sample_seed as the n=40 run
# so the first 40 items per branch hit cache (deterministic-sample with
# order="hash" extends the prefix).  The output dir overwrites the old
# n=40 transfer dir, which is fine -- the n=60 data is a strict superset
# and the n=40 data lives in git history if ever needed.
#
# Provider routing matches the original launchers (auto-selects cheapest
# tier each provider supports):
#   anthropic  -> Batch  (50% off)   - haiku, sonnet
#   openai     -> Flex   (50% off)   - gpt-5.4, gpt-5.4-mini
#   openrouter -> Std parallel       - gemma, qwen
#
# Sonnet uses its own seed tags (transfer-2026-05-09-sonnet,
# transfer-2026-05-09-sonnet-lastmatch) since its source PAIR data
# layout differs from the standard 6-victim matrix.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

# vshort | victim_canonical | provider | src_prefix (Haiku omits prefix)
# Provider matches the ORIGINAL transfer-2026-04-30-perseed launcher
# so cache_tag (which embeds api_model + target_sample_seed) matches
# and the first 40 items hit cache.
VICTIMS=(
    "haiku|claude-haiku-4-5|anthropic|"
    "gpt54|gpt-5.4|openai|gpt54_"
    "gpt54mini|gpt-5.4-mini|openai|gpt54mini_"
    "gemma4|google/gemma-4-26b-a4b-it|openrouter|gemma4_"
    "qwen35flash|qwen/qwen3.5-flash-02-23|openrouter|qwen35flash_"
    "qwen3-8b|qwen3-8b|openrouter|qwen3-8b_"
)

SONNET_VICTIM="claude-sonnet-4.6"
SONNET_PROVIDER="anthropic"

# ── Phase 1: single_v1 (first-match) transfer at n=60 ─────────────────
echo "=== Phase 1: single_v1 transfer at n=60 ==="
PIDS=()
for vic_spec in "${VICTIMS[@]}"; do
    IFS='|' read -r vshort victim provider srcprefix <<< "$vic_spec"
    log="logs/transfer_mtbench_${vshort}_n60.log"

    src_kimi="results/pair_matrix_2026-04-27/mtbench_${srcprefix}kimi"
    src_gemini3="results/pair_matrix_2026-04-27/mtbench_${srcprefix}gemini3"
    src_v4pro="results/pair_matrix_2026-04-27/mtbench_${srcprefix}v4pro"
    src_baseline_kimi="results/pair_matrix_2026-04-27/mtbench_${srcprefix}baseline_kimi"
    src_baseline_gemini3="results/pair_matrix_2026-04-27/mtbench_${srcprefix}baseline_gemini3"
    src_baseline_v4pro="results/pair_matrix_2026-04-27/mtbench_${srcprefix}v4pro_baseline"
    src_seeds_def_kimi="results/pair_matrix_2026-04-27/mtbench_${srcprefix}kimi_seeds_def"
    src_seeds_def_gemini3="results/pair_matrix_2026-04-27/mtbench_${srcprefix}gemini3_seeds_def"
    src_seeds_def_v4pro="results/pair_matrix_2026-04-27/mtbench_${srcprefix}v4pro_seeds_def"

    echo "Launching mtbench / $vshort / single_v1 n=60 transfer -> $log"
    ( uv run python scripts/run_attack_transfer.py mtbench \
        --source-dir \
            "$src_kimi" "$src_gemini3" "$src_v4pro" \
            "$src_baseline_kimi" "$src_baseline_gemini3" "$src_baseline_v4pro" \
            "$src_seeds_def_kimi" "$src_seeds_def_gemini3" "$src_seeds_def_v4pro" \
        --attack-source per_branch --top-k 1 --diagonal-only \
        --per-seed-held-out \
        --target-victim-models "$victim" --target-victim-provider "$provider" \
        --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
        --target-n-items 60 \
        --target-sample-seed transfer-2026-04-30-perseed \
        > "$log" 2>&1 ) &
    PIDS+=($!)
done
# Sonnet (first-match) uses its own seed tag for cache-hit alignment with
# the prior Sonnet-only n=40 transfer run.
log="logs/transfer_mtbench_sonnet_n60.log"
echo "Launching mtbench / sonnet / single_v1 n=60 transfer -> $log"
( uv run python scripts/run_attack_transfer.py mtbench \
    --source-dir \
        "results/pair_matrix_sonnet_2026-05-09/mtbench_sonnet_kimi" \
        "results/pair_matrix_sonnet_2026-05-09/mtbench_sonnet_gemini3" \
        "results/pair_matrix_sonnet_2026-05-09/mtbench_sonnet_v4pro" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "$SONNET_VICTIM" --target-victim-provider "$SONNET_PROVIDER" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 60 \
    --target-sample-seed transfer-2026-05-09-sonnet \
    > "$log" 2>&1 ) &
PIDS+=($!)

echo "Launched ${#PIDS[@]} single_v1 transfer runs; waiting..."
wait "${PIDS[@]}"
echo "Phase 1 complete."

# ── Phase 2: single_v1_last (last-match) transfer at n=60 ─────────────
echo "=== Phase 2: single_v1_last transfer at n=60 ==="
PIDS=()
for vic_spec in "${VICTIMS[@]}"; do
    IFS='|' read -r vshort victim provider srcprefix <<< "$vic_spec"
    log="logs/transfer_mtbench_${vshort}_n60_lastmatch.log"

    src_kimi="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}kimi"
    src_gemini3="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}gemini3"
    src_v4pro="results/pair_matrix_2026-05-06_lastmatch/mtbench_${srcprefix}v4pro"

    echo "Launching mtbench / $vshort / lastmatch n=60 transfer -> $log"
    ( uv run python scripts/run_attack_transfer.py mtbench \
        --source-dir "$src_kimi" "$src_gemini3" "$src_v4pro" \
        --attack-source per_branch --top-k 1 --diagonal-only \
        --per-seed-held-out \
        --target-victim-models "$victim" --target-victim-provider "$provider" \
        --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
        --target-n-items 60 \
        --target-sample-seed transfer-2026-05-06-lastmatch \
        --prompt-variant single_v1_last \
        > "$log" 2>&1 ) &
    PIDS+=($!)
done
# Sonnet last-match
log="logs/transfer_mtbench_sonnet_n60_lastmatch.log"
echo "Launching mtbench / sonnet / lastmatch n=60 transfer -> $log"
( uv run python scripts/run_attack_transfer.py mtbench \
    --source-dir \
        "results/pair_matrix_sonnet_2026-05-09_lastmatch/mtbench_sonnet_kimi" \
        "results/pair_matrix_sonnet_2026-05-09_lastmatch/mtbench_sonnet_gemini3" \
        "results/pair_matrix_sonnet_2026-05-09_lastmatch/mtbench_sonnet_v4pro" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "$SONNET_VICTIM" --target-victim-provider "$SONNET_PROVIDER" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 60 \
    --target-sample-seed transfer-2026-05-09-sonnet-lastmatch \
    --prompt-variant single_v1_last \
    > "$log" 2>&1 ) &
PIDS+=($!)

echo "Launched ${#PIDS[@]} lastmatch transfer runs; waiting..."
wait "${PIDS[@]}"
echo "Phase 2 complete."

echo "=== All MT-Bench n=60 transfer reruns complete ==="
