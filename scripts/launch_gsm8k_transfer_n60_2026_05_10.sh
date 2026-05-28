#!/bin/bash
# GSM8K transfer rerun at n=60.  Matches the MT-Bench and Arena-Hard
# n=60 bumps done earlier today so the body transfer table has
# consistent per-cell precision across all three tasks.
#
# Cache reuse strategy: keep the same target_sample_seed as the n=40
# runs so the first 40 items per branch hit cache (deterministic
# sample with order="hash" extends the prefix).  The output dirs
# overwrite the old n=40 transfer dirs, which is fine -- the n=60
# data is a strict superset and the n=40 data lives in git history.
#
# Provider routing matches the original launchers (auto-selects
# cheapest tier each provider supports):
#   anthropic  -> Batch  (50% off)   - haiku, sonnet
#   openai     -> Flex   (50% off)   - gpt-5.4, gpt-5.4-mini
#   openrouter -> Std parallel       - gemma, qwen
#
# Sonnet uses its own seed tag (transfer-2026-05-09-sonnet-gsm8k)
# since its source PAIR data layout differs from the standard
# 6-victim matrix.
#
# Cross-seed item overlap remains harmless under per-seed-held-out:
# each seed's PAIR loop is independent, so an item only "leaks" to
# the optimizer if it was in THAT seed's own 20 search items.  At
# 6 seeds * 60 items / ~1100 pool ~= 33%, each seed still gets a
# fresh random GSM8K sample.

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

SONNET_VICTIM="claude-sonnet-4.6"
SONNET_PROVIDER="anthropic"

echo "=== GSM8K last-match transfer at n=60 ==="
PIDS=()
for vic_spec in "${VICTIMS[@]}"; do
    IFS='|' read -r vshort victim provider srcprefix <<< "$vic_spec"
    log="logs/transfer_gsm8k_${vshort}_n60.log"

    src_kimi="results/pair_matrix_2026-05-02/gsm8k_${srcprefix}kimi"
    src_gemini3="results/pair_matrix_2026-05-02/gsm8k_${srcprefix}gemini3"
    src_v4pro="results/pair_matrix_2026-05-02/gsm8k_${srcprefix}v4pro"

    echo "Launching gsm8k / $vshort / n=60 transfer -> $log"
    ( uv run python scripts/run_attack_transfer.py gsm8k \
        --source-dir "$src_kimi" "$src_gemini3" "$src_v4pro" \
        --attack-source per_branch --top-k 1 --diagonal-only \
        --per-seed-held-out \
        --target-victim-models "$victim" --target-victim-provider "$provider" \
        --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
        --target-n-items 60 \
        --target-sample-seed transfer-2026-05-02-lastmatch \
        --prompt-variant verdict_last \
        --data data/gsm8k_full_pool.json \
        > "$log" 2>&1 ) &
    PIDS+=($!)
done
# Sonnet uses its own seed tag for cache-hit alignment with the prior
# Sonnet-only n=40 transfer run.
log="logs/transfer_gsm8k_sonnet_n60.log"
echo "Launching gsm8k / sonnet / n=60 transfer -> $log"
( uv run python scripts/run_attack_transfer.py gsm8k \
    --source-dir \
        "results/pair_matrix_sonnet_2026-05-09_gsm8k/gsm8k_sonnet_kimi" \
        "results/pair_matrix_sonnet_2026-05-09_gsm8k/gsm8k_sonnet_gemini3" \
        "results/pair_matrix_sonnet_2026-05-09_gsm8k/gsm8k_sonnet_v4pro" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "$SONNET_VICTIM" --target-victim-provider "$SONNET_PROVIDER" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 60 \
    --target-sample-seed transfer-2026-05-09-sonnet-gsm8k \
    --prompt-variant verdict_last \
    --data data/gsm8k_full_pool.json \
    > "$log" 2>&1 ) &
PIDS+=($!)

echo "Launched ${#PIDS[@]} gsm8k transfer runs; waiting..."
wait "${PIDS[@]}"
echo "=== GSM8K n=60 transfer rerun complete ==="
