#!/bin/bash
# Overnight: full Sonnet-4.6 PAIR matrix + transfer for GSM8K and Arena-Hard.
#
# Mirrors the existing 6-victim matrix scope for these two tasks, so
# Sonnet rows can be added to the body's transfer table alongside the
# existing 6 victims (Haiku + 5 others).
#
# Cost-saver: --victim-provider anthropic auto-selects Batch strategy
# (~50% off, slower wall clock — fine for overnight). Total cost
# estimate ~$140-200; wall clock 3-6 hrs.
#
# Layout mirrors the May 2 launchers:
#   GSM8K     uses --prompt-variant verdict_last
#   Arena-Hard uses --attack-mode replace
#
# After PAIR completes, transfer eval pulls the same (attacker, seed)
# branches and re-evaluates on per-seed held-out items (n=40 per
# branch). Transfer victim is also Sonnet via Anthropic batch.
#
# Usage:
#   bash scripts/launch_sonnet_overnight_2026_05_09.sh

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

VICTIM_CANONICAL="claude-sonnet-4.6"
VICTIM_PROVIDER="anthropic"
VICTIM_SHORT="sonnet"

ATTACKERS=(
    "kimi|moonshotai/kimi-k2.5|openrouter"
    "gemini3|google/gemini-3-flash-preview|openrouter"
    "v4pro|deepseek/deepseek-v4-pro|openrouter"
)

# ------------------------------------------------------------------------
# Phase 1: GSM8K PAIR matrix (3 attackers in parallel)
# ------------------------------------------------------------------------
echo "================================================================"
echo "Phase 1: Sonnet GSM8K PAIR matrix"
echo "================================================================"

GSM8K_DIRS=()
PIDS=()
for atk_spec in "${ATTACKERS[@]}"; do
    IFS='|' read -r atkshort attacker atkprov <<< "$atk_spec"
    outdir="results/pair_matrix_sonnet_2026-05-09_gsm8k/gsm8k_${VICTIM_SHORT}_${atkshort}"
    GSM8K_DIRS+=("$outdir")
    log="logs/sonnet_overnight_gsm8k_${atkshort}.log"
    echo "Launching gsm8k / sonnet / $atkshort -> $outdir"
    ( uv run python scripts/run_pair_matrix.py gsm8k \
        --victim-model "$VICTIM_CANONICAL" --victim-provider "$VICTIM_PROVIDER" \
        --attacker-model "$attacker" --attacker-provider "$atkprov" \
        --conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
        --sample-seeds matrix-a matrix-b matrix-c matrix-d matrix-e matrix-f \
        --n-items 20 --n-turns 7 \
        --prompt-variant verdict_last \
        --data data/gsm8k_full_pool.json \
        --output-dir "$outdir" \
        > "$log" 2>&1 ) &
    PIDS+=($!)
done
echo "Launched ${#PIDS[@]} GSM8K Sonnet attacker branches; waiting..."
wait "${PIDS[@]}"
echo "GSM8K PAIR matrix complete."

# ------------------------------------------------------------------------
# Phase 2: Arena-Hard PAIR matrix (3 attackers in parallel)
# ------------------------------------------------------------------------
echo "================================================================"
echo "Phase 2: Sonnet Arena-Hard PAIR matrix"
echo "================================================================"

ARENA_DIRS=()
PIDS=()
for atk_spec in "${ATTACKERS[@]}"; do
    IFS='|' read -r atkshort attacker atkprov <<< "$atk_spec"
    outdir="results/pair_matrix_sonnet_2026-05-09_arena/arena_hard_${VICTIM_SHORT}_${atkshort}"
    ARENA_DIRS+=("$outdir")
    log="logs/sonnet_overnight_arena_${atkshort}.log"
    echo "Launching arena_hard / sonnet / $atkshort -> $outdir"
    ( uv run python scripts/run_pair_matrix.py arena_hard \
        --victim-model "$VICTIM_CANONICAL" --victim-provider "$VICTIM_PROVIDER" \
        --attacker-model "$attacker" --attacker-provider "$atkprov" \
        --conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
        --sample-seeds matrix-a matrix-b matrix-c matrix-d matrix-e matrix-f \
        --n-items 20 --n-turns 7 \
        --attack-mode replace \
        --output-dir "$outdir" \
        > "$log" 2>&1 ) &
    PIDS+=($!)
done
echo "Launched ${#PIDS[@]} Arena-Hard Sonnet attacker branches; waiting..."
wait "${PIDS[@]}"
echo "Arena-Hard PAIR matrix complete."

# ------------------------------------------------------------------------
# Phase 3: Transfer eval — GSM8K
# ------------------------------------------------------------------------
echo "================================================================"
echo "Phase 3: Sonnet GSM8K transfer eval"
echo "================================================================"
( uv run python scripts/run_attack_transfer.py gsm8k \
    --source-dir "${GSM8K_DIRS[@]}" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "$VICTIM_CANONICAL" --target-victim-provider "$VICTIM_PROVIDER" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 40 \
    --target-sample-seed transfer-2026-05-09-sonnet-gsm8k \
    --prompt-variant verdict_last \
    --data data/gsm8k_full_pool.json \
    > logs/sonnet_overnight_transfer_gsm8k.log 2>&1 )
echo "GSM8K transfer eval complete."

# ------------------------------------------------------------------------
# Phase 4: Transfer eval — Arena-Hard
# ------------------------------------------------------------------------
echo "================================================================"
echo "Phase 4: Sonnet Arena-Hard transfer eval"
echo "================================================================"
( uv run python scripts/run_attack_transfer.py arena_hard \
    --source-dir "${ARENA_DIRS[@]}" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "$VICTIM_CANONICAL" --target-victim-provider "$VICTIM_PROVIDER" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 40 \
    --target-sample-seed transfer-2026-05-09-sonnet-arena \
    --attack-mode replace \
    > logs/sonnet_overnight_transfer_arena.log 2>&1 )
echo "Arena-Hard transfer eval complete."

echo "================================================================"
echo "All Sonnet overnight runs complete."
echo "================================================================"
echo "GSM8K PAIR dirs:"
printf '  %s\n' "${GSM8K_DIRS[@]}"
echo "Arena-Hard PAIR dirs:"
printf '  %s\n' "${ARENA_DIRS[@]}"
echo "Transfer outputs in results/pair_gsm8k_transfer/ and results/pair_arena_hard_transfer/"
