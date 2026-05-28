#!/bin/bash
# Arena-Hard pinned-B transfer rerun at n=60. The main n=60 launcher
# (launch_arena_hard_transfer_n60_2026_05_10.sh) covers only the
# slot-unaware variant; this script bumps the pinned-B ablation to
# match. Same cache-extension trick: target_sample_seed is reused so
# the first 40 items hit cache and only items 41-60 are new requests.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

PIDS=()

# GPT-5.4 (sources in pair_matrix_pinned_b_2026-05-09)
log="logs/transfer_arena_hard_pinned_b_gpt54_n60.log"
echo "Launching arena_hard / gpt54 / pinned-B / n=60 transfer -> $log"
( uv run python scripts/run_attack_transfer.py arena_hard \
    --source-dir \
        "results/pair_matrix_pinned_b_2026-05-09/arena_hard_gpt54_kimi" \
        "results/pair_matrix_pinned_b_2026-05-09/arena_hard_gpt54_gemini3" \
        "results/pair_matrix_pinned_b_2026-05-09/arena_hard_gpt54_v4pro" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "gpt-5.4" --target-victim-provider "openai" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 60 \
    --target-sample-seed transfer-2026-05-09-pinned-b \
    --prompt-variant pinned_b \
    --attack-mode replace \
    --slot-seed transfer-2026-04-27 \
    > "$log" 2>&1 ) &
PIDS+=($!)

# GPT-5.4-mini (sources in pair_matrix_pinned_b_2026-05-08)
log="logs/transfer_arena_hard_pinned_b_gpt54mini_n60.log"
echo "Launching arena_hard / gpt54mini / pinned-B / n=60 transfer -> $log"
( uv run python scripts/run_attack_transfer.py arena_hard \
    --source-dir \
        "results/pair_matrix_pinned_b_2026-05-08/arena_hard_gpt54mini_kimi" \
        "results/pair_matrix_pinned_b_2026-05-08/arena_hard_gpt54mini_gemini3" \
        "results/pair_matrix_pinned_b_2026-05-08/arena_hard_gpt54mini_v4pro" \
    --attack-source per_branch --top-k 1 --diagonal-only \
    --per-seed-held-out \
    --target-victim-models "gpt-5.4-mini" --target-victim-provider "openai" \
    --target-conditions baseline multi_msg system_distrust tool_wrapped_v2 tool_distrust_v2 \
    --target-n-items 60 \
    --target-sample-seed transfer-2026-05-09-pinned-b \
    --prompt-variant pinned_b \
    --attack-mode replace \
    --slot-seed transfer-2026-04-27 \
    > "$log" 2>&1 ) &
PIDS+=($!)

echo "Launched ${#PIDS[@]} pinned-B transfer runs; waiting..."
wait "${PIDS[@]}"
echo "=== Pinned-B n=60 transfer rerun complete ==="
