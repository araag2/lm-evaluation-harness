#!/bin/bash
# ================================================
# Qwen3-8B: Multi-turn CoT + Self-Refine CoT — Small Tasks (No Limit)
# ================================================
# Runs multi-turn_CoT, then self-refine_CoT, then extracts results for both.
#
# Usage:
#   bash examples/new_experiments/run_qwen3-8b_CoT_and_self-refine_small-tasks.sh
#   bash examples/new_experiments/run_qwen3-8b_CoT_and_self-refine_small-tasks.sh --dry-run
# ================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF="${SCRIPT_DIR}/qwen3-8b_multi-turn_CoT_and_self-refine_small-tasks.conf"

# Source utils for log helpers
source "${RUNNER_DIR}/lib/eval_utils.sh"

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

OUTPUT_BASE="../outputs/qwen3-8b_multi-turn_CoT+self-refine_small-tasks"
RESULTS_DIR="${OUTPUT_BASE}/results"
RESULTS_NAME="qwen3-8b_CoT_vs_self-refine_small-tasks"

print_separator
log_info "Qwen3-8B: Multi-turn CoT + Self-Refine CoT Pipeline"
log_info "Config:  $CONF"
log_info "Output:  $OUTPUT_BASE"
log_info "Dry run: $DRY_RUN"
print_separator

# -----------------------------------------------
# Step 1: Multi-turn CoT
# -----------------------------------------------
print_separator
log_info "Step 1/3 — Running multi-turn_CoT"
print_separator

CMD_COT="${RUNNER_DIR}/run_multi_turn.sh --config \"$CONF\" --mode multi-turn_CoT"
[ "$DRY_RUN" = true ] && CMD_COT="$CMD_COT --dry-run"

eval bash $CMD_COT

log_success "Step 1/3 complete — multi-turn_CoT finished"
echo ""

# -----------------------------------------------
# Step 2: Self-Refine CoT
# -----------------------------------------------
print_separator
log_info "Step 2/3 — Running self-refine_CoT"
print_separator

CMD_SR="${RUNNER_DIR}/run_multi_turn.sh --config \"$CONF\" --mode self-refine_CoT"
[ "$DRY_RUN" = true ] && CMD_SR="$CMD_SR --dry-run"

eval bash $CMD_SR

log_success "Step 2/3 complete — self-refine_CoT finished"
echo ""

# -----------------------------------------------
# Step 3: Extract results for both runs
# -----------------------------------------------
print_separator
log_info "Step 3/3 — Extracting results"
print_separator

CMD_EXTRACT="${RUNNER_DIR}/extract_results.sh \
    --input-list \"${OUTPUT_BASE}/multi-turn_CoT,${OUTPUT_BASE}/self-refine_CoT\" \
    --output \"${RESULTS_DIR}\" \
    --name \"${RESULTS_NAME}\""
[ "$DRY_RUN" = true ] && CMD_EXTRACT="$CMD_EXTRACT --dry-run"

eval bash $CMD_EXTRACT

log_success "Step 3/3 complete — results extracted to ${RESULTS_DIR}/${RESULTS_NAME}.*"
print_separator
log_success "All done!"
print_separator
