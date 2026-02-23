#!/bin/bash
# ================================================
# Smoke Test — All Modes, Limit 1, Tiny Model
# ================================================
# Runs every evaluation mode with limit=1 and the smallest model
# (Qwen2-0.5B) to verify all tasks and pipelines are functional.
#
# Usage:
#   bash examples/new_experiments/run_smoke_test.sh
#   bash examples/new_experiments/run_smoke_test.sh --dry-run
# ================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF="${SCRIPT_DIR}/smoke_test.conf"

# Source utils for log helpers
source "${RUNNER_DIR}/lib/eval_utils.sh"

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

DRY_FLAG=""
[ "$DRY_RUN" = true ] && DRY_FLAG="--dry-run"

MODES_TOTAL=5
MODES_DONE=0
FAILED_MODES=()

print_separator
log_info "Smoke Test — All Modes"
log_info "Config:  $CONF"
log_info "Dry run: $DRY_RUN"
print_separator

# -----------------------------------------------
# Helper: run a step, track pass/fail
# -----------------------------------------------
run_step() {
    local step="$1"
    local total="$2"
    local label="$3"
    local cmd="$4"

    print_separator
    log_info "Step ${step}/${total} — ${label}"
    print_separator

    if eval bash $cmd $DRY_FLAG; then
        MODES_DONE=$((MODES_DONE + 1))
        log_success "Step ${step}/${total} — ${label} passed"
    else
        FAILED_MODES+=("$label")
        log_error "Step ${step}/${total} — ${label} FAILED (continuing)"
    fi
    echo ""
}

# -----------------------------------------------
# Step 1: 0-shot (single-turn)
# -----------------------------------------------
run_step 1 $MODES_TOTAL "0-shot (single-turn)" \
    "\"${RUNNER_DIR}/run_single_turn.sh\" --config \"$CONF\""

# -----------------------------------------------
# Step 2: multi-turn CoT
# -----------------------------------------------
run_step 2 $MODES_TOTAL "multi-turn_CoT" \
    "\"${RUNNER_DIR}/run_multi_turn.sh\" --config \"$CONF\" --mode multi-turn_CoT"

# -----------------------------------------------
# Step 3: multi-turn CoT-SC
# -----------------------------------------------
run_step 3 $MODES_TOTAL "multi-turn_CoT-SC" \
    "\"${RUNNER_DIR}/run_multi_turn.sh\" --config \"$CONF\" --mode multi-turn_CoT-SC"

# -----------------------------------------------
# Step 4: self-refine CoT
# -----------------------------------------------
run_step 4 $MODES_TOTAL "self-refine_CoT" \
    "\"${RUNNER_DIR}/run_multi_turn.sh\" --config \"$CONF\" --mode self-refine_CoT"

# -----------------------------------------------
# Step 5: cross-consistency
# -----------------------------------------------
run_step 5 $MODES_TOTAL "cross-consistency" \
    "\"${RUNNER_DIR}/run_multi_turn.sh\" --config \"$CONF\" --mode cross-consistency"

# -----------------------------------------------
# Final summary
# -----------------------------------------------
print_separator
log_info "Smoke Test Summary"
print_separator
echo "Passed: ${MODES_DONE}/${MODES_TOTAL}"
echo "Failed: ${#FAILED_MODES[@]}/${MODES_TOTAL}"

if [ ${#FAILED_MODES[@]} -gt 0 ]; then
    echo ""
    echo "Failed modes:"
    for m in "${FAILED_MODES[@]}"; do
        echo "  ✗ $m"
    done
    print_separator
    exit 1
fi

print_separator
log_success "All modes passed!"
print_separator
