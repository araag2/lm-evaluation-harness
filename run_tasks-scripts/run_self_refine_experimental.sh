#!/bin/bash
# ================================================
# Self-Refine CoT Experimental Runner
# ================================================
# Runs mode_self_refine_CoT_experimental for all models on the SMALL task set.
# Checkpoints are captured at iterations 1, 5, and 10.
# stop_on_degradation is ON by default (pass --no-degradation to disable).
#
# Usage examples:
#   ./run_self_refine_experimental.sh
#   ./run_self_refine_experimental.sh --model-group 8B --output outputs/exp
#   ./run_self_refine_experimental.sh --model qwen3-4b --task-pairs QA --limit 50
#   ./run_self_refine_experimental.sh --no-degradation
#   ./run_self_refine_experimental.sh --feedback-max-tokens 500

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/lib/eval_utils.sh"
source "${SCRIPT_DIR}/config/models.conf"
source "${SCRIPT_DIR}/config/tasks.conf"
source "${SCRIPT_DIR}/config/tasks_pairs.conf"

# ── Defaults ────────────────────────────────────────────────────────────────
PROVIDER="vllm"
MODE="self-refine_CoT_experimental"
REASONING_MODELS=()
ANSWERING_MODELS=()
TASK_PAIRS=()
OUTPUT_BASE="./outputs"
CUDA_DEVICES="0"
BATCH_SIZE="auto"
SEED="0"
LIMIT=""
DRY_RUN=false
SKIP_EXISTING=false
NO_DEGRADATION=false           # --no-degradation disables stop_on_degradation
FEEDBACK_MAX_TOKENS="1000"
PROFILE="BALANCED_MEM"
MAX_LENGTH_OVERRIDE=""
GPU_MEM_UTIL_OVERRIDE=""
SWAP_SPACE_OVERRIDE=""
# ─────────────────────────────────────────────────────────────────────────────

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                IFS=',' read -ra MODEL_LIST <<< "$2"
                for m in "${MODEL_LIST[@]}"; do
                    model_args="$(get_model_args "$m")"
                    REASONING_MODELS+=("$model_args")
                    ANSWERING_MODELS+=("$model_args")
                done
                shift 2
                ;;
            --model-group)
                case "$2" in
                    TINY)    REASONING_MODELS=("${MODELS_TINY[@]}");    ANSWERING_MODELS=("${MODELS_TINY[@]}") ;;
                    4B)      REASONING_MODELS=("${MODELS_4B[@]}");      ANSWERING_MODELS=("${MODELS_4B[@]}") ;;
                    8B)      REASONING_MODELS=("${MODELS_8B[@]}");      ANSWERING_MODELS=("${MODELS_8B[@]}") ;;
                    MEDICAL) REASONING_MODELS=("${MODELS_MEDICAL[@]}"); ANSWERING_MODELS=("${MODELS_MEDICAL[@]}") ;;
                    ALL)     REASONING_MODELS=("${MODELS_ALL[@]}");     ANSWERING_MODELS=("${MODELS_ALL[@]}") ;;
                    *) log_error "Unknown model group: $2"; exit 1 ;;
                esac
                shift 2
                ;;
            --task-pairs)
                case "$2" in
                    QA)           TASK_PAIRS=("${QA_TASK_PAIRS[@]}") ;;
                    NLI)          TASK_PAIRS=("${NLI_TASK_PAIRS[@]}") ;;
                    IE)           TASK_PAIRS=("${IE_TASK_PAIRS[@]}") ;;
                    ES)           TASK_PAIRS=("${ES_TASK_PAIRS[@]}") ;;
                    RANKING)      TASK_PAIRS=("${RANKING_TASK_PAIRS[@]}") ;;
                    TRIALBENCH)   TASK_PAIRS=("${TRIALBENCH_TASK_PAIRS[@]}") ;;
                    TRIALPANORAMA) TASK_PAIRS=("${TRIALPANORAMA_TASK_PAIRS[@]}") ;;
                    TREC)         TASK_PAIRS=("${TREC_TASK_PAIRS[@]}") ;;
                    SMALL)        TASK_PAIRS=("${SMALL_TASK_PAIRS[@]}") ;;
                    ALL)          TASK_PAIRS=("${ALL_TASK_PAIRS[@]}") ;;
                    *) TASK_PAIRS+=("$2") ;;  # Custom pair
                esac
                shift 2
                ;;
            --output)            OUTPUT_BASE="$2";          shift 2 ;;
            --gpu|--cuda)        CUDA_DEVICES="$2";         shift 2 ;;
            --batch-size)        BATCH_SIZE="$2";           shift 2 ;;
            --profile)           PROFILE="$2";              shift 2 ;;
            --max-length)        MAX_LENGTH_OVERRIDE="$2";  shift 2 ;;
            --gpu-mem-util)      GPU_MEM_UTIL_OVERRIDE="$2"; shift 2 ;;
            --swap-space)        SWAP_SPACE_OVERRIDE="$2";  shift 2 ;;
            --seed)              SEED="$2";                 shift 2 ;;
            --limit)             LIMIT="$2";                shift 2 ;;
            --feedback-max-tokens) FEEDBACK_MAX_TOKENS="$2"; shift 2 ;;
            --no-degradation)    NO_DEGRADATION=true;       shift ;;
            --skip-existing)     SKIP_EXISTING=true;        shift ;;
            --dry-run)           DRY_RUN=true;              shift ;;
            --config)
                [ -f "$2" ] && source "$2" || { log_error "Config not found: $2"; exit 1; }
                shift 2
                ;;
            --help|-h)
                cat << EOF
Self-Refine CoT Experimental Runner

Runs mode=self-refine_CoT_experimental (checkpoints at iterations 1, 5, 10)
for all models on the SMALL task set by default.

Usage: $0 [OPTIONS]

Model Selection:
  --model MODEL[,MODEL2,...]      Comma-separated model preset names
  --model-group GROUP             TINY | 4B | 8B | MEDICAL | ALL  (default: ALL)

Task Selection:
  --task-pairs GROUP              QA | NLI | IE | ES | SMALL | ALL  (default: SMALL)
                                  Or custom: "task:CoT|task:0-shot"

Self-Refine Options:
  --no-degradation                Disable stop_on_degradation (on by default)
  --feedback-max-tokens N         Cap feedback generation tokens (default: 1000)

Evaluation Options:
  --output PATH                   Base output directory (default: ./outputs)
  --gpu ID                        CUDA device (default: 0)
  --batch-size SIZE               Batch size (default: auto)
    --profile NAME                  Runtime profile: LOW_MEM, BALANCED_MEM, HIGH_MEM (default: BALANCED_MEM)
    --max-length N                  Override model max_length in model_args
    --gpu-mem-util F                Override model gpu_memory_utilization (0-1)
    --swap-space GB                 Override model swap_space (GB)
  --seed SEED                     Random seed (default: 0)
  --limit NUM                     Limit samples per task (for quick tests)

Other:
  --skip-existing                 Skip models that already have results
  --dry-run                       Print configuration without running
  --config FILE                   Source a bash config file
  --help, -h                      Show this message

Examples:
  # Defaults: all models, SMALL tasks, degradation ON, 1000-token feedback cap
  $0

  # Only 8B models, custom output path
  $0 --model-group 8B --output outputs/self_refine_exp

  # Quick test: 50 samples, single model
  $0 --model qwen3-4b --limit 50

  # Disable early stopping
  $0 --no-degradation
EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Run one Python evaluation for the experimental mode
run_self_refine_exp() {
    local provider="$1"
    local reasoning_model="$2"
    local answering_model="$3"
    local reasoning_tasks="$4"    # space-separated
    local answering_tasks="$5"    # space-separated
    local output_path="$6"
    local batch_size="${7:-auto}"
    local seed="${8:-0}"
    local cuda_devices="${9:-0}"
    local limit="${10:-}"
    local feedback_max_tokens="${11:-1000}"
    local no_degradation="${12:-false}"

    log_info "Running Self-Refine Experimental"
    log_info "Output: ${output_path}"

    local deg_flag="--stop_on_degradation"
    [ "$no_degradation" = true ] && deg_flag="--no_stop_on_degradation"

    local cmd="VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -m lm_eval.reasoning_modes \
        --provider $provider \
        --mode $MODE \
        --reasoning_models \"$reasoning_model\" \
        --answering_models \"$answering_model\" \
        --reasoning_tasks $reasoning_tasks \
        --answering_tasks $answering_tasks \
        --output_path \"$output_path\" \
        --batch_size $batch_size \
        --seed $seed \
        --log_samples \
        --feedback_max_tokens $feedback_max_tokens \
        $deg_flag"

    if [ -n "$limit" ]; then
        cmd="$cmd --limit $limit"
    fi

    # Clear GPU memory
    log_info "Clearing GPU memory..."
    CUDA_VISIBLE_DEVICES="$cuda_devices" python -c \
        "import torch; torch.cuda.empty_cache(); import gc; gc.collect(); print('GPU memory cleared')" || true

    # Kill leftover vLLM processes
    local _pgid
    _pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    if [ -n "$_pgid" ]; then
        pkill -g "$_pgid" -f "EngineCore" 2>/dev/null || true
        pkill -g "$_pgid" -f "vllm"       2>/dev/null || true
    fi

    eval "$cmd"
    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Completed: $reasoning_tasks"
        return 0
    else
        log_error "Failed: $reasoning_tasks — exit code $status"
        return $status
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
parse_args "$@"

# Apply defaults if nothing was specified
[ ${#REASONING_MODELS[@]} -eq 0 ] && {
    REASONING_MODELS=("${MODELS_ALL[@]}")
    ANSWERING_MODELS=("${MODELS_ALL[@]}")
}
[ ${#TASK_PAIRS[@]} -eq 0 ] && TASK_PAIRS=("${SMALL_TASK_PAIRS[@]}")

# Validate
[ ${#REASONING_MODELS[@]} -eq 0 ] && { log_error "No models found"; exit 1; }
[ ${#TASK_PAIRS[@]}       -eq 0 ] && { log_error "No task pairs found"; exit 1; }

case "$PROFILE" in
    LOW_MEM|BALANCED_MEM|HIGH_MEM) ;;
    *)
        log_error "Unknown profile: $PROFILE (expected LOW_MEM, BALANCED_MEM, or HIGH_MEM)"
        exit 1
        ;;
esac

check_gpu "$CUDA_DEVICES"

# Apply runtime model arg overrides (useful for OOM mitigation / speed tuning)
for i in "${!REASONING_MODELS[@]}"; do
    REASONING_MODELS[$i]="$(apply_model_profile "${REASONING_MODELS[$i]}" "$PROFILE")"
    REASONING_MODELS[$i]="$(apply_model_arg_overrides "${REASONING_MODELS[$i]}" "$MAX_LENGTH_OVERRIDE" "$GPU_MEM_UTIL_OVERRIDE" "$SWAP_SPACE_OVERRIDE" "")"
done
for i in "${!ANSWERING_MODELS[@]}"; do
    ANSWERING_MODELS[$i]="$(apply_model_profile "${ANSWERING_MODELS[$i]}" "$PROFILE")"
    ANSWERING_MODELS[$i]="$(apply_model_arg_overrides "${ANSWERING_MODELS[$i]}" "$MAX_LENGTH_OVERRIDE" "$GPU_MEM_UTIL_OVERRIDE" "$SWAP_SPACE_OVERRIDE" "")"
done

TOTAL_RUNS=${#REASONING_MODELS[@]}
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SUCCESSFUL_RUN_LIST=()
FAILED_RUN_LIST=()

# ── Print configuration ───────────────────────────────────────────────────────
print_separator
log_info "Self-Refine CoT Experimental Configuration"
print_separator
echo "Provider:                $PROVIDER"
echo "Mode:                    $MODE"
echo "Checkpoints:             1, 5, 10 iterations"
echo "Stop on Degradation:     $([ "$NO_DEGRADATION" = true ] && echo "OFF" || echo "ON")"
echo "Feedback Max Tokens:     $FEEDBACK_MAX_TOKENS"
echo "Models:                  ${#REASONING_MODELS[@]}"
for m in "${REASONING_MODELS[@]}"; do echo "  - $m"; done
echo "Task Pairs:              ${#TASK_PAIRS[@]}"
for p in "${TASK_PAIRS[@]}"; do echo "  - $p"; done
echo "Output Base:             $OUTPUT_BASE"
echo "GPU:                     $CUDA_DEVICES"
echo "Batch Size:              $BATCH_SIZE"
echo "Profile:                 $PROFILE"
echo "Max Length Ovrd:         ${MAX_LENGTH_OVERRIDE:-None}"
echo "GPU Mem Util Ovrd:       ${GPU_MEM_UTIL_OVERRIDE:-None}"
echo "Swap Space Ovrd:         ${SWAP_SPACE_OVERRIDE:-None}"
echo "Seed:                    $SEED"
[ -n "$LIMIT" ] && echo "Limit:                   $LIMIT" || echo "Limit:                   None"
echo "Total Runs:              $TOTAL_RUNS  (${#TASK_PAIRS[@]} task pairs batched per model)"
echo "Dry Run:                 $DRY_RUN"
print_separator

[ "$DRY_RUN" = true ] && { log_info "Dry-run: nothing will execute"; exit 0; }

START_TIME=$(date +%s)

# ── Execution loop — one Python call per model, all task pairs batched ────────
for i in "${!REASONING_MODELS[@]}"; do
    REASONING_MODEL="${REASONING_MODELS[$i]}"
    ANSWERING_MODEL="${ANSWERING_MODELS[$i]}"
    REASONING_MODEL_NAME=$(get_model_name "$REASONING_MODEL")

    CURRENT_RUN=$((CURRENT_RUN + 1))
    show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "${#TASK_PAIRS[@]} task pairs with ${REASONING_MODEL_NAME}"

    # Build space-separated task lists
    REASONING_TASKS_STR=""
    ANSWERING_TASKS_STR=""
    for TASK_PAIR in "${TASK_PAIRS[@]}"; do
        REASONING_TASKS_STR="${REASONING_TASKS_STR}${TASK_PAIR%%|*} "
        ANSWERING_TASKS_STR="${ANSWERING_TASKS_STR}${TASK_PAIR##*|} "
    done
    REASONING_TASKS_STR="${REASONING_TASKS_STR% }"
    ANSWERING_TASKS_STR="${ANSWERING_TASKS_STR% }"

    OUTPUT_PATH="${OUTPUT_BASE}/${MODE}"
    mkdir -p "$OUTPUT_PATH"

    if [ "$SKIP_EXISTING" = true ] && has_existing_results "${OUTPUT_PATH}/${REASONING_MODEL_NAME}"; then
        log_info "Skipping (results exist): ${MODE} / ${REASONING_MODEL_NAME}"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        SUCCESSFUL_RUN_LIST+=("[SKIPPED] ${REASONING_MODEL_NAME}")
        echo ""
        continue
    fi

    if run_self_refine_exp "$PROVIDER" \
                           "$REASONING_MODEL" "$ANSWERING_MODEL" \
                           "$REASONING_TASKS_STR" "$ANSWERING_TASKS_STR" \
                           "$OUTPUT_PATH" "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES" \
                           "$LIMIT" "$FEEDBACK_MAX_TOKENS" "$NO_DEGRADATION"; then
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        SUCCESSFUL_RUN_LIST+=("${REASONING_MODEL_NAME}")
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_RUN_LIST+=("${REASONING_MODEL_NAME}")
        log_warning "Continuing with next model..."
    fi

    echo ""
done

END_TIME=$(date +%s)
show_summary "$TOTAL_RUNS" "$SUCCESSFUL_RUNS" "$FAILED_RUNS" "$START_TIME" "$END_TIME" \
             SUCCESSFUL_RUN_LIST FAILED_RUN_LIST

[ $FAILED_RUNS -gt 0 ] && exit 1
exit 0
