#!/bin/bash
# ================================================
# Single-Turn Evaluation Runner
# ================================================
# Usage examples:
#   ./run_single_turn.sh --model qwen3-4b --task MedQA --mode 0-shot
#   ./run_single_turn.sh --model llama-8b --task-group MEDQA_TASKS --modes "0-shot CoT"
#   ./run_single_turn.sh --config my_eval.conf --dry-run

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Source configurations
source "${SCRIPT_DIR}/config/models.conf"
source "${SCRIPT_DIR}/config/tasks.conf"
source "${SCRIPT_DIR}/config/tasks_pairs.conf"

# Default values
PROVIDER="vllm"
MODELS=()
TASKS=()
MODES=("0-shot")
OUTPUT_BASE="./outputs/unified_runner"
CUDA_DEVICES="0"
BATCH_SIZE="auto"
TASKS_PER_RUN=""
SEED="0"
LIMIT=""
DRY_RUN=false
USE_TIMESTAMP=false
VERBOSE=false
SKIP_EXISTING=false
OOM_BACKOFF=true
PROFILE="BALANCED_MEM"
MAX_LENGTH_OVERRIDE=""
GPU_MEM_UTIL_OVERRIDE=""
SWAP_SPACE_OVERRIDE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            IFS=',' read -ra MODEL_LIST <<< "$2"
            for m in "${MODEL_LIST[@]}"; do
                MODELS+=("$(get_model_args "$m")")
            done
            shift 2
            ;;
        --model-group)
            case "$2" in
                TINY) MODELS=("${MODELS_TINY[@]}") ;;
                4B) MODELS=("${MODELS_4B[@]}") ;;
                8B) MODELS=("${MODELS_8B[@]}") ;;
                MEDICAL) MODELS=("${MODELS_MEDICAL[@]}") ;;
                ALL) MODELS=("${MODELS_ALL[@]}") ;;
                *) log_error "Unknown model group: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --task)
            IFS=',' read -ra TASKS <<< "$2"
            shift 2
            ;;
        --task-group)
            case "$2" in
                QA) TASKS=("${QA_TASKS[@]}") ;;
                NLI) TASKS=("${NLI_TASKS[@]}") ;;
                IE) TASKS=("${IE_TASKS[@]}") ;;
                ES) TASKS=("${ES_TASKS[@]}") ;;
                RANKING) TASKS=("${RANKING_TASKS[@]}") ;;
                TRIALBENCH) TASKS=("${TRIALBENCH_TASKS[@]}") ;;
                TRIALPANORAMA) TASKS=("${TRIALPANORAMA_TASKS[@]}") ;;
                TREC) TASKS=("${TREC_TASKS[@]}") ;;
                SMALL) TASKS=("${SMALL_TASKS[@]}") ;;
                ALL) TASKS=("${ALL_TASKS[@]}") ;;
                *) log_error "Unknown task group: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --mode|--modes)
            IFS=',' read -ra MODES <<< "$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --gpu|--cuda)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH_OVERRIDE="$2"
            shift 2
            ;;
        --gpu-mem-util)
            GPU_MEM_UTIL_OVERRIDE="$2"
            shift 2
            ;;
        --swap-space)
            SWAP_SPACE_OVERRIDE="$2"
            shift 2
            ;;
        --tasks-per-run)
            TASKS_PER_RUN="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --timestamp)
            USE_TIMESTAMP=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --oom-backoff)
            OOM_BACKOFF=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --config)
            if [ -f "$2" ]; then
                source "$2"
            else
                log_error "Config file not found: $2"
                exit 1
            fi
            shift 2
            ;;
        --help|-h)
            cat << EOF
Single-Turn Evaluation Runner

Usage: $0 [OPTIONS]

Model Selection:
  --model MODEL[,MODEL2,...]       Specify model(s) by preset name or full args
  --model-group GROUP              Use predefined model group (TINY, 4B, 8B, MEDICAL, ALL)

Task Selection:
  --task TASK[,TASK2,...]          Specify task(s) by name
  --task-group GROUP               Use predefined task group (QA, NLI, IE, ES, RANKING, TRIALBENCH, TRIALPANORAMA, TREC, SMALL, ALL)

Evaluation Options:
  --mode MODE[,MODE2,...]          Inference mode(s) (0-shot, CoT, SC)
  --output PATH                    Base output directory (default: ./outputs/unified_runner)
  --gpu ID                         CUDA device ID (default: 0)
  --batch-size SIZE                Batch size (default: auto)
    --profile NAME                   Runtime profile: LOW_MEM, BALANCED_MEM, HIGH_MEM (default: BALANCED_MEM)
    --max-length N                   Override model max_length in model_args
    --gpu-mem-util F                 Override model gpu_memory_utilization (0-1)
    --swap-space GB                  Override model swap_space (GB)
    --tasks-per-run N                Number of tasks per lm_eval call (default: all tasks)
  --seed SEED                      Random seed (default: 0)
  --limit NUM                      Limit number of samples per task

Other Options:
  --timestamp                      Add timestamp to output paths
    --oom-backoff                    On OOM, retry failed task chunks as single-task runs
  --dry-run                        Show what would run without executing
  --verbose                        Show detailed information
  --config FILE                    Load configuration from file
  --help, -h                       Show this help message

Model Presets:
  qwen3-0.5b, gemma-270m, qwen3-4b, gemma-4b, qwen3-8b, llama-8b, 
  deepseek-8b, ministral-8b, fleming-7b, panacea-7b

Task Groups:
  QA, NLI, IE, ES, RANKING,
  TRIALBENCH, TRIALPANORAMA, TREC, SMALL, ALL

Examples:
  # Run single model on single task
  $0 --model qwen3-4b --task MedQA --mode 0-shot

  # Run multiple models on task group
  $0 --model qwen3-4b,llama-8b --task-group QA --modes "0-shot,CoT"

  # Use model group with dry-run
  $0 --model-group 8B --task-group TRIALBENCH --modes "0-shot,CoT,SC" --dry-run

  # Load from config file
  $0 --config examples/default/quick_run.conf

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

# Validate inputs
if [ ${#MODELS[@]} -eq 0 ]; then
    log_error "No models specified. Use --model or --model-group"
    exit 1
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    log_error "No tasks specified. Use --task or --task-group"
    exit 1
fi

if [ -n "$TASKS_PER_RUN" ]; then
    if ! [[ "$TASKS_PER_RUN" =~ ^[0-9]+$ ]] || [ "$TASKS_PER_RUN" -le 0 ]; then
        log_error "--tasks-per-run must be a positive integer"
        exit 1
    fi
fi

case "$PROFILE" in
    LOW_MEM|BALANCED_MEM|HIGH_MEM) ;;
    *)
        log_error "Unknown profile: $PROFILE (expected LOW_MEM, BALANCED_MEM, or HIGH_MEM)"
        exit 1
        ;;
esac

# Check GPU availability
check_gpu "$CUDA_DEVICES"

# Apply runtime model arg overrides (useful for OOM mitigation / speed tuning)
for i in "${!MODELS[@]}"; do
    MODELS[$i]="$(apply_model_profile "${MODELS[$i]}" "$PROFILE")"
    MODELS[$i]="$(apply_model_arg_overrides "${MODELS[$i]}" "$MAX_LENGTH_OVERRIDE" "$GPU_MEM_UTIL_OVERRIDE" "$SWAP_SPACE_OVERRIDE" "")"
done

# One or more lm_eval calls per (model, mode), depending on TASKS_PER_RUN
TASK_CHUNK_SIZE=${TASKS_PER_RUN:-${#TASKS[@]}}
TASK_CHUNKS=$(( (${#TASKS[@]} + TASK_CHUNK_SIZE - 1) / TASK_CHUNK_SIZE ))
TOTAL_RUNS=$((${#MODELS[@]} * ${#MODES[@]} * TASK_CHUNKS))
if [ "$TASK_CHUNK_SIZE" -lt "${#TASKS[@]}" ]; then
    log_info "Task chunking enabled: ${#TASKS[@]} tasks will run in ${TASK_CHUNKS} chunk(s) of up to ${TASK_CHUNK_SIZE} task(s) per model+mode."
else
    log_info "Task batching enabled: ${#TASKS[@]} tasks will be evaluated per model+mode in a single model load."
fi
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SUCCESSFUL_RUN_LIST=()
FAILED_RUN_LIST=()

# Show configuration
print_separator
log_info "Evaluation Configuration"
print_separator
echo "Provider:        $PROVIDER"
echo "Models:          ${#MODELS[@]}"
echo "Model Details:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo "Tasks:           ${#TASKS[@]}"
echo "Task Details:"
for task in "${TASKS[@]}"; do
    echo "  - $task"
done
echo "Modes:           ${MODES[*]}"
echo "Limit:           ${LIMIT:-None}"
echo "Output Base:     $OUTPUT_BASE"
echo "GPU:             $CUDA_DEVICES"
echo "Batch Size:      $BATCH_SIZE"
echo "Profile:         $PROFILE"
echo "Max Length Ovrd: ${MAX_LENGTH_OVERRIDE:-None}"
echo "GPU Mem Util Ovrd: ${GPU_MEM_UTIL_OVERRIDE:-None}"
echo "Swap Space Ovrd: ${SWAP_SPACE_OVERRIDE:-None}"
echo "Tasks/Run:       ${TASKS_PER_RUN:-all}"
echo "OOM Backoff:     $OOM_BACKOFF"
echo "Seed:            $SEED"
echo "Total Runs:      $TOTAL_RUNS  (${TASK_CHUNKS} chunk(s) per model+mode, up to ${TASK_CHUNK_SIZE} tasks each)"
echo "Dry Run:         $DRY_RUN"
print_separator

if [ "$VERBOSE" = true ]; then
    echo ""
    log_info "Models:"
    for model in "${MODELS[@]}"; do
        echo "  - $(get_model_name "$model")"
    done
    echo ""
    log_info "Tasks:"
    for task in "${TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
fi

# Exit if dry-run
if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run mode: No evaluations will be executed"
    exit 0
fi

# Record start time
START_TIME=$(date +%s)

# Main execution loop — each (model, mode) may run in multiple task chunks
for MODEL_ARGS in "${MODELS[@]}"; do
    MODEL_NAME=$(get_model_name "$MODEL_ARGS")

    for MODE in "${MODES[@]}"; do
        for ((TASK_OFFSET=0; TASK_OFFSET<${#TASKS[@]}; TASK_OFFSET+=TASK_CHUNK_SIZE)); do
            TASK_CHUNK=("${TASKS[@]:TASK_OFFSET:TASK_CHUNK_SIZE}")
            CURRENT_RUN=$((CURRENT_RUN + 1))

            show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "${#TASK_CHUNK[@]} tasks (${MODE}) with ${MODEL_NAME}"

            # Build comma-separated task list for this mode and chunk
            TASK_LIST=""
            for TASK in "${TASK_CHUNK[@]}"; do
                TASK_LIST="${TASK_LIST:+${TASK_LIST},}${TASK}_${MODE}"
            done

            # Output base for this (mode, model); lm_eval creates per-task subdirs inside
            OUTPUT_PATH="${OUTPUT_BASE}/single-turn/${MODE}/${MODEL_NAME}"
            mkdir -p "$OUTPUT_PATH"

            if [ "$SKIP_EXISTING" = true ] && [ "$TASK_CHUNKS" -eq 1 ] && has_existing_results "$OUTPUT_PATH"; then
                log_info "Skipping (results exist): ${MODE} / ${MODEL_NAME}"
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                SUCCESSFUL_RUN_LIST+=("[SKIPPED] ${#TASK_CHUNK[@]} tasks [${MODEL_NAME}] (${MODE}): ${TASK_LIST}")
                echo ""
                continue
            fi

            if run_batch_evaluation "$PROVIDER" "$MODEL_ARGS" "$TASK_LIST" "$OUTPUT_PATH" \
                                    "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES" "$LIMIT"; then
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                SUCCESSFUL_RUN_LIST+=("${#TASK_CHUNK[@]} tasks [${MODEL_NAME}] (${MODE}): ${TASK_LIST}")
            else
                status=$?
                chunk_ok=false

                if [ "$OOM_BACKOFF" = true ] && [ $status -eq 99 ]; then
                    log_warning "OOM detected for chunk (${#TASK_CHUNK[@]} tasks). Retrying per-task."
                    chunk_ok=true
                    for TASK in "${TASK_CHUNK[@]}"; do
                        SINGLE_TASK_LIST="${TASK}_${MODE}"
                        retry_bs="$BATCH_SIZE"
                        if ! run_batch_evaluation "$PROVIDER" "$MODEL_ARGS" "$SINGLE_TASK_LIST" "$OUTPUT_PATH" \
                                                 "$retry_bs" "$SEED" "$CUDA_DEVICES" "$LIMIT"; then
                            single_status=$?
                            if [ $single_status -eq 99 ] && [ "$retry_bs" != "1" ]; then
                                log_warning "OOM persists for ${TASK}_${MODE}; retrying with batch_size=1"
                                if ! run_batch_evaluation "$PROVIDER" "$MODEL_ARGS" "$SINGLE_TASK_LIST" "$OUTPUT_PATH" \
                                                         "1" "$SEED" "$CUDA_DEVICES" "$LIMIT"; then
                                    chunk_ok=false
                                fi
                            else
                                chunk_ok=false
                            fi
                        fi
                    done
                fi

                if [ "$chunk_ok" = true ]; then
                    SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                    SUCCESSFUL_RUN_LIST+=("[OOM-backoff] ${#TASK_CHUNK[@]} tasks [${MODEL_NAME}] (${MODE}): ${TASK_LIST}")
                else
                    FAILED_RUNS=$((FAILED_RUNS + 1))
                    FAILED_RUN_LIST+=("${#TASK_CHUNK[@]} tasks [${MODEL_NAME}] (${MODE}): ${TASK_LIST}")
                    log_warning "Continuing with next evaluation..."
                fi
            fi

            echo ""
        done
    done
done

# Record end time
END_TIME=$(date +%s)

# Show summary
show_summary "$TOTAL_RUNS" "$SUCCESSFUL_RUNS" "$FAILED_RUNS" "$START_TIME" "$END_TIME" SUCCESSFUL_RUN_LIST FAILED_RUN_LIST

# Exit with error if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi
