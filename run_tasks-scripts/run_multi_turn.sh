#!/bin/bash
# ================================================
# Multi-Turn Evaluation Runner (CoT, SC-CoT, MBR, Cross-Consistency)
# ================================================
# Usage examples:
#   ./run_multi_turn.sh --mode multi-turn_CoT --model qwen3-4b --task-pairs MEDQA
#   ./run_multi_turn.sh --mode multi-turn_CoT-SC --model-group 8B --task-pairs ALL
#   ./run_multi_turn.sh --mode cross-consistency --model-group 8B --task-pairs ALL

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
MODE="multi-turn_CoT"
REASONING_MODELS=()
ANSWERING_MODELS=()
TASK_PAIRS=()
OUTPUT_BASE="./outputs"
CUDA_DEVICES="0"
BATCH_SIZE="auto"
SEED="0"
LIMIT=""
DRY_RUN=false
USE_TIMESTAMP=false
SAME_MODEL=true  # Use same model for reasoning and answering by default
SKIP_EXISTING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            IFS=',' read -ra MODEL_LIST <<< "$2"
            for m in "${MODEL_LIST[@]}"; do
                model_args="$(get_model_args "$m")"
                REASONING_MODELS+=("$model_args")
                if [ "$SAME_MODEL" = true ]; then
                    ANSWERING_MODELS+=("$model_args")
                fi
            done
            shift 2
            ;;
        --model-group)
            case "$2" in
                TINY) 
                    REASONING_MODELS=("${MODELS_TINY[@]}")
                    ANSWERING_MODELS=("${MODELS_TINY[@]}")
                    ;;
                4B) 
                    REASONING_MODELS=("${MODELS_4B[@]}")
                    ANSWERING_MODELS=("${MODELS_4B[@]}")
                    ;;
                8B) 
                    REASONING_MODELS=("${MODELS_8B[@]}")
                    ANSWERING_MODELS=("${MODELS_8B[@]}")
                    ;;
                MEDICAL) 
                    REASONING_MODELS=("${MODELS_MEDICAL[@]}")
                    ANSWERING_MODELS=("${MODELS_MEDICAL[@]}")
                    ;;
                ALL) 
                    REASONING_MODELS=("${MODELS_ALL[@]}")
                    ANSWERING_MODELS=("${MODELS_ALL[@]}")
                    ;;
                *) log_error "Unknown model group: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --reasoning-model)
            SAME_MODEL=false
            IFS=',' read -ra MODEL_LIST <<< "$2"
            for m in "${MODEL_LIST[@]}"; do
                REASONING_MODELS+=("$(get_model_args "$m")")
            done
            shift 2
            ;;
        --answering-model)
            SAME_MODEL=false
            IFS=',' read -ra MODEL_LIST <<< "$2"
            for m in "${MODEL_LIST[@]}"; do
                ANSWERING_MODELS+=("$(get_model_args "$m")")
            done
            shift 2
            ;;
        --task-pairs)
            case "$2" in
                QA) TASK_PAIRS=("${QA_TASK_PAIRS[@]}") ;;
                NLI) TASK_PAIRS=("${NLI_TASK_PAIRS[@]}") ;;
                IE) TASK_PAIRS=("${IE_TASK_PAIRS[@]}") ;;
                ES) TASK_PAIRS=("${ES_TASK_PAIRS[@]}") ;;
                RANKING) TASK_PAIRS=("${RANKING_TASK_PAIRS[@]}") ;;
                TRIALBENCH) TASK_PAIRS=("${TRIALBENCH_TASK_PAIRS[@]}") ;;
                TRIALPANORAMA) TASK_PAIRS=("${TRIALPANORAMA_TASK_PAIRS[@]}") ;;
                TREC) TASK_PAIRS=("${TREC_TASK_PAIRS[@]}") ;;
                SMALL) TASK_PAIRS=("${SMALL_TASK_PAIRS[@]}") ;;
                ALL) TASK_PAIRS=("${ALL_TASK_PAIRS[@]}") ;;
                QA_SC) TASK_PAIRS=("${QA_SC_TASK_PAIRS[@]}") ;;
                NLI_SC) TASK_PAIRS=("${NLI_SC_TASK_PAIRS[@]}") ;;
                IE_SC) TASK_PAIRS=("${IE_SC_TASK_PAIRS[@]}") ;;
                ES_SC) TASK_PAIRS=("${ES_SC_TASK_PAIRS[@]}") ;;
                RANKING_SC) TASK_PAIRS=("${RANKING_SC_TASK_PAIRS[@]}") ;;
                TRIALBENCH_SC) TASK_PAIRS=("${TRIALBENCH_SC_TASK_PAIRS[@]}") ;;
                TRIALPANORAMA_SC) TASK_PAIRS=("${TRIALPANORAMA_SC_TASK_PAIRS[@]}") ;;
                TREC_SC) TASK_PAIRS=("${TREC_SC_TASK_PAIRS[@]}") ;;
                SMALL_SC) TASK_PAIRS=("${SMALL_SC_TASK_PAIRS[@]}") ;;
                ALL_SC) TASK_PAIRS=("${ALL_SC_TASK_PAIRS[@]}") ;;
                *) 
                    # Custom pair format: "task1:CoT|task1:0-shot"
                    TASK_PAIRS+=("$2")
                    ;;
            esac
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
        --dry-run)
            DRY_RUN=true
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
Multi-Turn Evaluation Runner

Usage: $0 [OPTIONS]

Mode Selection:
  --mode MODE                      Evaluation mode (multi-turn_CoT, multi-turn_CoT-SC, multi-turn_CoT-MBR, self-refine_CoT, cross-consistency)
                                   Note: cross-consistency passes all models in a single call per task pair

Model Selection:
  --model MODEL[,MODEL2,...]       Use same model(s) for reasoning and answering
  --model-group GROUP              Use predefined model group (TINY, 4B, 8B, MEDICAL, ALL)
  --reasoning-model MODEL          Separate reasoning model
  --answering-model MODEL          Separate answering model

Task Selection:
  --task-pairs GROUP               Task pair group (QA, NLI, IE, ES, RANKING, TRIALBENCH, TRIALPANORAMA, TREC, SMALL, ALL)
                                   SC variants (use CoT_SC reasoning tasks): QA_SC, NLI_SC, IE_SC, ES_SC,
                                   RANKING_SC, TRIALBENCH_SC, TRIALPANORAMA_SC, TREC_SC, SMALL_SC, ALL_SC
                                   Or custom pair: "task:CoT|task:0-shot"

Evaluation Options:
  --output PATH                    Base output directory (default: ./outputs)
  --gpu ID                         CUDA device ID (default: 0)
  --batch-size SIZE                Batch size (default: auto)
  --seed SEED                      Random seed (default: 0)
  --limit NUM                      Limit number of samples per task

Other Options:
  --timestamp                      Add timestamp to output paths
  --dry-run                        Show what would run without executing
  --config FILE                    Load configuration from file
  --help, -h                       Show this help message

Examples:
  # Run CoT with single model on QA tasks
  $0 --mode multi-turn_CoT --model qwen3-4b --task-pairs QA

  # Run SC-CoT with all 8B models on all tasks
  $0 --mode multi-turn_CoT-SC --model-group 8B --task-pairs ALL

  # Custom task pair with different models
  $0 --mode multi-turn_CoT --reasoning-model qwen3-4b --answering-model llama-8b \\
     --task-pairs "MedQA:CoT|MedQA:0-shot"

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
if [ ${#REASONING_MODELS[@]} -eq 0 ]; then
    log_error "No reasoning models specified"
    exit 1
fi

if [ ${#ANSWERING_MODELS[@]} -eq 0 ]; then
    log_error "No answering models specified"
    exit 1
fi

if [ ${#TASK_PAIRS[@]} -eq 0 ]; then
    log_error "No task pairs specified"
    exit 1
fi

# Check GPU availability
check_gpu "$CUDA_DEVICES"

# Calculate total runs
if [ "$MODE" = "cross-consistency" ]; then
    TOTAL_RUNS=${#TASK_PAIRS[@]}
else
    # One Python call per model — all task pairs are batched inside
    TOTAL_RUNS=${#REASONING_MODELS[@]}
    log_info "Task-pair batching enabled: ${#TASK_PAIRS[@]} task pairs will be evaluated per model in a single model load."
fi
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SUCCESSFUL_RUN_LIST=()
FAILED_RUN_LIST=()

# Show configuration
print_separator
log_info "Multi-Turn Evaluation Configuration"
print_separator
echo "Provider:           $PROVIDER"
echo "Mode:               $MODE"
echo "Reasoning Models:   ${#REASONING_MODELS[@]}"
echo "Reasoning Model Details:"
for model in "${REASONING_MODELS[@]}"; do
    echo "  - $model"
done
echo "Answering Models:   ${#ANSWERING_MODELS[@]}"
echo "Answering Model Details:"
for model in "${ANSWERING_MODELS[@]}"; do
    echo "  - $model"
done
echo "Task Pairs:         ${#TASK_PAIRS[@]}"
echo "Task Details:"
for pair in "${TASK_PAIRS[@]}"; do
    echo "  - $pair"
done
echo "Output Base:        $OUTPUT_BASE"
echo "GPU:                $CUDA_DEVICES"
echo "Batch Size:         $BATCH_SIZE"
echo "Seed:               $SEED"
if [ -n "$LIMIT" ]; then
    echo "Limit:              $LIMIT"
else
    echo "Limit:              None"
fi
echo "Total Runs:         $TOTAL_RUNS  (batched: ${#TASK_PAIRS[@]} task pairs per run)"
echo "Dry Run:            $DRY_RUN"
print_separator

# Exit if dry-run
if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run mode: No evaluations will be executed"
    exit 0
fi

# Record start time
START_TIME=$(date +%s)

# Main execution loop
if [ "$MODE" = "cross-consistency" ]; then
    # Cross-consistency: one call per task pair with ALL models passed together.
    # Join with a space — each element is then a separate argparse token.
    # (Comma-joining is wrong: model_args strings already contain commas internally.)
    REASONING_MODELS_STR=$(printf '%s ' "${REASONING_MODELS[@]}")
    ANSWERING_MODELS_STR=$(printf '%s ' "${ANSWERING_MODELS[@]}")
    REASONING_MODELS_STR="${REASONING_MODELS_STR% }"
    ANSWERING_MODELS_STR="${ANSWERING_MODELS_STR% }"

    for TASK_PAIR in "${TASK_PAIRS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Split task pair
        REASONING_TASK="${TASK_PAIR%%|*}"
        ANSWERING_TASK="${TASK_PAIR##*|}"

        show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "${REASONING_TASK} -> ${ANSWERING_TASK}"

        # __main__.py appends {task}/{model} automatically — just pass the mode root
        OUTPUT_PATH="${OUTPUT_BASE}/${MODE}"
        mkdir -p "$OUTPUT_PATH"

        if run_multi_turn_evaluation "$PROVIDER" "$MODE" \
                                    "$REASONING_MODELS_STR" "$ANSWERING_MODELS_STR" \
                                    "$REASONING_TASK" "$ANSWERING_TASK" \
                                    "$OUTPUT_PATH" "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES" "$LIMIT"; then
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            SUCCESSFUL_RUN_LIST+=("${REASONING_TASK} -> ${ANSWERING_TASK}")
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            FAILED_RUN_LIST+=("${REASONING_TASK} -> ${ANSWERING_TASK}")
            log_warning "Continuing with next evaluation..."
        fi

        echo ""
    done
else
    # CoT / CoT-SC / self-refine: one Python call per model, ALL task pairs batched
    for i in "${!REASONING_MODELS[@]}"; do
        REASONING_MODEL="${REASONING_MODELS[$i]}"
        ANSWERING_MODEL="${ANSWERING_MODELS[$i]}"

        REASONING_MODEL_NAME=$(get_model_name "$REASONING_MODEL")

        CURRENT_RUN=$((CURRENT_RUN + 1))
        show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "${#TASK_PAIRS[@]} task pairs with ${REASONING_MODEL_NAME}"

        # Build space-separated task lists — argparse nargs='+' expects separate tokens
        REASONING_TASKS_STR=""
        ANSWERING_TASKS_STR=""
        for TASK_PAIR in "${TASK_PAIRS[@]}"; do
            REASONING_TASK="${TASK_PAIR%%|*}"
            ANSWERING_TASK="${TASK_PAIR##*|}"
            REASONING_TASKS_STR="${REASONING_TASKS_STR}${REASONING_TASK} "
            ANSWERING_TASKS_STR="${ANSWERING_TASKS_STR}${ANSWERING_TASK} "
        done
        REASONING_TASKS_STR="${REASONING_TASKS_STR% }"
        ANSWERING_TASKS_STR="${ANSWERING_TASKS_STR% }"

        # Output base: pass the mode-level dir; __main__.py appends {task}/{model}
        OUTPUT_PATH="${OUTPUT_BASE}/${MODE}"
        mkdir -p "$OUTPUT_PATH"

        if [ "$SKIP_EXISTING" = true ] && has_existing_results "${OUTPUT_PATH}/${REASONING_MODEL_NAME}"; then
            log_info "Skipping (results exist): ${MODE} / ${REASONING_MODEL_NAME}"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            SUCCESSFUL_RUN_LIST+=("[SKIPPED] ${#TASK_PAIRS[@]} task pairs with ${REASONING_MODEL_NAME}")
            echo ""
            continue
        fi

        if run_multi_turn_evaluation "$PROVIDER" "$MODE" \
                                    "$REASONING_MODEL" "$ANSWERING_MODEL" \
                                    "$REASONING_TASKS_STR" "$ANSWERING_TASKS_STR" \
                                    "$OUTPUT_PATH" "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES" "$LIMIT"; then
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            SUCCESSFUL_RUN_LIST+=("${#TASK_PAIRS[@]} task pairs with ${REASONING_MODEL_NAME}")
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            FAILED_RUN_LIST+=("${#TASK_PAIRS[@]} task pairs with ${REASONING_MODEL_NAME}")
            log_warning "Continuing with next evaluation..."
        fi

        echo ""
    done
fi

# Record end time
END_TIME=$(date +%s)

# Show summary
show_summary "$TOTAL_RUNS" "$SUCCESSFUL_RUNS" "$FAILED_RUNS" "$START_TIME" "$END_TIME" SUCCESSFUL_RUN_LIST FAILED_RUN_LIST

# Exit with error if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi
