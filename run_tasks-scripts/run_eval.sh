#!/bin/bash
# ================================================
# Unified Evaluation Runner
# ================================================
# Usage examples:
#   ./run_eval.sh --model qwen3-4b --task MedQA --mode 0-shot
#   ./run_eval.sh --model llama-8b --task-group MEDQA_TASKS --modes "0-shot CoT"
#   ./run_eval.sh --config my_eval.conf --dry-run

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
SEED="0"
LIMIT=""
DRY_RUN=false
USE_TIMESTAMP=false
VERBOSE=false

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
Unified Evaluation Runner

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
  --seed SEED                      Random seed (default: 0)
  --limit NUM                      Limit number of samples per task

Other Options:
  --timestamp                      Add timestamp to output paths
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
  $0 --config my_evaluation.conf

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

# Check GPU availability
check_gpu "$CUDA_DEVICES"

# Calculate total runs
TOTAL_RUNS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#MODES[@]}))
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
echo "Seed:            $SEED"
echo "Total Runs:      $TOTAL_RUNS"
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

# Main execution loop
for MODEL_ARGS in "${MODELS[@]}"; do
    MODEL_NAME=$(get_model_name "$MODEL_ARGS")
    
    for TASK in "${TASKS[@]}"; do
        for MODE in "${MODES[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            
            show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "${TASK} (${MODE}) with ${MODEL_NAME}"
            
            OUTPUT_PATH=$(build_output_path "$OUTPUT_BASE" "$TASK" "$MODE" "$MODEL_NAME")
            mkdir -p "$OUTPUT_PATH"
            
            if run_single_evaluation "$PROVIDER" "$MODEL_ARGS" "$TASK" "$MODE" "$OUTPUT_PATH" \
                                    "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES" "$LIMIT"; then
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                SUCCESSFUL_RUN_LIST+=("${TASK} (${MODE}) with ${MODEL_NAME}")
            else
                FAILED_RUNS=$((FAILED_RUNS + 1))
                FAILED_RUN_LIST+=("${TASK} (${MODE}) with ${MODEL_NAME}")
                log_warning "Continuing with next evaluation..."
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
