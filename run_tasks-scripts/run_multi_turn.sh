#!/bin/bash
# ================================================
# Multi-Turn Evaluation Runner (CoT, SC-CoT, MBR)
# ================================================
# Usage examples:
#   ./run_multi_turn.sh --mode multi-turn_CoT --model qwen3-4b --task-pairs MEDQA
#   ./run_multi_turn.sh --mode multi-turn_CoT-SC --model-group 8B --task-pairs ALL

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
DRY_RUN=false
USE_TIMESTAMP=false
SAME_MODEL=true  # Use same model for reasoning and answering by default

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
        --timestamp)
            USE_TIMESTAMP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << EOF
Multi-Turn Evaluation Runner

Usage: $0 [OPTIONS]

Mode Selection:
  --mode MODE                      Evaluation mode (multi-turn_CoT, multi-turn_CoT-SC, multi-turn_CoT-MBR)

Model Selection:
  --model MODEL[,MODEL2,...]       Use same model(s) for reasoning and answering
  --model-group GROUP              Use predefined model group (TINY, 4B, 8B, MEDICAL, ALL)
  --reasoning-model MODEL          Separate reasoning model
  --answering-model MODEL          Separate answering model

Task Selection:
  --task-pairs GROUP               Task pair group (QA, NLI, IE, ES, RANKING, TRIALBENCH, TRIALPANORAMA, TREC, SMALL, ALL)
                                   Or custom pair: "task:CoT|task:0-shot"

Evaluation Options:
  --output PATH                    Base output directory (default: ./outputs)
  --gpu ID                         CUDA device ID (default: 0)
  --batch-size SIZE                Batch size (default: auto)
  --seed SEED                      Random seed (default: 0)

Other Options:
  --timestamp                      Add timestamp to output paths
  --dry-run                        Show what would run without executing
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
TOTAL_RUNS=$((${#REASONING_MODELS[@]} * ${#TASK_PAIRS[@]}))
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Show configuration
print_separator
log_info "Multi-Turn Evaluation Configuration"
print_separator
echo "Provider:        $PROVIDER"
echo "Mode:            $MODE"
echo "Reasoning Models: ${#REASONING_MODELS[@]}"
echo "Answering Models: ${#ANSWERING_MODELS[@]}"
echo "Task Pairs:      ${#TASK_PAIRS[@]}"
echo "Output Base:     $OUTPUT_BASE"
echo "GPU:             $CUDA_DEVICES"
echo "Total Runs:      $TOTAL_RUNS"
echo "Dry Run:         $DRY_RUN"
print_separator

# Exit if dry-run
if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run mode: No evaluations will be executed"
    exit 0
fi

# Record start time
START_TIME=$(date +%s)

# Main execution loop
for i in "${!REASONING_MODELS[@]}"; do
    REASONING_MODEL="${REASONING_MODELS[$i]}"
    ANSWERING_MODEL="${ANSWERING_MODELS[$i]}"
    
    REASONING_MODEL_NAME=$(get_model_name "$REASONING_MODEL")
    ANSWERING_MODEL_NAME=$(get_model_name "$ANSWERING_MODEL")
    
    for TASK_PAIR in "${TASK_PAIRS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        # Split task pair
        REASONING_TASK="${TASK_PAIR%%|*}"
        ANSWERING_TASK="${TASK_PAIR##*|}"
        
        show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "${REASONING_TASK} -> ${ANSWERING_TASK}"
        
        # Build output path
        TASK_NAME=$(echo ${REASONING_TASK} | tr ':' '_')
        OUTPUT_PATH="${OUTPUT_BASE}/${MODE}/${TASK_NAME}/${REASONING_MODEL_NAME}"
        
        if [ "$USE_TIMESTAMP" = true ]; then
            OUTPUT_PATH=$(create_output_dir "$OUTPUT_PATH" true)
        else
            mkdir -p "$OUTPUT_PATH"
        fi
        
        if run_multi_turn_evaluation "$PROVIDER" "$MODE" \
                                    "$REASONING_MODEL" "$ANSWERING_MODEL" \
                                    "$REASONING_TASK" "$ANSWERING_TASK" \
                                    "$OUTPUT_PATH" "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES"; then
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            log_warning "Continuing with next evaluation..."
        fi
        
        echo ""
    done
done

# Record end time
END_TIME=$(date +%s)

# Show summary
show_summary "$TOTAL_RUNS" "$SUCCESSFUL_RUNS" "$FAILED_RUNS" "$START_TIME" "$END_TIME"

# Exit with error if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi
