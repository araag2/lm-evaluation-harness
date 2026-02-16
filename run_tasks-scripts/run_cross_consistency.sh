#!/bin/bash
# ================================================
# Cross-Consistency Evaluation Runner
# ================================================
# Usage examples:
#   ./run_cross_consistency.sh --reasoning-model qwen3-4b --answering-model llama-8b --reasoning-task MedQA:CoT --answering-task MedQA:0-shot
#   ./run_cross_consistency.sh --model-group TINY --reasoning-task MedQA:CoT --answering-task MedQA:0-shot --limit 10

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Source configurations
source "${SCRIPT_DIR}/config/models.conf"
source "${SCRIPT_DIR}/config/tasks.conf"

# Default values
REASONING_MODELS=()
ANSWERING_MODELS=()
REASONING_TASK=""
ANSWERING_TASK=""
OUTPUT_BASE="./outputs/cross_consistency"
LIMIT=""
SEED="0"
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reasoning-model)
            IFS=',' read -ra MODEL_LIST <<< "$2"
            for m in "${MODEL_LIST[@]}"; do
                REASONING_MODELS+=("$(get_model_args "$m")")
            done
            shift 2
            ;;
        --answering-model)
            IFS=',' read -ra MODEL_LIST <<< "$2"
            for m in "${MODEL_LIST[@]}"; do
                ANSWERING_MODELS+=("$(get_model_args "$m")")
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
        --reasoning-task)
            REASONING_TASK="$2"
            shift 2
            ;;
        --answering-task)
            ANSWERING_TASK="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
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
Cross-Consistency Evaluation Runner

Cross-consistency evaluates reasoning chains by having multiple models verify each other's outputs.

Usage: $0 [OPTIONS]

Model Selection:
  --reasoning-model MODEL[,MODEL2,...]    Models that generate reasoning chains
  --answering-model MODEL[,MODEL2,...]    Models that verify reasoning chains
  --model-group GROUP                     Use predefined model group (TINY, 4B, 8B, MEDICAL, ALL)

Task Selection:
  --reasoning-task TASK                   Task for generating reasoning (e.g., MedQA:CoT)
  --answering-task TASK                   Task for verification (e.g., MedQA:0-shot)

Evaluation Options:
  --output PATH                           Base output directory (default: ./outputs/cross_consistency)
  --limit NUM                             Limit number of samples to evaluate
  --seed SEED                             Random seed (default: 0)

Other Options:
  --dry-run                               Show what would run without executing
  --config FILE                           Load configuration from file
  --help, -h                              Show this help message

Examples:
  # Basic cross-consistency with different models
  $0 --reasoning-model qwen3-4b --answering-model llama-8b \\
     --reasoning-task MedQA:CoT --answering-task MedQA:0-shot

  # Test with tiny models and limited samples
  $0 --model-group TINY --reasoning-task MedQA_0-shot:MedQA --answering-task MedQA_0-shot:MedQA --limit 10

  # Full evaluation with 8B models
  $0 --model-group 8B --reasoning-task MedQA:CoT --answering-task MedQA:0-shot

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

if [ -z "$REASONING_TASK" ]; then
    log_error "No reasoning task specified"
    exit 1
fi

if [ -z "$ANSWERING_TASK" ]; then
    log_error "No answering task specified"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python -m lm_eval.reasoning_modes"
CMD="$CMD --mode cross-consistency"
CMD="$CMD --reasoning_models $(printf '%s,' "${REASONING_MODELS[@]}" | sed 's/,$//')"
CMD="$CMD --answering_models $(printf '%s,' "${ANSWERING_MODELS[@]}" | sed 's/,$//')"
CMD="$CMD --reasoning_tasks $REASONING_TASK"
CMD="$CMD --answering_tasks $ANSWERING_TASK"
CMD="$CMD --output_path $OUTPUT_DIR"
CMD="$CMD --seed $SEED"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Show configuration
print_separator
log_info "Cross-Consistency Evaluation Configuration"
print_separator
echo "Provider:           $PROVIDER"
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
echo "Reasoning Task:     $REASONING_TASK"
echo "Answering Task:     $ANSWERING_TASK"
echo "Output Directory:   $OUTPUT_DIR"
echo "GPU:                $CUDA_DEVICES"
echo "Batch Size:         $BATCH_SIZE"
echo "Seed:               $SEED"
if [ -n "$LIMIT" ]; then
    echo "Limit:              $LIMIT"
else
    echo "Limit:              None"
fi
echo "Dry Run:            $DRY_RUN"
print_separator
log_info "Command: $CMD"

if [ "$DRY_RUN" = true ]; then
    log_info "Dry run - exiting without execution"
    exit 0
fi

# Execute
log_info "Starting cross-consistency evaluation..."

# Clear GPU memory before starting evaluation
log_info "Clearing GPU memory..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect(); print('GPU memory cleared')" || true

# Kill any remaining vLLM processes
pkill -f "EngineCore" || true
pkill -f "vllm" || true

if eval "$CMD"; then
    log_success "Cross-consistency evaluation completed!"
    log_info "Results saved to: $OUTPUT_DIR"
    
    # Show summary
    print_separator
    log_info "Evaluation Summary"
    print_separator
    echo "Total Runs:      1"
    echo "Successful:      1"
    echo "Failed:          0"
    echo ""
    echo "Successful Runs:"
    echo "  ✓ Cross-consistency: ${REASONING_TASK} -> ${ANSWERING_TASK}"
    print_separator
else
    log_error "Cross-consistency evaluation failed!"
    
    # Show summary
    print_separator
    log_info "Evaluation Summary"
    print_separator
    echo "Total Runs:      1"
    echo "Successful:      0"
    echo "Failed:          1"
    echo ""
    echo "Failed Runs:"
    echo "  ✗ Cross-consistency: ${REASONING_TASK} -> ${ANSWERING_TASK}"
    print_separator
    exit 1
fi</content>
<parameter name="filePath">/user/home/aguimas/data/PhD/Active_Dev/lm-evaluation-harness/run_tasks-scripts/run_cross_consistency.sh