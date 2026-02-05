#!/bin/bash
# ================================================
# Evaluation Utility Functions
# ================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Print separator
print_separator() {
    echo "=================================================="
}

# Extract model name from model args
get_model_name() {
    local model_args="$1"
    echo "$model_args" | sed -n 's/.*pretrained=\([^,)]*\).*/\1/p' | tr '/' '_'
}

# Build output path
build_output_path() {
    local base_dir="$1"
    local task="$2"
    local mode="$3"
    local model_name="$4"
    local timestamp="${5:-}"
    
    local path="${base_dir}/$(echo ${task} | tr ':' '_')"
    
    if [ -n "$mode" ]; then
        path="${path}/${mode}"
    fi
    
    if [ -n "$model_name" ]; then
        path="${path}/${model_name}"
    fi
    
    if [ -n "$timestamp" ]; then
        path="${path}/${timestamp}"
    fi
    
    echo "$path"
}

# Run single evaluation
run_single_evaluation() {
    local provider="$1"
    local model_args="$2"
    local task="$3"
    local mode="$4"
    local output_path="$5"
    local batch_size="${6:-auto}"
    local seed="${7:-0}"
    local cuda_devices="${8:-0}"
    local limit="${9:-}"
    
    log_info "Running: Task=${task}, Mode=${mode}"
    log_info "Output: ${output_path}"
    
    local cmd="VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -m lm_eval \
        --model $provider \
        --model_args \"$model_args\" \
        --tasks \"${task}_${mode}\" \
        --batch_size $batch_size \
        --seed $seed \
        --output_path \"$output_path\" \
        --log_samples"
    
    if [ -n "$limit" ]; then
        cmd="$cmd --limit $limit"
    fi
    
    eval "$cmd"
}

# Run multi-turn evaluation
run_multi_turn_evaluation() {
    local provider="$1"
    local mode="$2"
    local reasoning_model="$3"
    local answering_model="$4"
    local reasoning_task="$5"
    local answering_task="$6"
    local output_path="$7"
    local batch_size="${8:-auto}"
    local seed="${9:-0}"
    local cuda_devices="${10:-0}"
    
    log_info "Running Multi-Turn: ${reasoning_task} -> ${answering_task}"
    log_info "Output: ${output_path}"
    
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -m lm_eval.reasoning_modes \
        --provider $provider \
        --mode $mode \
        --reasoning_models "$reasoning_model" \
        --answering_models "$answering_model" \
        --reasoning_tasks "$reasoning_task" \
        --answering_tasks "$answering_task" \
        --output_path "$output_path" \
        --batch_size $batch_size \
        --seed $seed \
        --log_samples
    
    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Completed: ${reasoning_task} -> ${answering_task}"
        return 0
    else
        log_error "Failed: ${reasoning_task} -> ${answering_task} - Exit code: $status"
        return $status
    fi
}

# Run cross-consistency evaluation
run_cross_consistency() {
    local provider="$1"
    local reasoning_model="$2"
    local answering_models="$3"  # Space-separated list
    local reasoning_task="$4"
    local answering_task="$5"
    local output_path="$6"
    local vote_file="${7:-}"
    local batch_size="${8:-auto}"
    local seed="${9:-0}"
    local cuda_devices="${10:-0}"
    
    log_info "Running Cross-Consistency: ${reasoning_task} -> ${answering_task}"
    log_info "Output: ${output_path}"
    
    local cmd="VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -m lm_eval.reasoning_modes \
        --provider $provider \
        --mode cross-consistency \
        --reasoning_models $reasoning_model \
        --answering_models $answering_models \
        --reasoning_tasks $reasoning_task \
        --answering_tasks $answering_task \
        --output_path $output_path \
        --batch_size $batch_size \
        --seed $seed \
        --log_samples"
    
    if [ -n "$vote_file" ]; then
        cmd="$cmd --vote_file $vote_file"
    fi
    
    eval $cmd
    
    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Completed Cross-Consistency"
        return 0
    else
        log_error "Failed Cross-Consistency - Exit code: $status"
        return $status
    fi
}

# Run vote-only evaluation
run_vote_only() {
    local provider="$1"
    local vote_file="$2"
    local output_path="$3"
    local batch_size="${4:-auto}"
    local seed="${5:-0}"
    local cuda_devices="${6:-0}"
    
    log_info "Running Vote-Only on: $(basename $vote_file)"
    log_info "Output: ${output_path}"
    
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -m lm_eval.reasoning_modes \
        --provider $provider \
        --mode only-vote \
        --output_path "$output_path" \
        --batch_size $batch_size \
        --seed $seed \
        --log_samples \
        --vote_file "$vote_file"
    
    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Completed Vote-Only: $(basename $vote_file)"
        return 0
    else
        log_error "Failed Vote-Only - Exit code: $status"
        return $status
    fi
}

# Check if GPU is available
check_gpu() {
    local gpu_id="${1:-0}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. Cannot verify GPU availability."
        return 0
    fi
    
    if nvidia-smi -i $gpu_id &> /dev/null; then
        log_info "GPU $gpu_id is available"
        return 0
    else
        log_error "GPU $gpu_id is not available"
        return 1
    fi
}

# Estimate time and show progress
show_progress() {
    local current="$1"
    local total="$2"
    local task_name="$3"
    
    local percent=$((current * 100 / total))
    log_info "Progress: [$current/$total] ($percent%) - $task_name"
}

# Create timestamped output directory
create_output_dir() {
    local base_path="$1"
    local use_timestamp="${2:-false}"
    
    local output_path="$base_path"
    
    if [ "$use_timestamp" = "true" ]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        output_path="${base_path}/${timestamp}"
    fi
    
    mkdir -p "$output_path"
    echo "$output_path"
}

# Parse model preset name to get full args
get_model_args() {
    local preset="$1"
    
    # Source config if not already loaded
    if [ -z "$MODEL_QWEN3_4B" ]; then
        local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${script_dir}/../config/models.conf"
    fi
    
    case "$preset" in
        qwen3-4b) echo "$MODEL_QWEN3_4B" ;;
        gemma-4b) echo "$MODEL_GEMMA_4B" ;;
        qwen3-8b) echo "$MODEL_QWEN3_8B" ;;
        llama-8b) echo "$MODEL_LLAMA_8B" ;;
        deepseek-8b) echo "$MODEL_DEEPSEEK_8B" ;;
        ministral-8b) echo "$MODEL_MINISTRAL_8B" ;;
        fleming-7b) echo "$MODEL_FLEMING_7B" ;;
        panacea-7b) echo "$MODEL_PANACEA_7B" ;;
        *) echo "$preset" ;;  # Return as-is if not a preset
    esac
}

# Display summary statistics
show_summary() {
    local total_runs="$1"
    local successful_runs="$2"
    local failed_runs="$3"
    local start_time="$4"
    local end_time="$5"
    
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    print_separator
    log_info "Evaluation Summary"
    print_separator
    echo "Total Runs:      $total_runs"
    echo "Successful:      $successful_runs"
    echo "Failed:          $failed_runs"
    echo "Duration:        ${hours}h ${minutes}m ${seconds}s"
    print_separator
}