#!/bin/bash
# ================================================
# Interactive Evaluation Menu
# ================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Source configurations
source "${SCRIPT_DIR}/config/models.conf"
source "${SCRIPT_DIR}/config/tasks.conf"
source "${SCRIPT_DIR}/config/tasks_pairs.conf"

# Color for menu
CYAN='\033[0;36m'
NC='\033[0m'

# Menu functions
print_menu_header() {
    clear
    print_separator
    echo -e "${CYAN}LM Evaluation Harness - Interactive Runner${NC}"
    print_separator
    echo ""
}

select_mode() {
    echo "Select Evaluation Mode:"
    select MODE in "Single Task" "Multi-Turn CoT" "Multi-Turn SC-CoT" "Cross-Consistency" "Vote Only" "Exit"; do
        case $MODE in
            "Single Task") EVAL_MODE="single"; break ;;
            "Multi-Turn CoT") EVAL_MODE="multi-turn_CoT"; break ;;
            "Multi-Turn SC-CoT") EVAL_MODE="multi-turn_CoT-SC"; break ;;
            "Cross-Consistency") EVAL_MODE="cross-consistency"; break ;;
            "Vote Only") EVAL_MODE="vote-only"; break ;;
            "Exit") exit 0 ;;
            *) echo "Invalid selection" ;;
        esac
    done
    echo ""
}

select_models() {
    echo "Select Model Group:"
    select MODEL_GROUP in "Tiny Models" "4B Models" "8B Models" "Medical Models" "All Models" "Custom"; do
        case $MODEL_GROUP in
            "Tiny Models") SELECTED_MODELS=("${MODELS_TINY[@]}"); break ;;
            "4B Models") SELECTED_MODELS=("${MODELS_4B[@]}"); break ;;
            "8B Models") SELECTED_MODELS=("${MODELS_8B[@]}"); break ;;
            "Medical Models") SELECTED_MODELS=("${MODELS_MEDICAL[@]}"); break ;;
            "All Models") SELECTED_MODELS=("${MODELS_ALL[@]}"); break ;;
            "Custom")
                echo "Available model presets:"
                echo "  qwen3-0.5b, gemma-270m, qwen3-4b, gemma-4b,"
                echo "  qwen3-8b, llama-8b, deepseek-8b, ministral-8b,"
                echo "  fleming-7b, panacea-7b"
                read -p "Enter model preset(s) (comma-separated): " MODEL_INPUT
                IFS=',' read -ra MODEL_LIST <<< "$MODEL_INPUT"
                SELECTED_MODELS=()
                for m in "${MODEL_LIST[@]}"; do
                    SELECTED_MODELS+=("$(get_model_args "$m")")
                done
                break
                ;;
            *) echo "Invalid selection" ;;
        esac
    done
    echo ""
    log_info "Selected ${#SELECTED_MODELS[@]} model(s)"
}

select_tasks() {
    echo "Select Task Group:"
    select TASK_GROUP in "QA Tasks" "NLI Tasks" "IE Tasks" "ES Tasks" "Ranking Tasks" "TrialBench" "TrialPanorama" "TREC Tasks" "Small Tasks" "All Tasks" "Custom"; do
        case $TASK_GROUP in
            "QA Tasks") SELECTED_TASKS=("${QA_TASKS[@]}"); break ;;
            "NLI Tasks") SELECTED_TASKS=("${NLI_TASKS[@]}"); break ;;
            "IE Tasks") SELECTED_TASKS=("${IE_TASKS[@]}"); break ;;
            "ES Tasks") SELECTED_TASKS=("${ES_TASKS[@]}"); break ;;
            "Ranking Tasks") SELECTED_TASKS=("${RANKING_TASKS[@]}"); break ;;
            "TrialBench") SELECTED_TASKS=("${TRIALBENCH_TASKS[@]}"); break ;;
            "TrialPanorama") SELECTED_TASKS=("${TRIALPANORAMA_TASKS[@]}"); break ;;
            "TREC Tasks") SELECTED_TASKS=("${TREC_TASKS[@]}"); break ;;
            "Small Tasks") SELECTED_TASKS=("${SMALL_TASKS[@]}"); break ;;
            "All Tasks") SELECTED_TASKS=("${ALL_TASKS[@]}"); break ;;
            "Custom")
                read -p "Enter task name(s) (comma-separated): " TASK_INPUT
                IFS=',' read -ra SELECTED_TASKS <<< "$TASK_INPUT"
                break
                ;;
            *) echo "Invalid selection" ;;
        esac
    done
    echo ""
    log_info "Selected ${#SELECTED_TASKS[@]} task(s)"
}

select_task_pairs() {
    echo "Select Task Pair Group:"
    select PAIR_GROUP in "QA Pairs" "NLI Pairs" "IE Pairs" "ES Pairs" "Ranking Pairs" "TrialBench Pairs" "TrialPanorama Pairs" "TREC Pairs" "Small Pairs" "All Pairs" "Custom"; do
        case $PAIR_GROUP in
            "QA Pairs") SELECTED_PAIRS=("${QA_TASK_PAIRS[@]}"); break ;;
            "NLI Pairs") SELECTED_PAIRS=("${NLI_TASK_PAIRS[@]}"); break ;;
            "IE Pairs") SELECTED_PAIRS=("${IE_TASK_PAIRS[@]}"); break ;;
            "ES Pairs") SELECTED_PAIRS=("${ES_TASK_PAIRS[@]}"); break ;;
            "Ranking Pairs") SELECTED_PAIRS=("${RANKING_TASK_PAIRS[@]}"); break ;;
            "TrialBench Pairs") SELECTED_PAIRS=("${TRIALBENCH_TASK_PAIRS[@]}"); break ;;
            "TrialPanorama Pairs") SELECTED_PAIRS=("${TRIALPANORAMA_TASK_PAIRS[@]}"); break ;;
            "TREC Pairs") SELECTED_PAIRS=("${TREC_TASK_PAIRS[@]}"); break ;;
            "Small Pairs") SELECTED_PAIRS=("${SMALL_TASK_PAIRS[@]}"); break ;;
            "All Pairs") SELECTED_PAIRS=("${ALL_TASK_PAIRS[@]}"); break ;;
            "Custom")
                read -p "Enter task pair (format: 'task:CoT|task:0-shot'): " PAIR_INPUT
                SELECTED_PAIRS=("$PAIR_INPUT")
                break
                ;;
            *) echo "Invalid selection" ;;
        esac
    done
    echo ""
    log_info "Selected ${#SELECTED_PAIRS[@]} task pair(s)"
}

select_inference_modes() {
    echo "Select Inference Mode(s):"
    echo "1) 0-shot"
    echo "2) CoT"
    echo "3) SC"
    echo "4) All modes"
    read -p "Enter choice (1-4): " MODE_CHOICE
    
    case $MODE_CHOICE in
        1) INFERENCE_MODES=("0-shot") ;;
        2) INFERENCE_MODES=("CoT") ;;
        3) INFERENCE_MODES=("SC") ;;
        4) INFERENCE_MODES=("0-shot" "CoT" "SC") ;;
        *) INFERENCE_MODES=("0-shot") ;;
    esac
    echo ""
    log_info "Selected modes: ${INFERENCE_MODES[*]}"
}

configure_runtime() {
    echo "Runtime Configuration:"
    
    read -p "CUDA device ID [0]: " CUDA_INPUT
    CUDA_DEVICES=${CUDA_INPUT:-0}
    
    read -p "Batch size [auto]: " BATCH_INPUT
    BATCH_SIZE=${BATCH_INPUT:-auto}
    
    read -p "Random seed [0]: " SEED_INPUT
    SEED=${SEED_INPUT:-0}
    
    read -p "Output base directory [./outputs/interactive]: " OUTPUT_INPUT
    OUTPUT_BASE=${OUTPUT_INPUT:-./outputs/interactive}
    
    read -p "Add timestamp to output paths? [y/N]: " TIMESTAMP_INPUT
    if [[ "$TIMESTAMP_INPUT" =~ ^[Yy]$ ]]; then
        USE_TIMESTAMP=true
    else
        USE_TIMESTAMP=false
    fi
    
    echo ""
}

show_summary_and_confirm() {
    print_separator
    log_info "Interactive Evaluation Configuration"
    print_separator
    echo "Mode:            $EVAL_MODE"
    echo "Models:          ${#SELECTED_MODELS[@]}"
    echo "Model Details:   $(printf '\n%s' "${SELECTED_MODELS[@]}" | sed 's/^/  - /')"

    if [ "$EVAL_MODE" = "single" ]; then
        echo "Tasks:           ${#SELECTED_TASKS[@]}"
        echo "Task Details:    $(printf '\n%s' "${SELECTED_TASKS[@]}" | sed 's/^/  - /')"
        echo "Inference Modes: ${INFERENCE_MODES[*]}"
        TOTAL_RUNS=$((${#SELECTED_MODELS[@]} * ${#SELECTED_TASKS[@]} * ${#INFERENCE_MODES[@]}))
    elif [ "$EVAL_MODE" = "cross-consistency" ]; then
        echo "Task Pairs:      ${#SELECTED_PAIRS[@]}"
        echo "Pair Details:    $(printf '\n%s' "${SELECTED_PAIRS[@]}" | sed 's/^/  - /')"
        TOTAL_RUNS=${#SELECTED_PAIRS[@]}
    else
        echo "Task Pairs:      ${#SELECTED_PAIRS[@]}"
        echo "Pair Details:    $(printf '\n%s' "${SELECTED_PAIRS[@]}" | sed 's/^/  - /')"
        TOTAL_RUNS=$((${#SELECTED_MODELS[@]} * ${#SELECTED_PAIRS[@]}))
    fi

    echo "GPU:             $CUDA_DEVICES"
    echo "Batch Size:      $BATCH_SIZE"
    echo "Seed:            $SEED"
    echo "Output Base:     $OUTPUT_BASE"
    echo "Timestamp:       $USE_TIMESTAMP"
    echo "Total Runs:      $TOTAL_RUNS"
    print_separator

    read -p "Proceed with evaluation? [y/N]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        log_info "Evaluation cancelled"
        exit 0
    fi
    echo ""
}

run_single_mode() {
    local total_runs=$((${#SELECTED_MODELS[@]} * ${#SELECTED_TASKS[@]} * ${#INFERENCE_MODES[@]}))
    local current_run=0
    local successful=0
    local failed=0
    
    for MODEL_ARGS in "${SELECTED_MODELS[@]}"; do
        MODEL_NAME=$(get_model_name "$MODEL_ARGS")
        
        for TASK in "${SELECTED_TASKS[@]}"; do
            for MODE in "${INFERENCE_MODES[@]}"; do
                current_run=$((current_run + 1))
                show_progress "$current_run" "$total_runs" "${TASK} (${MODE})"
                
                OUTPUT_PATH=$(build_output_path "$OUTPUT_BASE" "$TASK" "$MODE" "$MODEL_NAME")
                
                if [ "$USE_TIMESTAMP" = true ]; then
                    OUTPUT_PATH=$(create_output_dir "$OUTPUT_PATH" true)
                else
                    mkdir -p "$OUTPUT_PATH"
                fi
                
                if run_single_evaluation "vllm" "$MODEL_ARGS" "$TASK" "$MODE" "$OUTPUT_PATH" \
                                        "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES"; then
                    successful=$((successful + 1))
                else
                    failed=$((failed + 1))
                fi
                echo ""
            done
        done
    done
    
    return $failed
}

run_cross_consistency_mode() {
    local total_runs=${#SELECTED_PAIRS[@]}
    local current_run=0
    local successful=0
    local failed=0

    local reasoning_models_str=$(IFS=','; echo "${SELECTED_MODELS[*]}")
    local answering_models_str="$reasoning_models_str"

    for TASK_PAIR in "${SELECTED_PAIRS[@]}"; do
        current_run=$((current_run + 1))

        REASONING_TASK="${TASK_PAIR%%|*}"
        ANSWERING_TASK="${TASK_PAIR##*|}"

        show_progress "$current_run" "$total_runs" "${REASONING_TASK} -> ${ANSWERING_TASK}"

        TASK_NAME="${ANSWERING_TASK%%:*}"
        OUTPUT_PATH="${OUTPUT_BASE}/cross-consistency/${TASK_NAME}"

        if [ "$USE_TIMESTAMP" = true ]; then
            OUTPUT_PATH=$(create_output_dir "$OUTPUT_PATH" true)
        else
            mkdir -p "$OUTPUT_PATH"
        fi

        if run_multi_turn_evaluation "vllm" "cross-consistency" \
                                    "$reasoning_models_str" "$answering_models_str" \
                                    "$REASONING_TASK" "$ANSWERING_TASK" \
                                    "$OUTPUT_PATH" "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES"; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
        echo ""
    done

    return $failed
}

run_multi_turn_mode() {
    local total_runs=$((${#SELECTED_MODELS[@]} * ${#SELECTED_PAIRS[@]}))
    local current_run=0
    local successful=0
    local failed=0
    
    for MODEL_ARGS in "${SELECTED_MODELS[@]}"; do
        MODEL_NAME=$(get_model_name "$MODEL_ARGS")
        
        for TASK_PAIR in "${SELECTED_PAIRS[@]}"; do
            current_run=$((current_run + 1))
            
            REASONING_TASK="${TASK_PAIR%%|*}"
            ANSWERING_TASK="${TASK_PAIR##*|}"
            
            show_progress "$current_run" "$total_runs" "${REASONING_TASK} -> ${ANSWERING_TASK}"
            
            TASK_NAME="${REASONING_TASK%%:*}"
            OUTPUT_PATH="${OUTPUT_BASE}/${EVAL_MODE}/${TASK_NAME}/${MODEL_NAME}"
            
            if [ "$USE_TIMESTAMP" = true ]; then
                OUTPUT_PATH=$(create_output_dir "$OUTPUT_PATH" true)
            else
                mkdir -p "$OUTPUT_PATH"
            fi
            
            if run_multi_turn_evaluation "vllm" "$EVAL_MODE" \
                                        "$MODEL_ARGS" "$MODEL_ARGS" \
                                        "$REASONING_TASK" "$ANSWERING_TASK" \
                                        "$OUTPUT_PATH" "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES"; then
                successful=$((successful + 1))
            else
                failed=$((failed + 1))
            fi
            echo ""
        done
    done
    
    return $failed
}

# Main menu flow
main() {
    print_menu_header
    
    # Step 1: Select mode
    select_mode
    
    # Step 2: Select models
    select_models
    
    # Step 3: Select tasks or task pairs
    if [ "$EVAL_MODE" = "single" ]; then
        select_tasks
        select_inference_modes
    else
        select_task_pairs
    fi
    
    # Step 4: Configure runtime
    configure_runtime
    
    # Step 5: Show summary and confirm
    show_summary_and_confirm
    
    # Step 6: Check GPU
    check_gpu "$CUDA_DEVICES"
    
    # Step 7: Execute
    START_TIME=$(date +%s)
    
    if [ "$EVAL_MODE" = "single" ]; then
        run_single_mode
        RESULT=$?
    elif [ "$EVAL_MODE" = "cross-consistency" ]; then
        run_cross_consistency_mode
        RESULT=$?
    else
        run_multi_turn_mode
        RESULT=$?
    fi
    
    END_TIME=$(date +%s)
    
    # Step 8: Show final summary
    # Note: Would need to track successful/failed runs properly
    log_info "Evaluation completed"
    
    exit $RESULT
}

# Run main menu
main
