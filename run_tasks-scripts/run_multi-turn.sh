#!/bin/bash
# ================================
# Models Configuration
# ================================
PROVIDER=vllm

PAIRS_OF_MODELS=(
    "pretrained=Qwen/Qwen3-0.6B|pretrained=Qwen/Qwen3-0.6B"
)

# Available Datasets: Evidence_Inference_v2, HINT, MedMCQA, MedNLI, MedQA, NLI4PR, PubMedQA, SemEval_NLI4CT, TREC_CDS, TREC_CT, TREC_Prec-Med, Trial_Meta-Analysis_type
PAIRS_OF_TASK_LIST=(
    "MedNLI:CoT|MedNLI:0-shot"
)

BASE_OUTPUT_DIR="/cfs/home/u021010/PhD/active_dev/outputs/Multi-Turn-Debug/"

BATCH_SIZE=auto
SEED=0

echo "[INFO] Starting runs..."
for PAIR_MODELS in "${PAIRS_OF_MODELS[@]}"; do
    MODEL_REASON=${PAIR_MODELS%%|*}
    MODEL_ANSWER=${PAIR_MODELS##*|}

    for PAIR_TASKS in "${PAIRS_OF_TASK_LIST[@]}"; do
        TASK_REASON=${PAIR_TASKS%%|*}
        TASK_ANSWER=${PAIR_TASKS##*|}

    echo "=================================================="
    echo "[INFO] Running evaluations:"
    echo "Provider = $PROVIDER"
    echo "Model Reason = $MODEL_REASON"
    echo "Model Answer = $MODEL_ANSWER"
    echo "Task Reason = $TASK_REASON"
    echo "Task Answer = $TASK_ANSWER"
    echo "Base Output Dir = $BASE_OUTPUT_DIR"
    echo "=================================================="

    OUTPUT_PATH="${BASE_OUTPUT_DIR}/$(echo ${TASK_REASON} | tr ':' '_')/$(echo ${MODEL_REASON#pretrained=} | tr '/' '_')/"

    mkdir -p "$OUTPUT_PATH"

    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 python -m lm_eval.multi-turn_cross-consistency \
        --provider $PROVIDER \
        --mode multi-turn \
        --reasoning_models $MODEL_REASON \
        --answering_models $MODEL_ANSWER \
        --reasoning_tasks $TASK_REASON \
        --answering_tasks $TASK_ANSWER \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --log_samples
        #--limit 2 #DEBUG ONLY


    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Completed: $TASK_REASON to $TASK_ANSWER"
    else
        echo "[ERROR] Failed: $TASK_REASON to $TASK_ANSWER (exit code $STATUS)"
    fi

    done
done

echo "[INFO] All Tasks completed."