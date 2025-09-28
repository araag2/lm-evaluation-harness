#!/bin/bash
# ================================
# Models Configuration
# ================================
PROVIDER=vllm

MODE=only-vote

BASE_OUTPUT_DIR="/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/$MODE"

CUDA_DEVICES=0
BATCH_SIZE=auto
SEED=0

LIST_OF_VOTE_FILES=(
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/MedNLI_CoT_SC/Qwen_Qwen3-4B-Instruct-2507/Summary_2025-09-27T18-43.json"
)

echo "=================================================="
echo "[INFO] Running only vote on these files:"
echo "Full list of vote files: ${LIST_OF_VOTE_FILES[@]}"
echo "=================================================="

echo "[INFO] Starting runs..."
for VOTE_FILE in "${LIST_OF_VOTE_FILES[@]}"; do
    echo "[INFO] Running evaluations for vote file: $VOTE_FILE"

    echo "=================================================="
    echo "[INFO] Running votes on:"
    echo "Provider = $PROVIDER"
    echo "Base Output Dir = $BASE_OUTPUT_DIR"
    echo "=================================================="

    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m lm_eval.reasoning_modes \
        --provider $PROVIDER \
        --mode $MODE \
        --output_path $BASE_OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --log_samples \
        --vote_file $VOTE_FILE \
        --limit 2

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Completed: $VOTE_FILE"
    else
        echo "[ERROR] Failed: $VOTE_FILE (exit code $STATUS)"
    fi
done

echo "[INFO] All Tasks completed."