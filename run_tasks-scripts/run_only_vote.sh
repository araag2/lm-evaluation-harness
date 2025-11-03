#!/bin/bash
# ================================
# Models Configuration
# ================================
PROVIDER=vllm

MODE=only-vote

BASE_OUTPUT_DIR="./outputs/$MODE"

CUDA_DEVICES=0
BATCH_SIZE=auto
SEED=0

LIST_OF_VOTE_FILES=(
    # TO:DO - Add the list of vote files here
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
        --vote_file $VOTE_FILE

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Completed: $VOTE_FILE"
    else
        echo "[ERROR] Failed: $VOTE_FILE (exit code $STATUS)"
    fi
done

echo "[INFO] All Tasks completed."