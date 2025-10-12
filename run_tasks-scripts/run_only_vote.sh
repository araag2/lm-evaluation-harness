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
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/PubMedQA_CoT_SC/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/Summary_2025-09-30T01-18.json"
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/PubMedQA_CoT_SC/google_gemma-3n-E4B-it/Summary_2025-09-29T11-36.json"
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/PubMedQA_CoT_SC/meta-llama_Llama-3.1-8B-Instruct/Summary_2025-09-29T16-12.json"
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/PubMedQA_CoT_SC/mistralai_Ministral-8B-Instruct-2410/Summary_2025-09-28T15-32.json"
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/PubMedQA_CoT_SC/Qwen_Qwen3-4B-Instruct-2507/Summary_2025-09-29T15-18.json"
    "/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/SC-Same-Params-as-CoT/multi-turn_CoT-SC/PubMedQA_CoT_SC/UbiquantAI_Fleming-R1-7B/Summary_2025-09-30T05-10.json"
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