#!/bin/bash
# ===============================================
# Cross-Consistency Evaluation Bash Script
# ===============================================

PROVIDER=vllm

#"pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=25000|pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=25000"
#"pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=25000|pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=25000"
#"pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=25000|pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=25000"
#"pretrained=google/gemma-3n-E4B-it,max_length=25000|pretrained=google/gemma-3n-E4B-it,max_length=25000"
#"pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=25000|pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=25000"
REASONING_MODELS=(
    "pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=25000"
    "pretrained=google/gemma-3n-E4B-it,max_length=25000"
)


ANSWERING_MODELS=("${REASONING_MODELS[@]}")

#"MedNLI:CoT|MedNLI:0-shot" WORKS
#"HINT:CoT|HINT:0-shot" WORKS 
#"MedMCQA:CoT|MedMCQA:0-shot" X 
#"MedQA:CoT|MedQA:0-shot"  X
#"PubMedQA:CoT|PubMedQA:0-shot" WORKS
#"Evidence_Inference_v2:CoT|Evidence_Inference_v2:0-shot" WORKS
#"NLI4PR_patient-lang_CoT|NLI4PR_patient-lang_0-shot" X
#"NLI4PR_medical-lang_CoT|NLI4PR_medical-lang_0-shot" X
#"SemEval_NLI4CT_2023_CoT|SemEval_NLI4CT_2023_0-shot" X
#"SemEval_NLI4CT_2024_CoT|SemEval_NLI4CT_2024_0-shot" X

PAIRS_OF_TASK_LIST=(
    "MedNLI:CoT|MedNLI:0-shot"
)

MODE=cross-consistency  # Updated mode to cross-consistency

BASE_OUTPUT_DIR="/cfs/home/u021010/PhD/active_dev/outputs/TEST/$MODE"

CUDA_DEVICES=2
BATCH_SIZE=auto
SEED=0

echo "=================================================="
echo "[INFO] Running Cross-Consistency with Reasoning Models:"
printf '%s\n' "${REASONING_MODELS[@]}"
echo "=================================================="

for TASK_PAIR in "${PAIRS_OF_TASK_LIST[@]}"; do
    TASK_REASON=${TASK_PAIR%%|*}
    TASK_ANSWER=${TASK_PAIR##*|}

    echo "--------------------------------------------------"
    echo "[INFO] Task Pair:"
    echo "  Reasoning Task: $TASK_REASON"
    echo "  Answering Task: $TASK_ANSWER"
    echo "--------------------------------------------------"

    # Construct a clean directory name based on first reasoning model
    MODEL_NAME_CLEAN=$(echo "${REASONING_MODELS[0]}" | sed -n 's/.*pretrained=\([^,)]*\).*/\1/p' | tr '/' '_')

    OUTPUT_PATH="${BASE_OUTPUT_DIR}/$(echo ${TASK_REASON} | tr ':' '_')/${MODEL_NAME_CLEAN}/"

    echo "[INFO] Output Path: $OUTPUT_PATH"

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    python -m lm_eval.reasoning_modes \
        --provider $PROVIDER \
        --mode $MODE \
        --reasoning_models ${REASONING_MODELS[@]} \
        --answering_models ${ANSWERING_MODELS[@]} \
        --reasoning_tasks $TASK_REASON \
        --answering_tasks $TASK_ANSWER \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --log_samples \
        --limit 1

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Cross-Consistency Completed for $TASK_REASON"
    else
        echo "[ERROR] Cross-Consistency Failed for $TASK_REASON (exit code $STATUS)"
    fi
done

echo "[INFO] All Cross-Consistency evaluations completed."
