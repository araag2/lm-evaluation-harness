#!/bin/bash
# ===============================================
# Cross-Consistency Evaluation Bash Script
# ===============================================

PROVIDER=vllm

#"pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8|pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8"
#"pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8|pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8"
#"pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8|pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8"
#"pretrained=google/gemma-3n-E4B-it,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=google/gemma-3n-E4B-it,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
#"pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8|pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=22000,gpu_memory_utilization=0.8,dtype=float16,swap_space=8"

REASONING_MODELS=(
    "pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
)


#ANSWERING_MODELS=("${REASONING_MODELS[@]}")

ANSWERING_MODELS=(
    "pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
    "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
    "pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
    "pretrained=google/gemma-3n-E4B-it,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=google/gemma-3n-E4B-it,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
    "pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=22000,gpu_memory_utilization=0.8,swap_space=8|pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=22000,gpu_memory_utilization=0.8,swap_space=8"
)

#"MedNLI:CoT|MedNLI:0-shot"
#"HINT:CoT|HINT:0-shot"
#"MedMCQA:CoT|MedMCQA:0-shot" 
#"MedQA:CoT|MedQA:0-shot"
#"PubMedQA:CoT|PubMedQA:0-shot"
#"Evidence_Inference_v2:CoT|Evidence_Inference_v2:0-shot"
#"NLI4PR:patient-lang_CoT|NLI4PR:patient-lang_0-shot"
#"NLI4PR:medical-lang_CoT|NLI4PR:medical-lang_0-shot"
#"SemEval_NLI4CT:2023_CoT|SemEval_NLI4CT:2023_0-shot"
#"SemEval_NLI4CT:2024_CoT|SemEval_NLI4CT:2024_0-shot"

PAIRS_OF_TASK_LIST=(
    "MedNLI:CoT|MedNLI:0-shot"
    "MedMCQA:CoT|MedMCQA:0-shot" 
    "MedQA:CoT|MedQA:0-shot"
    "PubMedQA:CoT|PubMedQA:0-shot"
)

MODE=cross-consistency 

BASE_OUTPUT_DIR="/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/$MODE-single-reasoning"

CUDA_DEVICES=0
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
        --log_samples 

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Cross-Consistency Completed for $TASK_REASON"
    else
        echo "[ERROR] Cross-Consistency Failed for $TASK_REASON (exit code $STATUS)"
    fi
done

echo "[INFO] All Cross-Consistency evaluations completed."