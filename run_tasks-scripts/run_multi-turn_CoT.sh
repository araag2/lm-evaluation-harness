#!/bin/bash
# ================================
# Models Configuration
# ================================
PROVIDER=vllm

#"pretrained=unsloth/Qwen3-8B,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8|pretrained=unsloth/Qwen3-8B,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8|pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8|pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=UbiquantAI/Fleming-R1-7B,max_length=22000,gpu_memory_utilization=0.75,swap_space=8|pretrained=UbiquantAI/Fleming-R1-7B,max_length=22000,gpu_memory_utilization=0.75,swap_space=8"

#"pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8|pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8|pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=google/gemma-3n-E4B-it,max_length=22000,gpu_memory_utilization=0.75,swap_space=8|pretrained=google/gemma-3n-E4B-it,max_length=22000,gpu_memory_utilization=0.75,swap_space=8"
#"pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8|pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=22000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"

PAIRS_OF_MODELS=(
    "pretrained=unsloth/Qwen3-8B,max_length=20000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8|pretrained=unsloth/Qwen3-8B,max_length=20000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8"
    "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=20000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8|pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=20000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8"
    "pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,max_length=20000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8|pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,max_length=20000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8"
    "pretrained=UbiquantAI/Fleming-R1-7B,max_length=20000,gpu_memory_utilization=0.9,swap_space=8|pretrained=UbiquantAI/Fleming-R1-7B,max_length=20000,gpu_memory_utilization=0.9,swap_space=8"
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
    "Trial_Meta_Analysis:type_CoT|Trial_Meta_Analysis:type_0-shot"
)

MODE=multi-turn_CoT

BASE_OUTPUT_DIR="../outputs/resource_paper/$MODE"

CUDA_DEVICES=0
BATCH_SIZE=auto
SEED=0

echo "=================================================="
echo "[INFO] Running These Models / Tasks:"
echo "Full list of model pairs: ${PAIRS_OF_MODELS[@]}"
echo "Full list of task pairs: ${PAIRS_OF_TASK_LIST[@]}"
echo "=================================================="

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

    OUTPUT_PATH="${BASE_OUTPUT_DIR}/$(echo ${TASK_REASON} | tr ':' '_')/$(echo ${MODEL_REASON} | sed -n 's/.*pretrained=\([^,)]*\).*/\1/p' | tr '/' '_')/"

    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m lm_eval.reasoning_modes \
        --provider $PROVIDER \
        --mode $MODE \
        --reasoning_models $MODEL_REASON \
        --answering_models $MODEL_ANSWER \
        --reasoning_tasks $TASK_REASON \
        --answering_tasks $TASK_ANSWER \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --log_samples
        
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Completed: $TASK_REASON to $TASK_ANSWER"
    else
        echo "[ERROR] Failed: $TASK_REASON to $TASK_ANSWER (exit code $STATUS)"
    fi

    done
done

echo "[INFO] All Tasks completed."