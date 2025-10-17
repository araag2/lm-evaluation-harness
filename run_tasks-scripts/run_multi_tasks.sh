#!/bin/bash

# ================================
# Models Configuration
# ================================
MODEL=vllm

#"pretrained=unsloth/Qwen3-8B,max_length=25000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=25000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,max_length=25000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=UbiquantAI/Fleming-R1-7B,max_length=25000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"
#"pretrained=linjc16/Panacea-7B-Chat,max_length=25000,gpu_memory_utilization=0.75,dtype=float16,swap_space=8"

#"pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=25000"
#"pretrained=google/gemma-3n-E4B-it,max_length=25000"
#"pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=25000"

MODELS=(
    "pretrained=unsloth/Qwen3-8B,max_length=25000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8,enable_prefix_caching=True"
    "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=25000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8,enable_prefix_caching=True"
    "pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,max_length=25000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8,enable_prefix_caching=True"
    "pretrained=UbiquantAI/Fleming-R1-7B,max_length=25000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8,enable_prefix_caching=True"
)

#MedNLI
#MedQA
#MedMCQA
#PubMedQA
#RCT_Summary
#Evidence_Inference_v2
#NLI4PR
#HINT
#TREC_CDS
#TREC_Prec-Med
#TREC_CT
#SemEval_NLI4CT
#Trial_Meta_Analysis_type
#Trial_Meta_Analysis_binary
#Trial_Meta_Analysis_continuous

TASK_LIST=(
    TREC_Prec-Med
    TREC_CDS
    TREC_CT
)

INFERENCE_MODES=(
    #0-shot
    #SC
    CoT
)

# Generation Params
CUDA_DEVICES=3  
BATCH_SIZE=auto
SEED=0

# Output base path
OUTPUT_BASE_PATH=../outputs/
RUN_NAME=resource_paper/

for MODEL_ARGS in "${MODELS[@]}"; do

    echo "=================================================="
    echo "[INFO] Running evaluations for:\n Model = $MODEL\n Model Args = $MODEL_ARGS\n. Output Base Path = $OUTPUT_BASE_PATH"
    echo "=================================================="

    echo -e "=================================================="
    echo "[INFO] Starting batch execution of ${TASK_LIST[@]} task scripts."
    echo "=================================================="

    echo -e "=================================================="
    echo "[INFO] Inference Modes: ${INFERENCE_MODES[@]}"
    echo "=================================================="

    # Loop over tasks
    for TASK in "${TASK_LIST[@]}"; do
        for INFERENCE_MODE in "${INFERENCE_MODES[@]}"; do
            OUTPUT_PATH="${OUTPUT_BASE_PATH}/${RUN_NAME}${TASK}/${INFERENCE_MODE}"

            echo -e "--------------------------------------------------"
            echo -e "[INFO] Running Task: $TASK with Inference Mode: $INFERENCE_MODE"
            echo -e "--------------------------------------------------"

            VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m lm_eval \
                --model $MODEL \
                --model_args $MODEL_ARGS \
                --tasks "${TASK}_${INFERENCE_MODE}" \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --output_path $OUTPUT_PATH \
                --log_samples
                #--write_out \
                #--apply_chat_template \
                #--limit 2 \
                #--predict_only \

            STATUS=$?
            if [ $STATUS -eq 0 ]; then
                echo "[SUCCESS] Completed: $TASK in Inference Mode: $INFERENCE_MODE"
            else
                echo "[ERROR] Failed: $TASK in Inference Mode: $INFERENCE_MODE (exit code $STATUS)"
            fi

        done
        echo -e "-------------------------------\n"
    done
done

echo "[INFO] All evaluations completed."
