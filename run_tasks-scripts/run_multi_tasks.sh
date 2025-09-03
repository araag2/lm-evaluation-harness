#!/bin/bash

# ================================
# Models Configuration
# ================================
MODEL=vllm

MODELS=(
    #"pretrained=meta-llama/Llama-3.1-8B-Instruct,max_length=20000"
    #"pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=20000"
    "pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=20000"
    #"pretrained=google/gemma-3n-E4B-it,max_length=20000"
    #"pretrained=mistralai/Ministral-8B-Instruct-2410,max_length=20000"
)

# meta-llama/Llama-3.1-8B-Instruct
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# Qwen/Qwen3-4B-Instruct-2507
# google/gemma-3n-E4B-it
# mistralai/Ministral-8B-Instruct-2410


# Tasks List (space-separated)
TASK_LIST=(
    #MedNLI
    #MedQA
    #MedMCQA
    #NLI4PR
    #PubMedQA
    #Trial_Meta-Analysis_type
    #Evidence_Inference_v2
    HINT
    SemEval_NLI4CT
    TREC_CDS
    TREC_CT
    TREC_Prec-Med
)

INFERENCE_MODES=(
    #0-shot
    #SC
    CoT
)

# Generation Params
BATCH_SIZE=auto
SEED=0

# Output base path
OUTPUT_BASE_PATH=/cfs/home/u021010/PhD/active_dev/outputs
RUN_NAME=all_tests/

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

            VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
                --model $MODEL \
                --model_args $MODEL_ARGS \
                --tasks "${TASK}_${INFERENCE_MODE}" \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --output_path $OUTPUT_PATH \
                --write_out \
                --log_samples 
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
