#!/bin/bash
# Model Configuration
MODEL=vllm
MODEL_ARGS="pretrained=google/gemma-3n-E4B-it,max_model_len=40000,do_sample=True,temperature=0.7,top_p=0.8,top_k=20"

# Tasks List (space-separated)
TASK_LIST=(
    MedMCQA
    MedNLI
    MedQA
    NLI4PR
    PubMedQA
    Trial_Meta-Analysis_type
    Evidence_Inference_v2
    HINT
    SemEval_NLI4CT
    TREC_CDS
    TREC_CT
    TREC_Prec-Med
)

INFERENCE_MODES=(
    0-shot
    SC
    CoT
)

# Generation Params
BATCH_SIZE=auto
SEED=0

# Output base path
OUTPUT_BASE_PATH=/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/outputs
RUN_NAME=/Run_2/

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

        CUDA_VISIBLE_DEVICES=$1 python -m lm_eval \
            --model $MODEL \
            --model_args $MODEL_ARGS \
            --tasks "${TASK}_${INFERENCE_MODE}" \
            --batch_size $BATCH_SIZE \
            --seed $SEED \
            --output_path $OUTPUT_PATH \
            --write_out \
            --log_samples \
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

echo "[INFO] All evaluations completed."
