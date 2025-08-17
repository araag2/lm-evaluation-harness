#!/bin/bash
MODEL=vllm
MODEL_ARGS="pretrained=Qwen/Qwen3-8B,max_length=20000"
BATCH_SIZE=auto
SEED=0

BASE_OUTPUT_DIR="/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/outputs"
DEFAULT_DATASET="MedNLI"

# Available Datasets: Evidence_Inference_v2, HINT, MedMCQA, MedNLI, MedQA, NLI4PR, PubMedQA, SemEval_NLI4CT, TREC_CDS, TREC_CT, TREC_Prec-Med, Trial_Meta-Analysis_type

DEFAULT_TASK_LIST=( # Can use comma-separated list on command prompt
    0-shot
    SC
    CoT
)

DATASET=${1:-$DEFAULT_DATASET}     # first arg = dataset, fallback to default
shift                              # move to next args (task list)

if [ $# -gt 0 ]; then
    if [[ "$1" == *,* ]]; then # If comma-separated string provided, split into array
        IFS=',' read -ra TASK_LIST <<< "$1"
    else
        TASK_LIST=("$@")
    fi
else
    TASK_LIST=("${DEFAULT_TASK_LIST[@]}")
fi

OUTPUT_BASE_PATH="${BASE_OUTPUT_DIR}/${DATASET}"

echo "=================================================="
echo "[INFO] Running evaluations:"
echo "Dataset = $DATASET"
echo "Model = $MODEL"
echo "Model Args = $MODEL_ARGS"
echo "Output Base Path = $OUTPUT_BASE_PATH/"
echo "Tasks = ${TASK_LIST[*]}"
echo "=================================================="

for TASK in "${TASK_LIST[@]}"; do
    OUTPUT_PATH="${OUTPUT_BASE_PATH}/${TASK}"

    DATASET_AND_TASK="${DATASET}_${TASK}"

    echo "--------------------------------------------------"
    echo "[INFO] Running Task: $DATASET_AND_TASK"
    echo "--------------------------------------------------"

    python -m lm_eval \
        --model $MODEL \
        --model_args $MODEL_ARGS \
        --tasks $DATASET_AND_TASK \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_path $OUTPUT_PATH \
        --write_out \
        --log_samples
        #--apply_chat_template \
        #--limit 10 \
        #--predict_only \

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Completed: $TASK"
    else
        echo "[ERROR] Failed: $TASK (exit code $STATUS)"
    fi

    echo -e "-------------------------------\n"
done

echo "[INFO] All Tasks completed."