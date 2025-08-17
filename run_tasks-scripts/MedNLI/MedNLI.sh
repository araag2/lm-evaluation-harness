#!/bin/bash
MODEL=vllm
MODEL_ARGS="pretrained=Qwen/Qwen3-8B,max_length=20000"
BATCH_SIZE=auto
SEED=0

DEFAULT_DATASET="MedNLI"

DEFAULT_TASK_LIST=(
    #0-shot
    #SC
    CoT
)

DATASET=${1:-$DEFAULT_DATASET}     # first arg = dataset, fallback to default
shift                              # move to next args (task list)
if [ $# -gt 0 ]; then
    TASK_LIST=("$@")               # remaining args = task list
else
    TASK_LIST=("${DEFAULT_TASK_LIST[@]}")
fi

OUTPUT_BASE_PATH="${BASE_OUTPUT_DIR}/${DATASET}"

echo "=================================================="
echo "[INFO] Running evaluations:"
echo "Dataset = $DATASET"
echo "Model = $MODEL"
echo "Model Args = $MODEL_ARGS"
echo "Output Base Path = $OUTPUT_BASE_PATH"
echo "Tasks = ${TASK_LIST[*]}"
echo "=================================================="

for TASK in "${TASK_LIST[@]}"; do
    OUTPUT_PATH="${OUTPUT_BASE_PATH}/${TASK}"

    echo "--------------------------------------------------"
    echo "[INFO] Running Task: $TASK"
    echo "--------------------------------------------------"

    python -m lm_eval \
        --model $MODEL \
        --model_args $MODEL_ARGS \
        --tasks "$DATASET_$TASK" \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_path $OUTPUT_PATH \
        --write_out \
        --log_samples \
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