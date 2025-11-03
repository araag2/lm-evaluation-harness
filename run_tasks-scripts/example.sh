#!/bin/bash

# ================================
# Models Configuration
# ================================
MODEL=vllm

TEST_MODEL= "pretrained=unsloth/Qwen3-8B,max_length=25000,gpu_memory_utilization=0.9,dtype=float16,swap_space=8,enable_prefix_caching=True"

TEST_TASK="MedNLI"

INFERENCE_MODE="0-shot"

# Generation Params
CUDA_DEVICES=0 
BATCH_SIZE=auto
SEED=0

# Output base path
OUTPUT_BASE_PATH=../outputs/
RUN_NAME=default_outputs/

echo "=================================================="
echo "[INFO] Running Test evaluation for:\n Model = $MODEL\n Model Args = $TEST_MODEL\n. Output Base Path = $OUTPUT_BASE_PATH"
echo "=================================================="

echo "=================================================="
echo "[INFO] Executing test evaluation for Task: $TEST_TASK with Inference Mode: $INFERENCE_MODE"
echo "=================================================="

OUTPUT_PATH="${OUTPUT_BASE_PATH}/${RUN_NAME}${TEST_TASK}/${INFERENCE_MODE}"

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m lm_eval \
    --model $MODEL \
    --model_args $TEST_MODEL \
    --tasks "${TEST_TASK}_${INFERENCE_MODE}" \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --write_out \
    --limit 10

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Completed: $TASK in Inference Mode: $INFERENCE_MODE"
else
    echo "[ERROR] Failed: $TASK in Inference Mode: $INFERENCE_MODE (exit code $STATUS)"
fi

echo -e "-------------------------------\n"
echo "[INFO] Test Evaluation Completed."
