#!/bin/bash
OUTPUT_PATH=../lm_harness_run-outputs/outputs/
TASK=SemEval_NLI4CT_2023
SPLIT=test

echo -e "-------------------------------\n"
echo -e "Running Write-Out Script for:\n Task = $TASK\n Split = $SPLIT\n Output Path = $OUTPUT_PATH\n"

CUDA_VISIBLE_DEVICES=$1 python -m scripts.write_out \
    --output_base_path $OUTPUT_PATH\
    --tasks $TASK \
    --sets $SPLIT \
    --num_examples 2 \
    --num_fewshot 0

echo "Done!"
echo -e "-------------------------------\n"