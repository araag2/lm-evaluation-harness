#!/bin/bash

OUTPUT_PATH=/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/error_breakdown
OUTPUT_NAME=error_breakdown

INPUT_FOLDER=/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/multi-turn_CoT/

echo -e "-------------------------------\n"
echo -e "Running Error Extraction Script with:\n"
echo -e " ➤ Output Path  = $OUTPUT_PATH"
echo -e " ➤ Output Name  = $OUTPUT_NAME"
echo -e " ➤ Input Folder = $INPUT_FOLDER"
echo -e "\n-------------------------------"

echo -e "Executing error extraction script...\n"

python scripts/extract_error-breakdown.py \
  --input_folder "$INPUT_FOLDER" \
  --output_dir "$OUTPUT_PATH" \
  --output_name "$OUTPUT_NAME" \

echo -e "\n✅ Done extracting error breakdown!"
echo -e "-------------------------------\n"
