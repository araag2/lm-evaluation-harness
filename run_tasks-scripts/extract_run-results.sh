#!/bin/bash

OUTPUT_PATH=/cfs/home/u021010/PhD/active_dev/outputs/template_outputs/
OUTPUT_NAME=meeting_summary

INPUT_FOLDERS_BASE=/cfs/home/u021010/PhD/active_dev/outputs/multi-turn_CoT

INPUT_FOLDERS=(
  "."
)

echo -e "-------------------------------\n"
echo -e "Running Extraction Script with:\n"
echo -e " ➤ Output Path  = $OUTPUT_PATH"
echo -e " ➤ Output Name  = $OUTPUT_NAME"
echo -e " ➤ Input Folders = ${INPUT_FOLDERS[*]}"

INPUT_ARGS=()
for folder in "${INPUT_FOLDERS[@]}"; do
  full_path="${INPUT_FOLDERS_BASE}/${folder}"
  INPUT_ARGS+=("$full_path")
  echo -e "   ↪ Including folder: $full_path"
done

echo -e "\n-------------------------------"
echo -e "Executing Python extraction script...\n"

python scripts/extract_run-results.py \
  --output_dir "$OUTPUT_PATH" \
  --output_name "$OUTPUT_NAME" \
  --input_folders "${INPUT_ARGS[@]}"

echo -e "\n✅ Done extracting results!"
echo -e "-------------------------------\n"
