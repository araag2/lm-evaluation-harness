#!/bin/bash
# Model Configuration
MODEL=hf
MODEL_ARGS="pretrained=Qwen/Qwen3-8B,max_length=20000,do_sample=True,temperature=0.7,top_p=0.8,top_k=20"

# Tasks List (space-separated)
TASK_LIST=(
    MedMCQA_0-shot
    MedNLI_0-shot
    MedQA_0-shot
    NLI4PR_0-shot
    PubMedQA_0-shot
    SemEval_NLI4CT_0-shot
    TREC_CDS_0-shot
    TREC_CT_0-shot
    TREC_Prec-Med_0-shot
    Trial_Meta-Analysis_type_0-shot
    Evidence_Inference_v2_0-shot
    HINT_0-shot
)

# Generation Params
BATCH_SIZE=auto
SEED=0

# Output base path
OUTPUT_BASE_PATH=/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/outputs

echo "=================================================="
echo "[INFO] Running evaluations for:\n Model = $MODEL\n Model Args = $MODEL_ARGS\n. Output Base Path = $OUTPUT_BASE_PATH"
echo "=================================================="

echo -e "=================================================="
echo "[INFO] Starting batch execution of ${TASK_LIST[@]} task scripts."
echo "=================================================="

# Loop over tasks
for TASK in "${TASK_LIST[@]}"; do
  OUTPUT_PATH="${OUTPUT_BASE_PATH}/${TASK%%_0-shot}/0-shot"

  echo -e "--------------------------------------------------"
  echo -e "[INFO] Running Task: $TASK"
  echo -e "--------------------------------------------------"

  CUDA_VISIBLE_DEVICES=$1 python -m lm_eval \
    --model $MODEL \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_path $OUTPUT_PATH \
    --apply_chat_template \
    --write_out
    #--limit 2 \
    #--log_samples \
    #--predict_only \

  STATUS=$?
  if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Completed: $TASK"
  else
    echo "[ERROR] Failed: $TASK (exit code $STATUS)"
  fi

  echo -e "-------------------------------\n"
done

echo "[INFO] All evaluations completed."
