#!/bin/bash
MODEL=hf
MODEL_ARGS="pretrained=Qwen/Qwen3-8B,max_length=20000,do_sample=True,temperature=0.7,top_p=0.8,top_k=20" # "pretrained=Qwen/Qwen3-8B,max_num_seqs=1,enable_chunked_prefill=True"
#enable_thinking=True,Temperature=0.6,TopP=0.95,TopK=20,MinP=0
TASKS=MedNLI_0-shot

# Generation Params
BATCH_SIZE=auto # auto
SEED=0

#Ouput Dir
OUTPUT_PATH=/user/home/aguimas/data/PhD/Active_Dev/lm_harness_run-outputs/outputs/MedNLI/0-shot/

echo -e "-------------------------------\n"
echo -e "Running MedNLI_0-shot eval for:\n Model = $MODEL\n TASKS = $TASKS\n Output PATH = $OUTPUT_PATH\n"

python -m lm_eval\
    --model $MODEL\
    --model_args $MODEL_ARGS \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_path $OUTPUT_PATH \
    --write_out \
    --log_samples \
    #--limit 10 \
    #--apply_chat_template \
    #--predict_only \

echo "Done with $TASKS"
echo -e "-------------------------------\n"