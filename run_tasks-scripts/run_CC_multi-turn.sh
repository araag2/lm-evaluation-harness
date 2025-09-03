VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 python -m lm_eval.multi-turn_cross-consistency \
  --provider vllm \
  --mode multi-turn \
  --reasoning_models pretrained=Qwen/Qwen3-0.6B \
  --answering_models pretrained=Qwen/Qwen3-0.6B \
  --reasoning_tasks MedNLI:CoT \
  --answering_tasks MedNLI:0-shot \
  --output_path /cfs/home/u021010/PhD/active_dev/outputs/CoT-Debug \
  --limit 2