# LM Evaluation Harness - Optimized Runner Scripts

This directory contains optimized, modular scripts for running evaluations with the LM Evaluation Harness.

## Directory Structure

```
run_tasks-scripts/
├── config/                      # Configuration files
│   ├── models.conf             # Model presets and groups
│   ├── tasks.conf              # Task definitions and groups
│   └── tasks_pairs.conf        # Task pair definitions for multi-turn evaluation
├── lib/                        # Utility functions
│   └── eval_utils.sh          # Shared helper functions
├── examples/                   # Example configurations
│   ├── example_quick_run.conf
│   ├── example_full_trialbench.conf
│   ├── example_multi_turn.conf
│   ├── example_cross_consistency_tiny.conf
│   ├── example_opt125m_small_tasks.conf
│   ├── example_opt125m_0shot_tasks_limited.conf
│   ├── example_opt125m_cot_tasks_limited.conf
│   └── example_opt125m_all_tasks_limited.conf
├── run_eval.sh                # Main unified runner
├── run_multi_turn.sh          # Multi-turn evaluation runner
├── run_cross_consistency.sh   # Cross-consistency evaluation runner
├── run_interactive.sh         # Interactive menu-driven runner
└── README.md                  # This file
```

## Quick Start

### 1. Simple Evaluation

```bash
# Run a single model on a single task
./run_eval.sh --model qwen3-4b --task MedQA --mode 0-shot

# Run multiple models on multiple tasks
./run_eval.sh --model qwen3-4b,llama-8b --task MedQA,MedMCQA --modes "0-shot,CoT"

# Test with limited samples for quick validation
./run_eval.sh --model opt125m --task MedQA --mode 0-shot --limit 10
```

### 2. Using Model and Task Groups

```bash
# Run all 8B models on all TrialBench tasks
./run_eval.sh --model-group 8B --task-group TRIALBENCH --modes "0-shot,CoT,SC"

# Run all medical models on MedQA suite
./run_eval.sh --model-group MEDICAL --task-group MEDQA --mode 0-shot

# Test with tiny model (fast testing)
./run_eval.sh --model-group TINY --task MedQA --mode 0-shot
```

### 2.5 Mode Compatibility Notes

**Important**: Not all tasks support all evaluation modes. Some tasks work better with specific modes:

- **0-shot mode**: Best for classification and simple QA tasks
- **CoT mode**: Required for complex reasoning, summarization, and analysis tasks (e.g., RCT_Summary)
- **SC mode**: Self-consistency voting (requires CoT-compatible tasks)

**Recommended configurations**:
- Use 0-shot for: MedQA, MedMCQA, MedNLI, PubMedQA, TrialBench tasks
- Use CoT for: RCT_Summary, complex analysis tasks, TREC ranking tasks
- Avoid mixing modes when running many tasks to prevent failures

### 3. Using Configuration Files

```bash
# Quick test run
./run_eval.sh --config examples/example_quick_run.conf

# Full TrialBench evaluation
./run_eval.sh --config examples/example_full_trialbench.conf

# OPT-125M testing configurations (fast validation)
./run_eval.sh --config examples/example_opt125m_small_tasks.conf      # Small tasks, low limit
./run_eval.sh --config examples/example_opt125m_0shot_tasks_limited.conf  # 0-shot compatible tasks
./run_eval.sh --config examples/example_opt125m_cot_tasks_limited.conf    # CoT compatible tasks
./run_eval.sh --config examples/example_opt125m_all_tasks_limited.conf    # All tasks (CoT only)

# Dry-run to preview what would execute
./run_eval.sh --config examples/example_full_trialbench.conf --dry-run
```

### 4. Multi-Turn Evaluations

```bash
# Multi-turn CoT with single model
./run_multi_turn.sh --mode multi-turn_CoT --model qwen3-4b --task-pairs MEDQA

# Multi-turn SC-CoT with all 8B models
./run_multi_turn.sh --mode multi-turn_CoT-SC --model-group 8B --task-pairs ALL

# Custom task pair
./run_multi_turn.sh --mode multi-turn_CoT \
    --reasoning-model qwen3-4b \
    --answering-model llama-8b \
    --task-pairs "MedQA:CoT|MedQA:0-shot"
```

### 4. Cross-Consistency Evaluations

Cross-consistency evaluates reasoning chains by having multiple models verify each other's outputs through majority voting.

```bash
# Basic cross-consistency with different models
./run_cross_consistency.sh --reasoning-model qwen3-4b --answering-model llama-8b \\
    --reasoning-task MedQA:CoT --answering-task MedQA:0-shot

# Test with tiny models and limited samples
./run_cross_consistency.sh --model-group TINY --reasoning-task MedQA:CoT \\
    --answering-task MedQA:0-shot --limit 10

# Use configuration file
./run_cross_consistency.sh --config examples/example_cross_consistency_tiny.conf
```

### 5. Interactive Mode

```bash
# Launch interactive menu
./run_interactive.sh
```

The interactive mode guides you through:
1. Selecting evaluation mode
2. Choosing models
3. Selecting tasks or task pairs
4. Configuring runtime parameters
5. Reviewing and confirming before execution

## Available Model Presets

Instead of typing full model arguments, use these presets:

- **4B Models**: `qwen3-4b`, `gemma-4b`
- **8B Models**: `qwen3-8b`, `llama-8b`, `deepseek-8b`, `ministral-8b`
- **Medical Models**: `fleming-7b`, `panacea-7b`

**Model Groups**: `TINY`, `4B`, `8B`, `MEDICAL`, `ALL`

## Available Task Groups

- **TRIALBENCH**: All 6 TrialBench tasks
- **TRIALPANORAMA**: TrialPanorama task
- **MEDQA**: MedQA, MedMCQA, PubMedQA, MedNLI
- **EVIDENCE**: Evidence_Inference_v2, NLI4PR, SemEval_NLI4CT
- **TREC**: TREC_CDS, TREC_Prec_Med, TREC_CT
- **META_ANALYSIS**: Trial meta-analysis tasks
- **SUMMARY**: RCT_Summary, HINT
- **ALL**: All available tasks

## Common Options

### run_eval.sh

```bash
--model MODEL[,MODEL2,...]       # Specify model(s) by preset or full args
--model-group GROUP              # Use model group (4B, 8B, MEDICAL, ALL)
--task TASK[,TASK2,...]          # Specify task(s) by name
--task-group GROUP               # Use task group
--mode MODE[,MODE2,...]          # Inference mode(s): 0-shot, CoT, SC
--output PATH                    # Base output directory
--gpu ID                         # CUDA device ID (default: 0)
--batch-size SIZE                # Batch size (default: auto)
--seed SEED                      # Random seed (default: 0)
--limit NUM                      # Limit number of samples per task
--timestamp                      # Add timestamp to output paths
--dry-run                        # Preview without executing
--verbose                        # Show detailed information
--config FILE                    # Load configuration from file
--help, -h                       # Show help message
```

### run_multi_turn.sh

```bash
--mode MODE                      # multi-turn_CoT, multi-turn_CoT-SC, multi-turn_CoT-MBR
--model MODEL[,MODEL2,...]       # Same model for reasoning and answering
--model-group GROUP              # Use model group
--reasoning-model MODEL          # Separate reasoning model
--answering-model MODEL          # Separate answering model
--task-pairs GROUP               # MEDQA, EVIDENCE, TREC, ALL, or custom pair
--output PATH                    # Base output directory
--gpu ID                         # CUDA device ID
--batch-size SIZE                # Batch size
--seed SEED                      # Random seed
--timestamp                      # Add timestamp to output paths
--dry-run                        # Preview without executing
--help, -h                       # Show help message
```

## Creating Custom Configurations

Create a `.conf` file with your settings:

```bash
# my_evaluation.conf

# Select specific models
MODELS=(
    "$MODEL_QWEN3_4B"
    "$MODEL_LLAMA_8B"
)

# Select tasks
TASKS=(
    "MedQA"
    "MedMCQA"
)

# Inference modes
MODES=("0-shot" "CoT")

# Runtime settings
OUTPUT_BASE="./outputs/my_experiment"
CUDA_DEVICES="0"
BATCH_SIZE="8"
SEED="42"
USE_TIMESTAMP=true
VERBOSE=true
```

Then run:

```bash
./run_eval.sh --config my_evaluation.conf
```

## Advanced Examples

### Example 1: Test Single Model on All Tasks

```bash
./run_eval.sh \
    --model qwen3-4b \
    --task-group ALL \
    --mode 0-shot \
    --output ./outputs/qwen_full_benchmark \
    --timestamp
```

### Example 2: Compare All Models on Single Task

```bash
./run_eval.sh \
    --model-group ALL \
    --task MedQA \
    --modes "0-shot,CoT,SC" \
    --output ./outputs/medqa_model_comparison \
    --timestamp
```

### Example 3: Dry-Run Before Executing

```bash
# Preview what would run
./run_eval.sh \
    --model-group 8B \
    --task-group TRIALBENCH \
    --modes "0-shot,CoT,SC" \
    --dry-run

# If satisfied, execute without --dry-run
./run_eval.sh \
    --model-group 8B \
    --task-group TRIALBENCH \
    --modes "0-shot,CoT,SC"
```

### Example 4: Custom GPU and Batch Settings

```bash
./run_eval.sh \
    --model qwen3-8b \
    --task-group MEDQA \
    --mode CoT \
    --gpu 1 \
    --batch-size 16 \
    --seed 42
```

### Example 5: Multi-Turn with Different Models

```bash
./run_multi_turn.sh \
    --mode multi-turn_CoT \
    --reasoning-model qwen3-4b \
    --answering-model llama-8b \
    --task-pairs "MedQA:CoT|MedQA:0-shot,MedMCQA:CoT|MedMCQA:0-shot"
```

## Features

### 1. Centralized Configuration
- Model and task definitions in separate config files
- Easy to add new models or tasks without editing scripts
- Reusable presets across different evaluations

### 2. Command-Line Interface
- No need to edit scripts for each run
- Support for both individual items and groups
- Dry-run mode to preview before execution

### 3. Helper Functions
- Shared utilities in `lib/eval_utils.sh`
- Colored logging for better visibility
- Progress tracking and time estimation
- Automatic error handling and reporting

### 4. Flexible Output Management
- Organized output directory structure
- Optional timestamps for versioning
- Automatic directory creation

### 5. Interactive Mode
- Menu-driven interface for easy use
- Step-by-step configuration
- Summary and confirmation before execution

## Output Structure

```
outputs/
└── {output_base}/
    └── {task_name}/
        └── {inference_mode}/
            └── {model_name}/
                └── [timestamp]/  # Optional
                    ├── results.json
                    └── samples/
```

## Tips and Best Practices

1. **Start with Dry-Run**: Use `--dry-run` to preview your evaluation plan
2. **Use Timestamps**: Add `--timestamp` for experiment versioning
3. **Create Configs**: Save frequently-used settings in config files
4. **Monitor GPU**: Check GPU availability before large runs
5. **Batch Sizes**: Use `auto` for automatic optimization, or set manually for control
6. **Incremental Testing**: Test with one model/task first, then scale up

## Troubleshooting

**Issue**: Model preset not found  
**Solution**: Check `config/models.conf` for available presets or use full model args

**Issue**: Task group not found  
**Solution**: Check `config/tasks.conf` for available task groups

**Issue**: GPU memory error  
**Solution**: Reduce `gpu_memory_utilization` in model config or use smaller batch size

**Issue**: Permission denied  
**Solution**: Make scripts executable: `chmod +x run_eval.sh run_multi_turn.sh run_interactive.sh`

## Migration from Old Scripts

Your original scripts are preserved. To migrate:

1. **Identify your common model/task combinations**
2. **Create a config file** with these settings
3. **Run with new scripts** using `--config`
4. **Compare outputs** to ensure consistency
5. **Adopt new scripts** once validated

## Contributing

To add new models or tasks:

1. Edit `config/models.conf` to add model presets
2. Edit `config/tasks.conf` to add task definitions
3. No changes needed to runner scripts!

## Support

For questions or issues:
- Check `--help` for detailed usage information
- Review example configs in `examples/`
- Try interactive mode for guided setup
