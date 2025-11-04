# OpenCTEval 

## Announcement

TO:DO - Update this section

---

## Overview

This project implements `OpenCTEval`, a comprehensive benchmark for evaluating generative language models on clinical trial data understanding and reasoning tasks. It is a forked project from the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository, which provides a simple unified framework to evaluate on a wide range of tasks.

## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone --depth 1 https://github.com/araag2/lm-evaluation-harness/tree/OpenCTEval
cd lm-evaluation-harness
pip install -e .
```

## Repo Structure

## Basic Usage

TO:DO - Update this section

### User Guide

A user guide detailing the full list of supported arguments is provided [here](./docs/original_tutorials/interface.md), and on the terminal by calling `lm_eval -h`. Alternatively, you can use `lm-eval` instead of `lm_eval`.

A list of supported tasks (or groupings of tasks) can be viewed with `lm-eval --tasks list`. Task descriptions and links to corresponding subfolders are provided [here](./lm_eval/tasks/README.md).

### Run Task Scripts

You can run evaluation scripts for all benchmark tasks using the bash scripts located in the [run scripts folder](./run_task-scripts/). These scripts have interchangeable model arguments, so you can easily evaluate different models by changing the model-related flags, and different tasks by changing the task-related flags. 

If you want to run a quick example, you can run the [example script](./run_task-scripts/run_example.sh) which evaluates Qwen-8B on 10 samples of the `MedNLI` task in `0-shot` setting:

```bash ./run_task-scripts/example.sh```

Other available files in the `run_task-scripts` folder include:

## Repository Structure

    .
    ├── run_task_scripts/       # Base folder for all bash scripts to run experiments
        ├── example.sh          # Example script to run a quick evaluation

        ├── run_multi_tasks.sh  # Script to run multiple tasks sequentially, in 0-shot and single-turn CoT settings
        ├── 
    └── README.md




### Manually Run Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. Qwen-8B) on `MedNLI` you can use the following command (this assumes you are using a CUDA-compatible GPU):

```bash
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-8B \
    --tasks MedNLI_0-shot \
    --device cuda:0 \
    --batch_size 8
```

> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lm_eval` via `--model_args pretrained=/path/to/model`

## OpenCTEval Benchmark Composition

(MCQ: Multiple-Choice Question, ENC: Entailment, Neutral, Contradiction)

| Task Name                                                      | Short Description of Task Types and Outputs         |
| :------------------------------------------------------------- | :-------------------------------------------------- |
| [MedNLI](lm_eval/tasks/MedNLI/)                                | **NLI** (ENC)                                       |
| [MedQA](lm_eval/tasks/MedQA/)                                  | **MCQ**                                             |
| [MedMCQA](lm_eval/tasks/MedMCQA/)                              | **MCQ**                                             |
| [PubMedQA](lm_eval/tasks/PubMedQA/)                            | **NLI** (ENC)                                       |
| [RCT Summary](lm_eval/tasks/RCT_Summary/)                      | **Summary** (Results Section)                       |
| [Evidence Inference 2.0](lm_eval/tasks/Evidence_Inference_v2/) | **NLI** (ENC)                                       |
| [NLI4PR](lm_eval/tasks/NLI4PR/)                                | **NLI** (ENC)                                       |
| [HINT](lm_eval/tasks/HINT/)                                    | **NLI** (EC)                                        |
| [Trial Meta Analysis](lm_eval/tasks/Trial_Meta_Analysis/)      | **NLI** (Outcome Type), **IE** (Extracting Results) |
| [TREC CDS Track](lm_eval/tasks/TREC_CDS/)                      | **NLI** (ENC), **Relevance Ranking**                |
| [TREC Prec-Med Track](lm_eval/tasks/TREC_Prec_Med/)            | **NLI** (ENC), **Relevance Ranking**                |
| [TREC CT Track](lm_eval/tasks/TREC_CT/)                        | **NLI** (ENC), **Relevance Ranking**                |
| [SemEval NLI4CT](lm_eval/tasks/SemEval_NLI4CT/)                | **NLI** (EC)                                        |

## OpenCTEval Results

The results of evaluating various models on the OpenCTEval benchmark can be found in the [results folder](./outputs/). Detailed analysis and visualizations will be provided in upcoming publications.

## Citation

### Our benchmark paper

Coming soon!

### Original lm-evaluation-harness paper

```text
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```
