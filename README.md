# OpenCTEval 

## Announcement

---

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

We also provide a number of optional dependencies for extended functionality. A detailed table is available at the end of this document.

## Basic Usage

### User Guide

A user guide detailing the full list of supported arguments is provided [here](./docs/interface.md), and on the terminal by calling `lm_eval -h`. Alternatively, you can use `lm-eval` instead of `lm_eval`.

A list of supported tasks (or groupings of tasks) can be viewed with `lm-eval --tasks list`. Task descriptions and links to corresponding subfolders are provided [here](./lm_eval/tasks/README.md).

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command (this assumes you are using a CUDA-compatible GPU):

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Models that are loaded via both `transformers.AutoModelForCausalLM` (autoregressive, decoder-only GPT style models) and `transformers.AutoModelForSeq2SeqLM` (such as encoder-decoder models like T5) in Huggingface are supported.

Batch size selection can be automated by setting the  ```--batch_size``` flag to ```auto```. This will perform automatic detection of the largest batch size that will fit on your device. On tasks where there is a large difference between the longest and shortest example, it can be helpful to periodically recompute the largest batch size, to gain a further speedup. To do this, append ```:N``` to above flag to automatically recompute the largest batch size ```N``` times. For example, to recompute the batch size 4 times, the command would be:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4
```

> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lm_eval` via `--model_args pretrained=/path/to/model`

## Benchmark Composition

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


## Citation

### Our benchmark paper

Coming soon!

### Original lm-evaluation-harness paper

```text
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```
