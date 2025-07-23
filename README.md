# Language Model Evaluation Harness

## We're Refactoring LM-Eval!
(as of 6/15/23)
We have a revamp of the Evaluation Harness library internals staged on the [big-refactor](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) branch! It is far along in progress, but before we start to move the `master` branch of the repository over to this new design with a new version release, we'd like to ensure that it's been tested by outside users and there are no glaring bugs.

We’d like your help to test it out! you can help by:
1. Trying out your current workloads on the big-refactor branch, and seeing if anything breaks or is counterintuitive,
2. Porting tasks supported in the previous version of the harness to the new YAML configuration format. Please check out our [task implementation guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/new_task_guide.md) for more information.

If you choose to port a task not yet completed according to [our checklist](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/tasks/README.md), then you can contribute it by opening a PR containing [Refactor] in the name with:
- A shell command to run the task in the `master` branch, and what the score is
- A shell command to run the task in your PR branch to `big-refactor`, and what the resulting score is, to show that we achieve equality between the two implementations.

Lastly, we'll no longer be accepting new feature requests beyond those that are already open to the master branch as we carry out this switch to the new version over the next week, though we will be accepting bugfixes to `master` branch and PRs to `big-refactor`. Feel free to reach out in the #lm-thunderdome channel of the EAI discord for more information.


## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

**Features:**

- 200+ subtasks / evaluation settings implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Evaluating with publicly available prompts ensures reproducibility and comparability between papers.
- Task versioning to ensure reproducibility when tasks are updated.

## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

We also provide a number of optional dependencies for extended functionality. A detailed table is available at the end of this document.

## Basic Usage

> [!Note]
> When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

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
    --device cuda:0
```

To evaluate models that are loaded via `AutoSeq2SeqLM` in Huggingface, you instead use `hf-seq2seq`. *To evaluate (causal) models across multiple GPUs, use `--model hf-causal-experimental`*

> [!Warning]
> Choosing the wrong model may result in erroneous outputs despite not erroring.


### Neural Magic `deepsparse`

Models from [SparseZoo](https://sparsezoo.neuralmagic.com/) can be evaluated directly in lm-evaluation-harness using [DeepSparse](https://github.com/neuralmagic/deepsparse):
```bash
pip install deepsparse-nightly[llm]
python main.py --model deepsparse --model_args pretrained=zoo:mpt-7b-gsm8k_mpt_pretrain-pruned70_quantized --tasks gsm8k
```

Compatible models hosted on Hugging Face Hub can be used as well:
```bash
python main.py --model deepsparse --model_args pretrained=hf:mgoin/TinyLlama-1.1B-Chat-v0.3-ds --tasks hellaswag
python main.py --model deepsparse --model_args pretrained=hf:neuralmagic/mpt-7b-gsm8k-pruned60-quant-ds --tasks gsm8k
```

### OpenVINO models converted via HuggingFace Optimum
```bash
python main.py --model optimum-causal --model_args pretrained=<model_path_or_name> --task lambada_openai
```

### Commercial APIs

Our library also supports language models served via the OpenAI API:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions \
    --model_args model=davinci-002 \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](https://goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=facebook/opt-125m,base_url=http://{yourip}:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16
```

Note that for externally hosted models, configs such as `--device` which relate to where to place a local model should not be used and do not function. Just like you can use `--model_args` to pass arbitrary arguments to the model constructor for local models, you can use it to pass arbitrary arguments to the model API for hosted models. See the documentation of the hosting service for information on what arguments they support.

| API or Inference Server                                                                                                   | Implemented?                                                                                            | `--model <xxx>` name                                | Models supported:                                                                                                                                                                                                                                                                                                                                          | Request Types:                                                                 |
| --------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| OpenAI Completions                                                                                                        | :heavy_check_mark:                                                                                      | `openai-completions`, `local-completions`           | All OpenAI Completions API models                                                                                                                                                                                                                                                                                                                          | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| OpenAI ChatCompletions                                                                                                    | :heavy_check_mark:                                                                                      | `openai-chat-completions`, `local-chat-completions` | [All ChatCompletions API models](https://platform.openai.com/docs/guides/gpt)                                                                                                                                                                                                                                                                              | `generate_until` (no logprobs)                                                 |
| Anthropic                                                                                                                 | :heavy_check_mark:                                                                                      | `anthropic`                                         | [Supported Anthropic Engines](https://docs.anthropic.com/claude/reference/selecting-a-model)                                                                                                                                                                                                                                                               | `generate_until` (no logprobs)                                                 |
| Anthropic Chat                                                                                                            | :heavy_check_mark:                                                                                      | `anthropic-chat`, `anthropic-chat-completions`      | [Supported Anthropic Engines](https://docs.anthropic.com/claude/docs/models-overview)                                                                                                                                                                                                                                                                      | `generate_until` (no logprobs)                                                 |
| Textsynth                                                                                                                 | :heavy_check_mark:                                                                                      | `textsynth`                                         | [All supported engines](https://textsynth.com/documentation.html#engines)                                                                                                                                                                                                                                                                                  | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| Cohere                                                                                                                    | [:hourglass: - blocked on Cohere API bug](https://github.com/EleutherAI/lm-evaluation-harness/pull/395) | N/A                                                 | [All `cohere.generate()` engines](https://docs.cohere.com/docs/models)                                                                                                                                                                                                                                                                                     | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) (via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)) | :heavy_check_mark:                                                                                      | `gguf`, `ggml`                                      | [All models supported by llama.cpp](https://github.com/ggerganov/llama.cpp)                                                                                                                                                                                                                                                                                | `generate_until`, `loglikelihood`, (perplexity evaluation not yet implemented) |
| vLLM                                                                                                                      | :heavy_check_mark:                                                                                      | `vllm`                                              | [Most HF Causal Language Models](https://docs.vllm.ai/en/latest/models/supported_models.html)                                                                                                                                                                                                                                                              | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| Mamba                                                                                                                     | :heavy_check_mark:                                                                                      | `mamba_ssm`                                         | [Mamba architecture Language Models via the `mamba_ssm` package](https://huggingface.co/state-spaces)                                                                                                                                                                                                                                                      | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| Huggingface Optimum (Causal LMs)                                                                                          | :heavy_check_mark:                                                                                      | `openvino`                                          | Any decoder-only AutoModelForCausalLM converted with Huggingface Optimum into OpenVINO™ Intermediate Representation (IR) format                                                                                                                                                                                                                            | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| Huggingface Optimum-intel IPEX (Causal LMs)                                                                               | :heavy_check_mark:                                                                                      | `ipex`                                              | Any decoder-only AutoModelForCausalLM                                                                                                                                                                                                                                                                                                                      | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| Neuron via AWS Inf2 (Causal LMs)                                                                                          | :heavy_check_mark:                                                                                      | `neuronx`                                           | Any decoder-only AutoModelForCausalLM supported to run on [huggingface-ami image for inferentia2](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2)                                                                                                                                                                                            | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| [Neural Magic DeepSparse](https://github.com/neuralmagic/deepsparse)                                                      | :heavy_check_mark:                                                                                      | `deepsparse`                                        | Any LM from [SparseZoo](https://sparsezoo.neuralmagic.com/) or on [HF Hub with the "deepsparse" tag](https://huggingface.co/models?other=deepsparse)                                                                                                                                                                                                       | `generate_until`, `loglikelihood`                                              |
| [Neural Magic SparseML](https://github.com/neuralmagic/sparseml)                                                          | :heavy_check_mark:                                                                                      | `sparseml`                                          | Any decoder-only AutoModelForCausalLM from [SparseZoo](https://sparsezoo.neuralmagic.com/) or on [HF Hub](https://huggingface.co/neuralmagic). Especially useful for models with quantization like [`zoo:llama2-7b-gsm8k_llama2_pretrain-pruned60_quantized`](https://sparsezoo.neuralmagic.com/models/llama2-7b-gsm8k_llama2_pretrain-pruned60_quantized) | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| NVIDIA NeMo                                                                                                               | :heavy_check_mark:                                                                                      | `nemo_lm`                                           | [All supported models](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/core/core.html#nemo-models)                                                                                                                                                                                                                                     | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |
| Watsonx.ai                                                                                                                | :heavy_check_mark:                                                                                      | `watsonx_llm`                                       | [Supported Watsonx.ai Engines](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx)                                                                                                                                                                                                                                 | `generate_until` `loglikelihood`                                               |
| [Your local inference server!](docs/API_guide.md)                                                                         | :heavy_check_mark:                                                                                      | `local-completions` or `local-chat-completions`     | Support for OpenAI API-compatible servers, with easy customization for other APIs.                                                                                                                                                                                                                                                                         | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                     |

Models which do not supply logits or logprobs can be used with tasks of type `generate_until` only, while local models, or APIs that supply logprobs/logits of their prompts, can be run on all task types: `generate_until`, `loglikelihood`, `loglikelihood_rolling`, and `multiple_choice`.

For more information on the different task `output_types` and model request types, see [our documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md#interface).

> [!Note]
> For best performance with closed chat model APIs such as Anthropic Claude 3 and GPT-4, we recommend carefully looking at a few sample outputs using `--limit 10` first to confirm answer extraction and scoring on generative tasks is performing as expected. providing `system="<some system prompt here>"` within `--model_args` for anthropic-chat-completions, to instruct the model what format to respond in, may be useful.

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

> [!Tip]
> You can inspect what the LM inputs look like by running the following command:

```bash
lm_eval --model openai \
    --model_args engine=davinci-002 \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## Advanced Usage Tips

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

GPTQ quantized models can be loaded by specifying their file names in `,quantized=NAME` (or `,quantized=True` for default names) in the `model_args` argument:

```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=model-name-or-path,quantized=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

GGUF or GGML quantized models can be loaded by using `llama-cpp-python` server:

```bash
python main.py \
    --model gguf \
    --model_args base_url=http://localhost:8000 \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

## Saving & Caching Results

To save evaluation results provide an `--output_path`. We also support logging model responses with the `--log_samples` flag for post-hoc analysis.

> [!TIP]
> Use `--use_cache <DIR>` to cache evaluation results and skip previously evaluated samples when resuming runs of the same (model, task) pairs. Note that caching is rank-dependent, so restart with the same GPU count if interrupted. You can also use --cache_requests to save dataset preprocessing steps for faster evaluation resumption.

To push results and samples to the Hugging Face Hub, first ensure an access token with write access is set in the `HF_TOKEN` environment variable. Then, use the `--hf_hub_log_args` flag to specify the organization, repository name, repository visibility, and whether to push results and samples to the Hub - [example dataset on the  HF Hub](https://huggingface.co/datasets/KonradSzafer/lm-eval-results-demo). For instance:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag \
    --log_samples \
    --output_path results \
    --hf_hub_log_args hub_results_org=EleutherAI,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False \
```

This allows you to easily download the results and samples from the Hub, using:

```python
from datasets import load_dataset

load_dataset("EleutherAI/lm-eval-results-private", "hellaswag", "latest")
```

For a full list of supported arguments, check out the [interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) guide in our documentation!

## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using both Weights & Biases (W&B) and Zeno.

### Zeno

You can use [Zeno](https://zenoml.com) to visualize the results of your eval harness runs.

First, head to [hub.zenoml.com](https://hub.zenoml.com) to create an account and get an API key [on your account page](https://hub.zenoml.com/account).
Add this key as an environment variable:

```bash
export ZENO_API_KEY=[your api key]
```

You'll also need to install the `lm_eval[zeno]` package extra.

To visualize the results, run the eval harness with the `log_samples` and `output_path` flags.
We expect `output_path` to contain multiple folders that represent individual model names.
You can thus run your evaluation on any number of tasks and models and upload all of the results as projects on Zeno.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path output/gpt-j-6B
```

Then, you can upload the resulting data using the `zeno_visualize` script:

```bash
python scripts/zeno_visualize.py \
    --data_path output \
    --project_name "Eleuther Project"
```

This will use all subfolders in `data_path` as different models and upload all tasks within these model folders to Zeno.
If you run the eval harness on multiple tasks, the `project_name` will be used as a prefix and one project will be created per task.

You can find an example of this workflow in [examples/visualize-zeno.ipynb](examples/visualize-zeno.ipynb).

### Weights and Biases

With the [Weights and Biases](https://wandb.ai/site) integration, you can now spend more time extracting deeper insights into your evaluation results. The integration is designed to streamline the process of logging and visualizing experiment results using the Weights & Biases (W&B) platform.

The integration provide functionalities

- to automatically log the evaluation results,
- log the samples as W&B Tables for easy visualization,
- log the `results.json` file as an artifact for version control,
- log the `<task_name>_eval_samples.json` file if the samples are logged,
- generate a comprehensive report for analysis and visualization with all the important metric,
- log task and cli specific configs,
- and more out of the box like the command used to run the evaluation, GPU/CPU counts, timestamp, etc.

First you'll need to install the lm_eval[wandb] package extra. Do `pip install lm_eval[wandb]`.

Authenticate your machine with an your unique W&B token. Visit https://wandb.ai/authorize to get one. Do `wandb login` in your command line terminal.

Run eval harness as usual with a `wandb_args` flag. Use this flag to provide arguments for initializing a wandb run ([wandb.init](https://docs.wandb.ai/ref/python/init)) as comma separated string arguments.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \
    --log_samples
```

In the stdout, you will find the link to the W&B run page as well as link to the generated report. You can find an example of this workflow in [examples/visualize-wandb.ipynb](examples/visualize-wandb.ipynb), and an example of how to integrate it beyond the CLI.

## How to Contribute or Learn More?

For more information on the library and how everything fits together, check out all of our [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)! We plan to post a larger roadmap of desired + planned library improvements soon, with more information on how contributors can help.

### Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/new_task_guide.md).

In general, we follow this priority list for addressing concerns about prompting and other eval details:

1. If there is widespread agreement among people who train LLMs, use the agreed upon procedure.
2. If there is a clear and unambiguous official implementation, use that procedure.
3. If there is widespread agreement among people who evaluate LLMs, use the agreed upon procedure.
4. If there are multiple common implementations but not universal or widespread agreement, use our preferred option among the common implementations. As before, prioritize choosing from among the implementations found in LLM training papers.

These are guidelines and not rules, and can be overruled in special circumstances.

We try to prioritize agreement with the procedures used by other groups to decrease the harm when people inevitably compare runs across different papers despite our discouragement of the practice. Historically, we also prioritized the implementation from [Language Models are Few Shot Learners](https://arxiv.org/abs/2005.14165) as our original goal was specifically to compare results with that paper.

### Support

The best way to get support is to open an issue on this repo or join the [EleutherAI Discord server](https://discord.gg/eleutherai). The `#lm-thunderdome` channel is dedicated to developing this project and the `#release-discussion` channel is for receiving support for our releases. If you've used the library and have had a positive (or negative) experience, we'd love to hear from you!

## Optional Extras

Extras dependencies can be installed via `pip install -e ".[NAME]"`

| Name                 | Use                                                   |
| -------------------- | ----------------------------------------------------- |
| api                  | For using api models (Anthropic, OpenAI API)          |
| audiolm_qwen         | For running Qwen2 audio models                        |
| deepsparse           | For running NM's DeepSparse models                    |
| dev                  | For linting PRs and contributions                     |
| gptq                 | For loading models with AutoGPTQ                      |
| gptqmodel            | For loading models with GPTQModel                     |
| hf_transfer          | For speeding up HF Hub file downloads                 |
| ibm_watsonx_ai       | For using IBM watsonx.ai model apis                   |
| ifeval               | For running the IFEval task                           |
| ipex                 | For running on optimum-intel ipex backend             |
| japanese_leaderboard | For running Japanese LLM Leaderboard tasks            |
| longbench            | For running LongBench tasks                           |
| mamba                | For loading Mamba SSM models                          |
| math                 | For running math task answer checking                 |
| multilingual         | For multilingual tokenizers                           |
| neuronx              | For running on AWS inf2 instances                     |
| optimum              | For running Intel OpenVINO models                     |
| promptsource         | For using PromptSource prompts                        |
| ruler                | For running RULER tasks                               |
| sae_lens             | For using SAELens to steer models                     |
| sentencepiece        | For using the sentencepiece tokenizer                 |
| sparseml             | For using NM's SparseML models                        |
| sparsify             | For using Sparsify to steer models                    |
| testing              | For running library test suite                        |
| vllm                 | For loading models with vLLM                          |
| wandb                | For integration with `Weights and Biases` platform    |
| zeno                 | For visualizing results with Zeno                     |
| -------------------- | ----------------------------------------------------- |
| all                  | Loads all extras (not recommended)                    |

## Cite as

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
