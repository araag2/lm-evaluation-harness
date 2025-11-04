# Trial Meta Analysis — Automatically Extracting Numerical Results from Randomized Controlled Trials with LLMs

Huggingface Dataset Link: [Trial Meta Analysis](https://huggingface.co/datasets/araag2/Trial_Meta-Analysis)

## Paper

Title: `Automatically Extracting Numerical Results from Randomized Controlled Trials with Large Language Models`

Paper Link: [arXiv](https://arxiv.org/abs/2405.01686)

Abstract:

`Meta-analyses statistically aggregate the findings of different randomized controlled trials (RCTs) to assess treatment effectiveness. Because this yields robust estimates of treatment effectiveness, results from meta-analyses are considered the strongest form of evidence. However, rigorous evidence syntheses are time-consuming and labor-intensive, requiring manual extraction of data from individual trials to be synthesized. Ideally, language technologies would permit fully automatic meta-analysis, on demand. This requires accurately extracting numerical results from individual trials, which has been beyond the capabilities of natural language processing (NLP) models to date. In this work, we evaluate whether modern large language models (LLMs) can reliably perform this task. We annotate (and release) a modest but granular evaluation dataset of clinical trial reports with numerical findings attached to interventions, comparators, and outcomes. Using this dataset, we evaluate the performance of seven LLMs applied zero-shot for the task of conditionally extracting numerical findings from trial reports. We find that massive LLMs that can accommodate lengthy inputs are tantalizingly close to realizing fully automatic meta-analysis, especially for dichotomous (binary) outcomes (e.g., mortality). However, LLMs -- including ones trained on biomedical texts -- perform poorly when the outcome measures are complex and tallying the results requires inference. This work charts a path toward fully automatic meta-analysis of RCTs via LLMs, while also highlighting the limitations of existing models for this aim.`

Homepage: [GitHub](https://github.com/jayded/llm-meta-analysis)

### Citation

```bibtex
@article{yun2024automatically,
  title={Automatically extracting numerical results from randomized controlled trials with large language models},
  author={Yun, Hye Sun and Pogrebitskiy, David and Marshall, Iain J and Wallace, Byron C},
  journal={arXiv preprint arXiv:2405.01686},
  year={2024}
}
```

### Groups, Tags, and Tasks

#### Groups

* `Trial_Meta_Analysis_type`: `Evaluates models on the Trial Meta Analysis dataset, focusing on distinguishing between binary or continuous outcomes.`
* `Trial_Meta_Analysis_binary`: `Evaluates models on the ability to extract numerical results for binary outcomes from the Trial Meta Analysis dataset.`
* `Trial_Meta_Analysis_continuous`: `Evaluates models on the ability to extract numerical results for continuous outcomes from the Trial Meta Analysis dataset.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `Trial_Meta_Analysis_type_0-shot`: `Trial Meta Analysis type classification in single-turn 0-shot setting`
* `Trial_Meta_Analysis_type_CoT`: `Trial Meta Analysis type classification in single-turn chain-of-thought 0-shot setting`
* `Trial_Meta_Analysis_type_SC`: `Trial Meta Analysis type classification in single-turn self-consistency setting`
* `Trial_Meta_Analysis_binary`: `Trial Meta Analysis binary outcome extraction in single-turn chain-of-thought 0-shot setting`
* `Trial_Meta_Analysis_continuous`: `Trial Meta Analysis continuous outcome extraction in single-turn chain-of-thought setting`

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
