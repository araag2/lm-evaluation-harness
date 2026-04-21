# Consort_QA - Evaluation of Clinical Trials Reporting Quality using Large Language Models

Huggingface Dataset Link: [Consort_QA](https://huggingface.co/datasets/araag2/Consort_QA)

## Paper

Title: `Evaluation of Clinical Trials Reporting Quality using Large Language Models`

Paper Link: [arXiv](https://arxiv.org/abs/2510.04338)

Abstract:

`Reporting quality is an important topic in clinical trial research articles, as it can impact clinical decisions. In this article, we test the ability of large language models to assess the reporting quality of this type of article using the Consolidated Standards of Reporting Trials (CONSORT). We create CONSORT-QA, an evaluation corpus from two studies on abstract reporting quality with CONSORT-abstract standards. We then evaluate the ability of different large generative language models (from the general domain or adapted to the biomedical domain) to correctly assess CONSORT criteria with different known prompting methods, including Chain-of-thought. Our best combination of model and prompting method achieves 85% accuracy. Using Chain-of-thought adds valuable information on the model's reasoning for completing the task.`

Homepage: [GitHub Pages](https://Consort_QA.github.io/)

### Citation

```bibtex
  @article{lai2025evaluation,
    title={Evaluation of Clinical Trials Reporting Quality using Large Language Models},
    author={La{\"\i}-king, Mathieu and Paroubek, Patrick},
    journal={arXiv:2510.04338},
    year={2025}
  }
```

### Groups, Tags, and Tasks

#### Groups

* `Consort_QA`: `Evaluates models on the Consort_QA dataset, which is a question answering task in the biomedical domain.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `Consort_QA_0-shot` : `Consort_QA in single-turn 0-shot setting`
* `Consort_QA_CoT`    : `Consort_QA in single-turn chain-of-thought 0-shot setting`
* `Consort_QA_SC`     : `Consort_QA in single-turn self-consistency setting`
* `Consort_QA_CoT_SC` : `Consort_QA in single-turn chain-of-thought with self-consistency setting`

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
