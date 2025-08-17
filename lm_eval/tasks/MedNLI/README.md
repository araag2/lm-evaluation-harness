# Task-name

## Paper

Title: `Lessons from Natural Language Inference in the Clinical Domain`

Paper Link: [arXiv](https://arxiv.org/abs/1808.06752)

Abstract:

State of the art models using deep neural networks have become very good in learning an accurate mapping from inputs to outputs. However, they still lack generalization capabilities in conditions that differ from the ones encountered during training. This is even more challenging in specialized, and knowledge intensive domains, where training data is limited. To address this gap, we introduce MedNLI - a dataset annotated by doctors, performing a natural language inference task (NLI), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. SNLI) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.

Homepage: [GitHub Pages](https://jgc128.github.io/mednli/)

### Citation

```bibtex
@article{romanov2018lessons,
	title = {Lessons from Natural Language Inference in the Clinical Domain},
	url = {http://arxiv.org/abs/1808.06752},
	abstract = {State of the art models using deep neural networks have become very good in learning an accurate mapping from inputs to outputs. However, they still lack generalization capabilities in conditions that differ from the ones encountered during training. This is even more challenging in specialized, and knowledge intensive domains, where training data is limited. To address this gap, we introduce {MedNLI} - a dataset annotated by doctors, performing a natural language inference task ({NLI}), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. {SNLI}) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.},
	journaltitle = {arXiv:1808.06752 [cs]},
	author = {Romanov, Alexey and Shivade, Chaitanya},
	urldate = {2018-08-27},
	date = {2018-08-21},
	eprinttype = {arxiv},
	eprint = {1808.06752},
}
```

### Groups, Tags, and Tasks

#### Groups

* `MedNLI`: `Evaluates models on the MedNLI dataset, which is a natural language inference task in the medical domain.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `few-shot`: `Evaluates models in few-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `MedNLI_0-shot` : `MedNLI in base 0-shot setting`
* `MedNLI_0-shot_SC` : `MedNLI 0-shot with self-consistency sampling`

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
