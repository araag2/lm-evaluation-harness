# EBM-NLP

## Paper

Title: `A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature`

Paper Link: [arXiv](https://arxiv.org/abs/1806.04185)

Abstract:

`We present a corpus of 5,000 richly annotated abstracts of medical articles describing clinical randomized controlled trials. Annotations include demarcations of text spans that describe the Patient population enrolled, the Interventions studied and to what they were Compared, and the Outcomes measured (the "PICO" elements). These spans are further annotated at a more granular level, e.g., individual interventions within them are marked and mapped onto a structured medical vocabulary. We acquired annotations from a diverse set of workers with varying levels of expertise and cost. We describe our data collection process and the corpus itself in detail. We then outline a set of challenging NLP tasks that would aid searching of the medical literature and the practice of evidence-based medicine.`

Homepage: [GitHub](https://github.com/bepnye/EBM-NLP)


### Citation

```bibtex
@inproceedings{nye2018corpus,
  title={A corpus with multi-level annotations of patients, interventions and outcomes to support language processing for medical literature},
  author={Nye, Benjamin and Li, Junyi Jessy and Patel, Roma and Yang, Yinfei and Marshall, Iain and Nenkova, Ani and Wallace, Byron C},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={197--207},
  year={2018}
}

```

### Groups, Tags, and Tasks

#### Groups

* `EBM-NLP`: `Evaluates models on the RCT Summary dataset, which involves generating narrative summaries of randomized controlled trials.`

#### Tags
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `EBM-NLP_CoT` : `RCT Summary in single-turn chain-of-thought setting`
* `EBM-NLP_SC` : `RCT Summary in single-turn self-consistency setting`

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