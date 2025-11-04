# MedQA-USMLE — A Large-scale Open Domain Question Answering Dataset from Medical Exams

Huggingface Dataset Link: [MedQA](https://huggingface.co/datasets/araag2/MedQA)

## Paper

Title: `What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams`

Paper Link: [arXiv](https://arxiv.org/abs/2009.13081)

Abstract:

`MedQA is a large-scale multiple-choice question-answering dataset designed to mimic the style of professional medical board exams, particularly the USMLE (United States Medical Licensing Examination). Introduced by Jin et al. in 2020 under the title “What Disease Does This Patient Have? A Large‑scale Open‑Domain Question Answering Dataset from Medical Exams”, the dataset supports open-domain QA via retrieval from medical textbooks`

Homepage: [GitHub](https://github.com/jind11/MedQA)

### Citation

```bibtex
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```

### Groups, Tags, and Tasks

#### Groups

* `MedQA`: `Evaluates models on the MedQA dataset, which is a question answering task in the medical domain.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `MedQA_0-shot` : `MedQA in single-turn 0-shot setting`
* `MedQA_CoT`    : `MedQA in single-turn chain-of-thought 0-shot setting`
* `MedQA_SC`     : `MedQA in single-turn self-consistency setting`
* `MedQA_CoT_SC` : `MedQA in single-turn chain-of-thought with self-consistency setting`

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
