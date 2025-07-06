# Task-name

## Paper

Title: `NLI4CT: Multi-Evidence Natural Language Inference for Clinical Trial Reports` and `SemEval-2024 Task 2: Safe Biomedical Natural Language Inference for Clinical Trials`

Paper Link: [2023 arXiv](https://arxiv.org/abs/2305.03598) and [2024 arXiv](https://arxiv.org/abs/2404.04963)

Abstract:

`How can we interpret and retrieve medical evidence to support clinical decisions? Clinical trial reports (CTR) amassed over the years contain indispensable information for the development of personalized medicine. However, it is practically infeasible to manually inspect over 400,000+ clinical trial reports in order to find the best evidence for experimental treatments. Natural Language Inference (NLI) offers a potential solution to this problem, by allowing the scalable computation of textual entailment. However, existing NLI models perform poorly on biomedical corpora, and previously published datasets fail to capture the full complexity of inference over CTRs. In this work, we present a novel resource to advance research on NLI for reasoning on CTRs. The resource includes two main tasks. Firstly, to determine the inference relation between a natural language statement, and a CTR. Secondly, to retrieve supporting facts to justify the predicted relation. We provide NLI4CT, a corpus of 2400 statements and CTRs, annotated for these tasks. Baselines on this corpus expose the limitations of existing NLI approaches, with 6 state-of-the-art NLI models achieving a maximum F1 score of 0.627. To the best of our knowledge, we are the first to design a task that covers the interpretation of full CTRs. To encourage further work on this challenging dataset, we make the corpus, competition leaderboard, and website, available on CodaLab, and code to replicate the baseline experiments on GitHub.`

Homepage: [GitHub Pages](https://sites.google.com/view/nli4ct/)


### Citation

```bibtex
@article{jullien2023nli4ct,
  title={NLI4CT: Multi-evidence natural language inference for clinical trial reports},
  author={Jullien, Mael and Valentino, Marco and Frost, Hannah and O'Regan, Paul and Landers, D{\'o}nal and Freitas, Andre},
  journal={arXiv preprint arXiv:2305.03598},
  year={2023}
}

@article{jullien2024semeval,
  title={SemEval-2024 task 2: Safe biomedical natural language inference for clinical trials},
  author={Jullien, Ma{\"e}l and Valentino, Marco and Freitas, Andr{\'e}},
  journal={arXiv preprint arXiv:2404.04963},
  year={2024}
}
```

### Groups, Tags, and Tasks

#### Groups

* `SemEval_NLI4CT`: `The SemEval NLI4CT task, which includes the 2023 and 2024 tasks on natural language inference for clinical trials, focusing on reasoning over clinical trial reports and statements about them.`

#### Tags

#### Tasks

* `SemEval_NLI4CT_2023` : `The 2023 SemEval NLI4CT task`
* `SemEval_NLI4CT_2024` : `The 2024 SemEval NLI4CT task`

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
