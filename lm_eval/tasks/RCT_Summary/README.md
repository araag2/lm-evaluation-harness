# Task-name

## Paper

Title: `Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization`

Paper Link: [arXiv](https://arxiv.org/abs/2008.11293)

Abstract:

`We consider the problem of automatically generating a narrative biomedical evidence summary from multiple trial reports. We evaluate modern neural models for abstractive summarization of relevant article abstracts from systematic reviews previously conducted by members of the Cochrane collaboration, using the authors conclusions section of the review abstract as our target. We enlist medical professionals to evaluate generated summaries, and we find that modern summarization systems yield consistently fluent and relevant synopses, but that they are not always factual. We propose new approaches that capitalize on domain-specific models to inform summarization, e.g., by explicitly demarcating snippets of inputs that convey key findings, and emphasizing the reports of large and high-quality trials. We find that these strategies modestly improve the factual accuracy of generated summaries. Finally, we propose a new method for automatically evaluating the factuality of generated narrative evidence syntheses using models that infer the directionality of reported findings.`

Homepage: [GitHub Pages](https://github.com/bwallace/RCT-summarization-data?tab=readme-ov-file)


### Citation

```bibtex
  @inproceedings{AMIA-summarization-2021,
      title = {{Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization}},
      author = {Byron C. Wallace and Sayantan Saha and Frank Soboczenski and Iain J. Marshall},
      Booktitle = {{Proceedings of AMIA Informatics Summit}},
      year = {2021},
  }
```

### Groups, Tags, and Tasks

#### Groups

* `RCT_Summary`: `The SemEval NLI4CT task, which includes the 2023 and 2024 tasks on natural language inference for clinical trials, focusing on reasoning over clinical trial reports and statements about them.`

#### Tags

#### Tasks

* `RCT_Summary` : `RCT Summary task, generating conclusion sections from Title and Abstract`

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