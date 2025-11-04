# HINT: Hierarchical interaction network for clinical-trial-outcome predictions

Huggingface Dataset Link: [HINT](https://arxiv.org/abs/2102.04252)

## Paper

Title: `HINT: Hierarchical Interaction Network for Trial Outcome Prediction Leveraging Web Data`

Paper Link: [arXiv](https://arxiv.org/abs/2102.04252)

Abstract:

`Clinical trials are crucial for drug development but are time consuming, expensive, and often burdensome on patients. More importantly, clinical trials face uncertain outcomes due to issues with efficacy, safety, or problems with patient recruitment. If we were better at predicting the results of clinical trials, we could avoid having to run trials that will inevitably fail more resources could be devoted to trials that are likely to succeed. In this paper, we propose Hierarchical INteraction Network (HINT) for more general, clinical trial outcome predictions for all diseases based on a comprehensive and diverse set of web data including molecule information of the drugs, target disease information, trial protocol and biomedical knowledge. HINT first encode these multi-modal data into latent embeddings, where an imputation module is designed to handle missing data. Next, these embeddings will be fed into the knowledge embedding module to generate knowledge embeddings that are pretrained using external knowledge on pharmaco-kinetic properties and trial risk from the web. Then the interaction graph module will connect all the embedding via domain knowledge to fully capture various trial components and their complex relations as well as their influences on trial outcomes. Finally, HINT learns a dynamic attentive graph neural network to predict trial outcome. Comprehensive experimental results show that HINT achieves strong predictive performance, obtaining 0.772, 0.607, 0.623, 0.703 on PR-AUC for Phase I, II, III, and indication outcome prediction, respectively. It also consistently outperforms the best baseline method by up to 12.4\% on PR-AUC.`

Homepage: [GitHub](https://github.com/futianfan/clinical-trial-outcome-prediction)

### Citation

```bibtex
@article{fu2021hint,
  title={HINT: Hierarchical Interaction Network for Trial Outcome Prediction Leveraging Web Data},
  author={Fu, Tianfan and Huang, Kexin and Xiao, Cao and Glass, Lucas M and Sun, Jimeng},
  journal={arXiv preprint arXiv:2102.04252},
  year={2021}
}
```

### Groups, Tags, and Tasks

#### Groups

* `HINT`: `Evaluates models on the HINT dataset, which is a hierarchical interaction network for clinical trial outcome predictions.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `HINT_0-shot`        : `HINT in single-turn 0-shot setting`
* `HINT_CoT`           : `HINT in single-turn chain-of-thought 0-shot setting`
* `HINT_SC`            : `HINT in single-turn self-consistency setting`
* `HINT_0-shot_CoT_SC` : `HINT in single-turn chain-of-thought with self-consistency setting`

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
