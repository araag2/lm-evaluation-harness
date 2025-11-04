# Natural Language Inference for Patient Recruitment (NLI4PR)

Huggingface Dataset Link: [NLI4PR](https://huggingface.co/datasets/araag2/NLI4PR)

## Paper

Title: `Am I eligible? Natural Language Inference for Clinical Trial Patient Recruitment: the Patient's Point of View`

Paper Link: [arXiv](https://arxiv.org/abs/2503.15718)

Abstract:

`Recruiting patients to participate in clinical trials can be challenging and time-consuming. Usually, participation in a clinical trial is initiated by a healthcare professional and proposed to the patient. Promoting clinical trials directly to patients via online recruitment might help to reach them more efficiently. In this study, we address the case where a patient is initiating their own recruitment process and wants to determine whether they are eligible for a given clinical trial, using their own language to describe their medical profile. To study whether this creates difficulties in the patient trial matching process, we design a new dataset and task, Natural Language Inference for Patient Recruitment (NLI4PR), in which patient language profiles must be matched to clinical trials. We create it by adapting the TREC 2022 Clinical Trial Track dataset, which provides patients' medical profiles, and rephrasing them manually using patient language. We also use the associated clinical trial reports where the patients are either eligible or excluded. We prompt several open-source Large Language Models on our task and achieve from 56.5 to 71.8 of F1 score using patient language, against 64.7 to 73.1 for the same task using medical language. When using patient language, we observe only a small loss in performance for the best model, suggesting that having the patient as a starting point could be adopted to help recruit patients for clinical trials. The corpus and code bases are all freely available on our Github and HuggingFace repositories.`

Homepage: [GitHub](https://aclanthology.org/2025.cl4health-1.21/)

### Citation

```bibtex
  @misc{aguiar2025ieligiblenaturallanguage,
        title={Am I eligible? Natural Language Inference for Clinical Trial Patient Recruitment: the Patient's Point of View}, 
        author={Mathilde Aguiar and Pierre Zweigenbaum and Nona Naderi},
        year={2025},
        eprint={2503.15718},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2503.15718}, 
  }
```

### Groups, Tags, and Tasks

#### Groups

* `NLI4PR_0-shot`: `Evaluates models on the NLI4PR dataset in zero-shot setting.`
* `NLI4PR_CoT`: `Evaluates models on the NLI4PR dataset with single-turn chain-of-thought prompting.`
* `NLI4PR_SC`: `Evaluates models on the NLI4PR dataset with single-turn self-consistency sampling.`
* `NLI4PR_CoT_SC`: `Evaluates models on the NLI4PR dataset with single-turn chain-of-thought prompting and self-consistency sampling.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `NLI4PR_medical-lang_0-shot` : `NLI4PR using medical language in single-turn 0-shot setting`
* `NLI4PR_patient-lang_0-shot` : `NLI4PR using patient language in single-turn 0-shot setting`
* `NLI4PR_medical-lang_CoT`    : `NLI4PR using medical language in single-turn chain-of-thought 0-shot setting`
* `NLI4PR_patient-lang_CoT`    : `NLI4PR using patient language in single-turn chain-of-thought 0-shot setting`
* `NLI4PR_medical-lang_SC`     : `NLI4PR using medical language in single-turn self-consistency setting`
* `NLI4PR_patient-lang_SC`     : `NLI4PR using patient language in single-turn self-consistency setting`
* `NLI4PR_medical-lang_CoT_SC` : `NLI4PR using medical language in single-turn chain-of-thought with self-consistency setting`
* `NLI4PR_patient-lang_CoT_SC` : `NLI4PR using patient language in single-turn chain-of-thought with self-consistency setting`

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
