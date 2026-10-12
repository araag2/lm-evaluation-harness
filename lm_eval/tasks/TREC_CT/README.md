# TREC Clinical Trials (2021, 2022 and 2023)

Huggingface Dataset Link: [TREC Clinical Trials](https://huggingface.co/datasets/araag2/TREC_Clinical-Trials)

## Paper

Title: `Overview of the TREC 2021 / 2022 / 2023 Clinical Trials Track`

Paper Link: [2021](https://trec.nist.gov/pubs/trec30/papers/Overview-2021.pdf) and [2022](https://trec.nist.gov/pubs/trec31/papers/Overview_trials.pdf) and [2023](https://trec.nist.gov/pubs/trec32/papers/overview_32.pdf)

Homepage: [TREC](https://www.trec-cds.org/)

### Citation

```bibtex
@inproceedings{soboroff2021overview,
  title={Overview of TREC 2021.},
  author={Soboroff, Ian},
  booktitle={TREC},
  year={2021}
}

@inproceedings{roberts2022overview,
  title={Overview of the TREC 2022 Clinical Trials Track.},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Bedrick, Steven and Hersh, William R},
  booktitle={TREC},
  year={2022}
}

@inproceedings{soboroff2023overview,
  title={Overview of TREC 2023.},
  author={Soboroff, Ian},
  booktitle={TREC},
  year={2023}
}
```

### Groups, Tags, and Tasks

#### Groups

* `TREC_CT_0-shot`: `Evaluates models on the TREC Clinical Trials dataset in zero-shot setting.`
* `TREC_CT_CoT`: `Evaluates models on the TREC Clinical Trials dataset with single-turn chain-of-thought prompting.`
* `TREC_CT_SC`: `Evaluates models on the TREC Clinical Trials dataset with single-turn self-consistency sampling.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `TREC_CT_2021_0-shot` : `TREC Clinical Trials 2021 in single-turn 0-shot setting`
* `TREC_CT_2022_0-shot` : `TREC Clinical Trials 2022 in single-turn 0-shot setting`
* `TREC_CT_2023_0-shot` : `TREC Clinical Trials 2023 in single-turn 0-shot setting` 
* `TREC_CT_2021_CoT`    : `TREC Clinical Trials 2021 in single-turn chain-of-thought 0-shot setting`
* `TREC_CT_2022_CoT`    : `TREC Clinical Trials 2022 in single-turn chain-of-thought 0-shot setting`
* `TREC_CT_2023_CoT`    : `TREC Clinical Trials 2023 in single-turn chain-of-thought 0-shot setting`
* `TREC_CT_2021_SC`     : `TREC Clinical Trials 2021 in single-turn self-consistency setting`
* `TREC_CT_2022_SC`     : `TREC Clinical Trials 2022 in single-turn self-consistency setting`
* `TREC_CT_2023_SC`     : `TREC Clinical Trials 2023 in single-turn self-consistency setting`

### Labels and metrics

`Label` comes from the TREC qrels: 0 `not relevant`, 1 `possibly relevant`, 2 `definitely relevant`. In the
track these grades are *not relevant*, *excluded* (the trial is about the patient's condition, but the patient
fails its eligibility criteria) and *eligible*.

Every task is three-way multiple choice over those labels (`doc_to_choice`).

* `acc`: exact three-way accuracy.
* `nDCG@10`: per `query_id`, with the graded label as gain and the probability of `definitely relevant` (the
  last choice) as the ranking score.
* `P@10`, `RecRank`: per `query_id`, ranked by the same score, with the label collapsed to binary:
  `possibly relevant` and `definitely relevant` are positive, `not relevant` is negative (`_binary_label` in
  `lm_eval/opencteval/metrics.py`).
* The `balanced` tasks (`process_docs_balanced`) keep every pair of either relevant grade and an equal number
  of `not relevant` pairs, sampled with seed 42.

A pair in which the patient has the trial's condition but fails a criterion is therefore `possibly relevant`
and counts as positive in the binary metrics.

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
