# TREC Precision Medicine (2017, 2018 and 2019)

Huggingface Dataset Link: [TREC Precision Medicine](https://huggingface.co/datasets/araag2/TREC_Precision-Medicine)

## Paper

Title: `Overview of the TREC 2017 / 2018 / 2019 Precision Medicine Track`

Paper Link: [2017](https://trec.nist.gov/pubs/trec26/papers/Overview-PM.pdf) and [2018](https://trec.nist.gov/pubs/trec27/papers/Overview-PM.pdf) and [2019](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.PM.pdf)

Homepage: [TREC](https://www.trec-cds.org/)

### Citation

```bibtex
@inproceedings{roberts2017overview,
  title={Overview of the TREC 2017 precision medicine track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Hersh, William R and Bedrick, Steven and Lazar, Alexander J and Pant, Shubham},
  booktitle={The... text REtrieval conference: TREC. Text REtrieval Conference},
  volume={26},
  pages={https--trec},
  year={2017}
}

@inproceedings{roberts2018overview,
  title={Overview of the TREC 2018 Precision Medicine Track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Hersh, William R and Bedrick, Steven and Lazar, Alexander J},
  booktitle={27th Text REtrieval Conference, TREC 2018},
  year={2018}
}

@inproceedings{roberts2019overview,
  title={Overview of the TREC 2019 precision medicine track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Hersh, William R and Bedrick, Steven and Lazar, Alexander J and Pant, Shubham and Meric-Bernstam, Funda},
  booktitle={The... Text REtrieval Conference: TREC. Text REtrieval Conference},
  volume={1250},
  pages={https--trec},
  year={2019}
}

@inproceedings{roberts2020overview,
  title={Overview of the TREC 2020 precision medicine track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Bedrick, Steven and Hersh, William R},
  booktitle={The... text REtrieval conference: TREC. Text REtrieval Conference},
  volume={1266},
  pages={https--trec},
  year={2020}
}
```

### Groups, Tags, and Tasks

#### Groups

* `TREC_Prec-Med_0-shot` : `TREC Precision Medicine tasks in zero-shot setting`
* `TREC_Prec-Med_CoT`    : `TREC Precision Medicine tasks with chain-of-thought prompting`
* `TREC_Prec-Med_SC`     : `TREC Precision Medicine tasks with self-consistency sampling`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `TREC_Prec-Med_2017_0-shot` : `TREC Precision Medicine 2017 in single-turn 0-shot setting`
* `TREC_Prec-Med_2018_0-shot` : `TREC Precision Medicine 2018 in single-turn 0-shot setting`
* `TREC_Prec-Med_2019_0-shot` : `TREC Precision Medicine 2019 in single-turn 0-shot setting`
* `TREC_Prec-Med_2017_CoT`    : `TREC Precision Medicine 2017 in single-turn chain-of-thought 0-shot setting`
* `TREC_Prec-Med_2018_CoT`    : `TREC Precision Medicine 2018 in single-turn chain-of-thought 0-shot setting`
* `TREC_Prec-Med_2019_CoT`    : `TREC Precision Medicine 2019 in single-turn chain-of-thought 0-shot setting`
* `TREC_Prec-Med_2017_SC`     : `TREC Precision Medicine 2017 in single-turn self-consistency setting`
* `TREC_Prec-Med_2018_SC`     : `TREC Precision Medicine 2018 in single-turn self-consistency setting`
* `TREC_Prec-Med_2019_SC`     : `TREC Precision Medicine 2019 in single-turn self-consistency setting`

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
