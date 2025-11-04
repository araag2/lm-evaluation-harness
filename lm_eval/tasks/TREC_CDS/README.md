# TREC Clinical Decision Support (2014, 2015 and 2016)

Huggingface Dataset Link: [TREC Clinical Decision Support](https://huggingface.co/datasets/araag2/TREC_Clinicial-Decision-Support)

## Paper

Title: `Overview of the TREC 2014 / 2015 / 2016 Clinical Decision Support Track`

Paper Link: [2014](https://trec.nist.gov/pubs/trec23/papers/overview-clinical.pdf) and [2015](https://trec.nist.gov/pubs/trec24/papers/Overview-CL.pdf) and [2016](https://trec.nist.gov/pubs/trec25/papers/Overview-CL.pdf)

Homepage: [TREC CDS](https://www.trec-cds.org/)

### Citation

```bibtex
@inproceedings{simpson2014overview,
  title={Overview of the TREC 2014 Clinical Decision Support Track.},
  author={Simpson, Matthew S and Voorhees, Ellen M and Hersh, William R},
  booktitle={TREC},
  year={2014}
}

@inproceedings{gobeill2015exploiting,
  title={Exploiting incoming and outgoing citations for improving Information Retrieval in the TREC 2015 Clinical Decision Support Track},
  author={Gobeill, Julien and Gaudinat, Arnaud and Ruch, Patrick},
  booktitle={Proceedings of The 24th Text REtrieval Conference (TREC 2015)},
  year={2015},
  organization={17-20 November 2015}
}

@inproceedings{roberts2016overview,
  title={Overview of the TREC 2016 Clinical Decision Support Track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Hersh, William R},
  booktitle={25th Text REtrieval Conference, TREC 2016},
  year={2016}
}

@article{nguyen2018benchmarking,
  title={Benchmarking clinical decision support search},
  author={Nguyen, Vincent and Karimi, Sarvnaz and Falamaki, Sara and Paris, Cecile},
  journal={arXiv preprint arXiv:1801.09322},
  year={2018}
}
```

### Groups, Tags, and Tasks

#### Groups

* `TREC_CDS_0-shot`: `Evaluates models on the TREC Clinical Decision Support dataset in zero-shot setting.`
* `TREC_CDS_CoT`: `Evaluates models on the TREC Clinical Decision Support dataset with single-turn chain-of-thought prompting.`
* `TREC_CDS_SC`: `Evaluates models on the TREC Clinical Decision Support dataset with single-turn self-consistency sampling.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `TREC_CDS_2014_0-shot` : `TREC Clinical Decision Support 2014 in single-turn 0-shot setting`
* `TREC_CDS_2015_0-shot` : `TREC Clinical Decision Support 2015 in single-turn 0-shot setting`
* `TREC_CDS_2016_0-shot` : `TREC Clinical Decision Support 2016 in single-turn 0-shot setting`
* `TREC_CDS_2014_CoT`    : `TREC Clinical Decision Support 2014 in single-turn chain-of-thought 0-shot setting`
* `TREC_CDS_2015_CoT`    : `TREC Clinical Decision Support 2015 in single-turn chain-of-thought 0-shot setting`
* `TREC_CDS_2016_CoT`    : `TREC Clinical Decision Support 2016 in single-turn chain-of-thought 0-shot setting`
* `TREC_CDS_2014_SC`     : `TREC Clinical Decision Support 2014 in single-turn self-consistency setting`
* `TREC_CDS_2015_SC`     : `TREC Clinical Decision Support 2015 in single-turn self-consistency setting`
* `TREC_CDS_2016_SC`     : `TREC Clinical Decision Support 2016 in single-turn self-consistency setting`
  
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
