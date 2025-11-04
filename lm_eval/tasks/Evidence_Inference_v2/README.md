# Evidence Inference v2

Huggingface Dataset Link: [Evidence Inference v2](https://huggingface.co/datasets/araag2/Evidence_Inference_v2)

## Paper

Title: `Evidence Inference 2.0: More Data, Better Models`

Paper Link: [arXiv](https://arxiv.org/abs/2005.04177)

Abstract:

`How do we most effectively treat a disease or condition? Ideally, we could consult a database of evidence gleaned from clinical trials to answer such questions. Unfortunately, no such database exists; clinical trial results are instead disseminated primarily via lengthy natural language articles. Perusing all such articles would be prohibitively time-consuming for healthcare practitioners; they instead tend to depend on manually compiled systematic reviews of medical literature to inform care. NLP may speed this process up, and eventually facilitate immediate consult of published evidence. The Evidence Inference dataset was recently released to facilitate research toward this end. This task entails inferring the comparative performance of two treatments, with respect to a given outcome, from a particular article (describing a clinical trial) and identifying supporting evidence. For instance: Does this article report that chemotherapy performed better than surgery for five-year survival rates of operable cancers? In this paper, we collect additional annotations to expand the Evidence Inference dataset by 25\%, provide stronger baseline models, systematically inspect the errors that these make, and probe dataset quality. We also release an abstract only (as opposed to full-texts) version of the task for rapid model prototyping. The updated corpus, documentation, and code for new baselines and evaluations are available at this http URL.`

Homepage: [Evidence Inference](https://evidence-inference.ebm-nlp.com/)

### Citation

```bibtex
@article{deyoung2020evidence,
  title={Evidence inference 2.0: More data, better models},
  author={DeYoung, Jay and Lehman, Eric and Nye, Ben and Marshall, Iain J and Wallace, Byron C},
  journal={arXiv preprint arXiv:2005.04177},
  year={2020}
}
```

### Groups, Tags, and Tasks

#### Groups

* `Evidence_Inference_v2`: `Evaluates models on the Evidence Inference v2 dataset, which involves inferring comparative performance of treatments from clinical trial articles.`

#### Tags

* `0-shot`: `Evaluates models in zero-shot setting`
* `chain-of-thought`: `Evaluates models with chain-of-thought prompting`
* `self-consistency`: `Evaluates models with self-consistency sampling`

#### Tasks

* `Evidence_Inference_v2_0-shot` : `Evidence Inference v2 in single-turn 0-shot setting`
* `Evidence_Inference_v2_CoT`    : `Evidence Inference v2 in single-turn chain-of-thought 0-shot setting`
* `Evidence_Inference_v2_SC`     : `Evidence Inference v2 in single-turn self-consistency setting`
* `Evidence_Inference_v2_CoT_SC` : `Evidence Inference v2 in single-turn chain-of-thought with self-consistency setting`

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
