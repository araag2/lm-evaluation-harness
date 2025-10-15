---
dataset_info:
- config_name: conversational_binary
  features:
  - name: id
    dtype: int64
  - name: prompt
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: completion
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: Label
    dtype: string
  splits:
  - name: dev
    num_bytes: 176937
    num_examples: 11
  - name: test
    num_bytes: 2318936
    num_examples: 172
  download_size: 611329
  dataset_size: 2495873
- config_name: conversational_continuous
  features:
  - name: id
    dtype: int64
  - name: prompt
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: completion
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: Label
    dtype: string
  splits:
  - name: dev
    num_bytes: 401299
    num_examples: 32
  - name: test
    num_bytes: 6722692
    num_examples: 484
  download_size: 1473987
  dataset_size: 7123991
- config_name: conversational_outcome-type
  features:
  - name: id
    dtype: int64
  - name: prompt
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: completion
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: Label
    dtype: string
  splits:
  - name: dev
    num_bytes: 523092
    num_examples: 43
  - name: test
    num_bytes: 8201941
    num_examples: 656
  download_size: 1568521
  dataset_size: 8725033
- config_name: processed
  features:
  - name: id
    dtype: int64
  - name: pmcid
    dtype: int64
  - name: outcome
    dtype: string
  - name: intervention
    dtype: string
  - name: comparator
    dtype: string
  - name: outcome_type
    dtype: string
  - name: intervention_events
    dtype: string
  - name: intervention_group_size
    dtype: string
  - name: comparator_events
    dtype: float64
  - name: comparator_group_size
    dtype: string
  - name: intervention_mean
    dtype: string
  - name: intervention_standard_deviation
    dtype: float64
  - name: comparator_mean
    dtype: float64
  - name: comparator_standard_deviation
    dtype: float64
  - name: notes
    dtype: string
  - name: is_data_in_figure_graphics
    dtype: bool
  - name: is_relevant_data_in_table
    dtype: bool
  - name: is_table_in_graphic_format
    dtype: bool
  - name: is_data_complete
    dtype: bool
  - name: Context
    dtype: string
  splits:
  - name: dev
    num_bytes: 496327
    num_examples: 43
  - name: test
    num_bytes: 7804844
    num_examples: 656
  download_size: 684957
  dataset_size: 8301171
- config_name: processed_binary
  features:
  - name: id
    dtype: int64
  - name: pmcid
    dtype: int64
  - name: outcome
    dtype: string
  - name: intervention
    dtype: string
  - name: comparator
    dtype: string
  - name: notes
    dtype: string
  - name: is_data_in_figure_graphics
    dtype: bool
  - name: is_relevant_data_in_table
    dtype: bool
  - name: is_table_in_graphic_format
    dtype: bool
  - name: is_data_complete
    dtype: bool
  - name: Context
    dtype: string
  - name: Completion Extraction
    dtype: string
  splits:
  - name: dev
    num_bytes: 157943
    num_examples: 11
  - name: test
    num_bytes: 2023623
    num_examples: 172
  download_size: 314417
  dataset_size: 2181566
- config_name: processed_continuous
  features:
  - name: id
    dtype: int64
  - name: pmcid
    dtype: int64
  - name: outcome
    dtype: string
  - name: intervention
    dtype: string
  - name: comparator
    dtype: string
  - name: notes
    dtype: string
  - name: is_data_in_figure_graphics
    dtype: bool
  - name: is_relevant_data_in_table
    dtype: bool
  - name: is_table_in_graphic_format
    dtype: bool
  - name: is_data_complete
    dtype: bool
  - name: Context
    dtype: string
  - name: Completion Extraction
    dtype: string
  splits:
  - name: dev
    num_bytes: 341340
    num_examples: 32
  - name: test
    num_bytes: 5825791
    num_examples: 484
  download_size: 566133
  dataset_size: 6167131
- config_name: source
  features:
  - name: id
    dtype: int64
  - name: evidence_inference_prompt_id
    dtype: int64
  - name: pmcid
    dtype: int64
  - name: outcome
    dtype: string
  - name: intervention
    dtype: string
  - name: comparator
    dtype: string
  - name: outcome_type
    dtype: string
  - name: intervention_events
    dtype: string
  - name: intervention_group_size
    dtype: string
  - name: comparator_events
    dtype: float64
  - name: comparator_group_size
    dtype: string
  - name: intervention_mean
    dtype: string
  - name: intervention_standard_deviation
    dtype: float64
  - name: comparator_mean
    dtype: float64
  - name: comparator_standard_deviation
    dtype: float64
  - name: notes
    dtype: string
  - name: is_data_in_figure_graphics
    dtype: bool
  - name: is_relevant_data_in_table
    dtype: bool
  - name: is_table_in_graphic_format
    dtype: bool
  - name: is_data_complete
    dtype: bool
  - name: tiktoken_with_attributes_xml_token_num
    dtype: int64
  - name: tiktoken_without_attributes_xml_token_num
    dtype: int64
  - name: tiktoken_without_attributes_markdown_token_num
    dtype: int64
  - name: split
    dtype: string
  - name: abstract
    dtype: string
  splits:
  - name: dev
    num_bytes: 498004
    num_examples: 43
  - name: test
    num_bytes: 7831084
    num_examples: 656
  download_size: 696118
  dataset_size: 8329088
configs:
- config_name: conversational_binary
  data_files:
  - split: dev
    path: conversational_binary/dev-*
  - split: test
    path: conversational_binary/test-*
- config_name: conversational_continuous
  data_files:
  - split: dev
    path: conversational_continuous/dev-*
  - split: test
    path: conversational_continuous/test-*
- config_name: conversational_outcome-type
  data_files:
  - split: dev
    path: conversational_outcome-type/dev-*
  - split: test
    path: conversational_outcome-type/test-*
- config_name: processed
  data_files:
  - split: dev
    path: processed/dev-*
  - split: test
    path: processed/test-*
- config_name: processed_binary
  data_files:
  - split: dev
    path: processed_binary/dev-*
  - split: test
    path: processed_binary/test-*
- config_name: processed_continuous
  data_files:
  - split: dev
    path: processed_continuous/dev-*
  - split: test
    path: processed_continuous/test-*
- config_name: source
  data_files:
  - split: dev
    path: source/dev-*
  - split: test
    path: source/test-*
license: cc-by-sa-4.0
task_categories:
- text-classification
- text-retrieval
language:
- en
tags:
- medical
pretty_name: Trial Meta Analysis
size_categories:
- 1K<n<10K
---

# Automatically Extracting Numerical Results from Randomized Controlled Trials with LLMs

## Dataset Description

|                                 | Links         | 
|:-------------------------------:|:-------------:|
| **Homepage:**                   |  [Github](https://github.com/jayded/llm-meta-analysis)  | 
| **Repository:**                 |  [Github](https://github.com/jayded/llm-meta-analysis)  | 
| **Paper:**                      |  [arXiv](https://arxiv.org/abs/2405.01686)  | 
| **Leaderboard:**                |  [PapersWithCode](https://paperswithcode.com/paper/automatically-extracting-numerical-results)  | 
| **Contact (Original Authors):** |  Hye Sun Yun (yun.hy@northeastern.edu) |
| **Contact (Curator):**          |  [Artur Guimarães](https://araag2.netlify.app/) (artur.guimas@gmail.com) |

  
### Dataset Summary

`The human-annotated data is available in the data folder as both csv and json formats. The dev set has 10 RCTs with 43 number of total ICO triplets. The test set has 110 RCTs with 656 number of total ICO triplets. Additional dataset statistics can be found in evaluation/data/analyze_data.ipynb file.`

### Data Instances

#### Source Format

```json
{
    "id":6,
    "evidence_inference_prompt_id":11451,
    "pmcid":1574360,
    "outcome":"The amount of saturated fat in the foods purchased",
    "intervention":"fully automated advice that recommended specific switches from selected products higher in saturated fat to alternate similar products lower in saturated fat",
    "comparator":"control received general non-specific advice about how to eat a diet lower in saturated fat","outcome_type":"continuous",
    "intervention_events":null,
    "intervention_group_size":"251",
    "comparator_events":null,
    "comparator_group_size":"246",
    "intervention_mean":"0.77",
    "intervention_standard_deviation":1.37,
    "comparator_mean":0.04,
    "comparator_standard_deviation":0.32,
    "notes":"Have to use CI to calculate the SDs",
    "is_data_in_figure_graphics":false,
    "is_relevant_data_in_table":false,
    "is_table_in_graphic_format":true,
    "is_data_complete":true,
    "tiktoken_with_attributes_xml_token_num":1959,
    "tiktoken_without_attributes_xml_token_num":1620,
    "tiktoken_without_attributes_markdown_token_num":1376,
    "split":"test",
    "abstract":"# Abstract\n\n \n\n## Objectives: The supermarket industry now services many customers through online food shopping over the Internet. The Internet shopping process offers a novel opportunity for the modification of dietary patterns. The aim of this study was to evaluate the effects on consumers' purchases of saturated fat of a fully automated computerised system that provided real-time advice tailored to the consumers' specific purchases recommending foods lower in saturated fat. \n\n## Design: This study was a blinded, randomised controlled trial. \n\n## Setting: The study was conducted in Sydney, New South Wales, Australia. \n\n## Participants: The participants were consumers using a commercial online Internet shopping site between February and June 2004. \n\n## Interventions: Individuals assigned to intervention received fully automated advice that recommended specific switches from selected products higher in saturated fat to alternate similar products lower in saturated fat. Participants assigned to control received general non-specific advice about how to eat a diet lower in saturated fat. \n\n## Outcome Measures: The outcome measure was the difference in saturated fat (grams per 100 g of food) in shopping baskets between the intervention and control groups. \n\n## Results: There were 497 randomised participants, mean age 40 y, each shopping for an average of about three people. The amount of saturated fat in the foods purchased by the intervention group was 0.66% lower (95% confidence interval 0.48--0.84, *p* < 0.001) than in the control group. The effects of the intervention were sustained over consecutive shopping episodes, and there was no difference in the average cost of the food bought by each group. \n\n## Conclusions: Fully automated, purchase-specific dietary advice offered to customers during Internet shopping can bring about changes in food purchasing habits that are likely to have significant public health implications. Because implementation is simple to initiate and maintain, this strategy would likely be highly cost-effective. # RESULTS \n\n## Participants A total of 4,548 individuals were offered participation, and 497 were randomised (251 intervention and 246 control) ([Figure 1](#)). Of these, 456 (224 intervention and 232 control) completed at least one episode of shopping that included one or more of the 524 foods studied. Median follow-up time completed by the end of the study in June 2004 was 35 d, and the median number of shopping episodes done by participants was three (range 1--20). The baseline characteristics documented on the questionnaire were balanced between randomised groups, with a mean participant age of 40 y and a proportion female of 88% ([Table 1](#)). <figure> <p><img src=\"\" \/><\/p> <figcaption>Flow Chart of Study<\/figcaption> <\/figure> ::::table-wrap\n\n::: caption\nBaseline Characteristics\n:::\n\n![](<>):::: \n\n## Outcomes and Estimation For the first occasion on which advice was offered, the amount of saturated fat in the food purchased by the intervention group after advice was a mean of 0.66% (0.48--0.84, *p \\<* 0.001) lower than in the corresponding foods purchased by the control group ([Table 2](#)), which is equivalent to an approximate 10% reduction in saturated fat content of foods purchased ([Figure 2](#)). This difference resulted from a decrease in the mean saturated fat content in the foods purchased following the advice offered to the intervention group of 0.77% (0.60--0.94, *p \\<* 0.001), with no corresponding decrease in the control group 0.04% (0.00--0.08, *p =* 0.07). The effect estimate for the primary outcome was 0.62% (0.46--0.79, *p \\<* 0.001) if analysis was restricted to only the 456 individuals that selected one of the 524 foods studied and was 0.58 (0.39--0.77, *p \\<* 0.001) if the change in saturated fat was set to zero for those individuals that did not select one of the foods. The subgroup analyses provided some evidence that the intervention had greater effects among individuals with higher body mass index and among people above 40 y of age (for both, homogeneity *p* < 0.03) ([Table 3](#)). There was no baseline variable that substantively altered the main result as a consequence of its inclusion or exclusion as a covariate. ::::table-wrap\n::: caption\nEffects of Intervention on Primary Outcome\n:::\n\n![](<>):::: <figure> <p><img src=\"\" \/><\/p> <figcaption>Effects of Repeated Advice in the Intervention (<em>n =<\/em> 115) and Control Group (<em>n =<\/em> 121)<br \/> Squares are placed at the point estimate of the effect observed in the intervention (A) and control (B) groups, and the vertical lines extend to the 95% confidence intervals around the estimate. The <em>p<\/em>-value for trend in the intervention group indicates a progressive decrease in effect size with repeated shopping episodes. There was no significant decrease in saturated fat at any time point in the control group, nor any trend over time (all <em>p<\/em> > 0.09).<\/figcaption> <\/figure> ::::table-wrap\n::: caption\nEffects of Intervention on Primary Outcome in Main Participant Subgroups\n:::\n\n![](<>):::: A secondary outcome of the study was to assess the impact of the intervention on the cost of foods purchased online. The mean cost per 100 g of the foods purchased by the intervention group was not different from that in the control group (intervention AUD$0.63 [0.58--0.68]\/100 g versus control AUD$0.62 [0.58--0.67]\/100 g, *p =* 0.19). The foods higher in saturated fat that were most commonly present in the shopping basket prior to advice being offered but absent after the advice had been offered were higher-fat dairy products. Lower-fat dairy products were the items most frequently added to the shopping basket after advice was provided. \n\n## Ancillary Analyses The effects of the intervention over repeated episodes of shopping were explored amongst the 115 participants that completed six shopping episodes during the study. These analyses demonstrated that for the intervention group the magnitude of the reduction in saturated fat achieved was greater in earlier compared to later shopping episodes (for trend, *p* = 0.01) ([Figure 2](#)) and showed that there was no effect of the control condition on saturated fat during any shopping episode (*p* > 0.09 for all six shopping episodes in the control group).\n"
}
```

### Data Fields

#### Source Format

TO:DO

### Data Splits

TO:DO

## Additional Information

### Dataset Curators

#### Original Paper

- Hye Sun Yun (yun.hy@northeastern.edu) - Northeastern University Boston, MA, USA
- David Pogrebitskiy (pogrebitskiy.d@northeastern.edu) - Northeastern University Boston, MA, USA
- Iain J. Marshall (iain.marshall@kcl.ac.uk) - King’s College London London, UK
- Byron C. Wallace (b.wallace@northeastern.edu) - Northeastern University Boston, MA, USA

#### Huggingface Curator

- [Artur Guimarães](https://araag2.netlify.app/) (artur.guimas@gmail.com) - INESC-ID / University of Lisbon - Instituto Superior Técnico

### Licensing Information

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)

### Citation Information

```bibtex
@article{yun2024automatically,
  title={Automatically extracting numerical results from randomized controlled trials with large language models},
  author={Yun, Hye Sun and Pogrebitskiy, David and Marshall, Iain J and Wallace, Byron C},
  journal={arXiv preprint arXiv:2405.01686},
  year={2024}
}
```

[10.48550/arXiv.2405.01686](https://doi.org/10.48550/arXiv.2405.01686)

### Contributions

Thanks to [araag2](https://github.com/araag2) for adding this dataset.