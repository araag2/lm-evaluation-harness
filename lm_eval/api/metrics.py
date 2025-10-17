import sys
import logging
import math
import os
import random
import re
import string
from collections.abc import Iterable
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, TypeVar

import numpy as np
import sacrebleu

from lm_eval.api.registry import register_aggregation, register_metric


T = TypeVar("T")

eval_logger = logging.getLogger(__name__)

### New Metrics Defined for my benchmark of lm-evaluation-harness ###
#-----------------------------------------------------------------------#
def items_to_query_id_dict(items, pos_label_index):
    """
    Group a list of items by query ID and extract the probability score of the positive label.

    Each item is a tuple: (doc, gold, pred, prob_norm), where `prob_norm` is a list of normalized
    probability scores. The function groups them by `doc["query_id"]` and pairs each entry
    with the positive label probability.

    Args:
        items: List of tuples. Each tuple contains:
            - doc: Dictionary containing at least a "query_id" key.
            - gold: Ground truth label (int).
            - pred: Predicted label (int).
            - prob_norm: List of floats representing normalized probability scores.
        pos_label_index_index: Index for the positive label probability within `prob_norm`.

    Returns:
        A dict mapping `query_id` to a list of tuples: (doc, gold, pred, positive_label_prob).
    """
    scores_by_query_id = defaultdict(list)
    for doc, gold, pred, prob_norm in items:
        score = prob_norm[pos_label_index] if isinstance(pos_label_index, int) else sum(prob_norm[i] for i in pos_label_index)
        scores_by_query_id[doc["query_id"]].append((doc, gold, pred, score))
    return scores_by_query_id

def score_per_query_id(items, score_function_fn, cutoff_fn = None):
    """
    Compute an average score per query using a generic scoring function and cutoff logic.

    Groups items by query ID, sorts them by positive-label probability, applies a cutoff
    function to select the top items, computes per-query scores, and returns their mean.

    Args:
        items: List of tuples as expected by `group_by_query`.
        score_function_fn: Scoring function from `sklearn.metrics` (e.g. precision_score),
            must accept y_true and y_pred lists and return a float.
        cutoff_fn: Function that takes the sorted list of items for a query and returns
            an integer cutoff count to evaluate.

    Returns:
        The mean score across all query IDs. If no items are provided, returns 0.0.
    """

    # Assuming the last label in available choices is the positive label
    grouped = items_to_query_id_dict(items, pos_label_index=len(items[0][3]) - 1)  # <query_id, (doc, gold, pred, prob_norm[pos_label_index])>
    scores = []

    for qid, docs in grouped.items():
        sorted_items = sorted(docs, key=lambda x: x[3], reverse=True)

        if cutoff_fn:
            sorted_items = sorted_items[:cutoff_fn(sorted_items)]
        
        # In TREC, 1 (maybe) or 2 (yes) are consolidated the positive labels
        y_true = [1 if gold > 0 else 0 for _, gold, _, _ in sorted_items] 
        y_pred = [1 if pred > 0 else 0 for _, _, pred, _ in sorted_items]
        
        scores.append(score_function_fn(y_true, y_pred, zero_division=0))

        # TO:DO Optional debug prints (remove later)
        #print(f"[DEBUG] Query {qid} – y_true: {y_true}, y_pred: {y_pred}")
        #for doc, gold, pred, prob in sorted_items:
        #    print(f"[DEBUG] doc_id: {doc['doc_id']}, Gold: {gold}, Pred: {pred}, Prob: {prob}")
        #print(f"[DEBUG] Query {qid} – Score: {scores[-1]}")
    
    return mean(scores) if scores else 0.0

@register_aggregation("P@10")
def P10_score(items):
    """
    Calculate Precision at 10 (P@10) as an aggregation metric. 
    P@10 is the proportion of relevant items in the top 10 items for each query_id.    

    Args:
        items (list): A list of tuples, where each tuple contains:
            - doc (dict): Document information, including 'query_id'.
            - gold (int): The ground truth label.
            - pred (int): The predicted label.
            - prob_norm (list[float]): Normalized probability score for each multiple choice option.
    Returns:
        float: The mean P@10 score across all query_ids.
    """
    from sklearn.metrics import precision_score
    return score_per_query_id(items, score_function_fn=precision_score, cutoff_fn=lambda x: min(10, len(x)))

@register_metric(
    metric="P@10",
    higher_is_better=True,
    output_type=["multiple_choice"], #TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="P@10",
)
def P10_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("P@5")
def P5_score(items):
    from sklearn.metrics import precision_score
    return score_per_query_id(items, score_function_fn=precision_score, cutoff_fn=lambda x: min(5, len(x)))

@register_metric(
    metric="P@5",
    higher_is_better=True,
    output_type=["multiple_choice"], #TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="P@5",
)
def P5_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("P@15")
def P15_score(items):
    from sklearn.metrics import precision_score
    return score_per_query_id(items, score_function_fn=precision_score, cutoff_fn=lambda x: min(15, len(x)))

@register_metric(
    metric="P@15",
    higher_is_better=True,
    output_type=["multiple_choice"], #TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="P@15",
)
def P15_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("R-Prec")
def R_prec_score(items):
    """
    Calculate R-Precision (R-Prec) as an aggregation metric.
    R-Prec is the proportion of relevant items in the top R items for each query_id, where R is the number of relevant items for that query_id.   

    Args:
        items (list): A list of tuples, where each tuple contains:
            - doc (dict): Document information, including 'query_id'.
            - gold (int): The ground truth label.
            - pred (int): The predicted label.
            - prob_norm (list[float]): Normalized probability score for each multiple choice option.
    Returns:
        float: The mean R-Precision score across all query_ids.
    """
    from sklearn.metrics import precision_score
    def cutoff_fn(docs):
        return sum(1 for _, gold, *_ in docs if gold > 0)
    return score_per_query_id(items, score_function_fn=precision_score, cutoff_fn=cutoff_fn)

@register_metric(
    metric="R-Prec",
    higher_is_better=True,
    output_type=["multiple_choice"],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="R-Prec",
)
def R_prec_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("MAP")
def MAP_score(items):
    from sklearn.metrics import average_precision_score
    grouped = items_to_query_id_dict(items, pos_label_index=1 if len(items[0][3]) == 2 else [1,2])
    ap_scores = []
    for qid, docs in grouped.items():
        # Sort by descending model confidence score
        sorted_items = sorted(docs, key=lambda x: x[1], reverse=True)

        y_true = [1 if gold > 0 else 0 for _, gold, _, _ in sorted_items] 
        y_score = [score for _, _, _, score in sorted_items]

        ap = average_precision_score(y_true, y_score)
        ap_scores.append(ap)
        #print(f"[DEBUG] Query {qid} – AP: {ap}")
    return mean(ap_scores) if ap_scores else 0.0

@register_metric(
    metric="MAP",
    higher_is_better=True,
    output_type=["multiple_choice"],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="MAP",
)
def MAP_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

def calculate_nDCG(items, k=None):
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG) as an aggregation metric.
    nDCG evaluates the quality of the ranking of items based on predicted scores.

    Args:
        items (list): A list of tuples, where each tuple contains:
            - doc (dict): Document information, including 'query_id'.
            - gold (int): The ground truth label.
            - pred (int): The predicted label.
            - prob_norm (list[float]): Normalized probability score for each multiple choice option.
        k (int, optional): The cutoff rank for nDCG calculation. If None, uses all items.

    Returns:
        float: The mean nDCG score across all query_ids. If no items are provided, returns 0.0.
    """
    from sklearn.metrics import ndcg_score
    scores_by_query_id = defaultdict(list)
    pos_label_index = len(items[0][3]) - 1
    for doc, gold, _, prob_norm in items:    
        scores_by_query_id[doc["query_id"]].append((doc, gold, prob_norm[pos_label_index]))

    ndcg_scores = []

    for _, docs in scores_by_query_id.items():
        # Sort items by predicted score in descending order
        sorted_items = sorted(docs, key=lambda x: x[2], reverse=True)

        # Prepare y_true and y_score for nDCG calculation
        y_true = [[gold] for _, gold, _ in sorted_items]
        y_score = [[prob_norm] for _, _, prob_norm in sorted_items]

        # Failsafe: if only one doc or all gold labels are identical, skip sklearn and set nDCG = 1.0
        if len(y_true) < 2 or sum([g[0] for g in y_true]) < 2:
            ndcg_scores.append(0.0)
            continue

        ndcg_scores.append(ndcg_score(y_true, y_score, k=k))

        # Optional: Debugging output
        #print(f"[DEBUG] Query {qid} – nDCG: {ndcg_scores[-1]}")
        #for doc, gold, pred, prob in sorted_items:
        #    print(f"[DEBUG] doc_id: {doc['doc_id']}, Gold: {gold}, Pred: {pred}, Prob: {prob}")

    return mean(ndcg_scores) if ndcg_scores else 0.0

@register_aggregation("nDCG")
def nDCG_score(items):
    return calculate_nDCG(items)

@register_metric(
    metric="nDCG",
    higher_is_better=True,
    output_type=["multiple_choice"],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="nDCG",
)
def nDCG_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("nDCG@5")
def nDCG5_score(items):
    return calculate_nDCG(items, k=5)

@register_metric(
    metric="nDCG@5",
    higher_is_better=True,
    output_type=["multiple_choice"],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="nDCG@5",
)
def nDCG5_fn(items):  # This is a passthrough function
    return items

@register_aggregation("nDCG@10")
def nDCG10_score(items):
    return calculate_nDCG(items, k=10)

@register_metric(
    metric="nDCG@10",
    higher_is_better=True,
    output_type=["multiple_choice"],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="nDCG@10",
)
def nDCG10_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("RecRank")
def RecRank_score(items):
    def reciprocal_rank_fn(y_true, y_pred=None, zero_division=0):
        """Compute Reciprocal Rank: inverse of first relevant item's rank."""
        try:
            rank = next(i for i, v in enumerate(y_true, start=1) if v == 1)
            return 1.0 / rank
        except StopIteration:
            return 0.0
    return score_per_query_id(items, score_function_fn=reciprocal_rank_fn, cutoff_fn=None)

@register_metric(
    metric="RecRank",
    higher_is_better=True,
    output_type=["multiple_choice"],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="RecRank",
)
def RecRank_fn(items):  # This is a passthrough function
    return items

#-----------------------------------------------------------------------#

@register_aggregation("rouge_l")
def rouge_l(items):
    from rouge_score import rouge_scorer
    """
    Rouge-L is a metric for evaluating the quality of summaries by comparing them
    to reference summaries. It measures the longest common subsequence (LCS) between
    the generated summary and the reference summary, taking into account both precision
    and recall.

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Compute average Rouge-L F1 across all examples
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(refs, preds)
    ]
    return sum(scores) / len(scores) if scores else 0.0

@register_metric(
    metric="rouge_l",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="rouge_l",
)
def rouge_l_fn(items):  # This is a passthrough function
    return items


#-----------------------------------------------------------------------#

def parse_outcome_text(text: str) -> dict:
    lines = text.splitlines()
    outcome = {}
    current_section = None
    # pattern to capture “key: value” with optional whitespace
    kv_pattern = re.compile(r"^\s*([a-zA-Z_]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    # pattern to detect a top section
    section_pattern = re.compile(r"^\s*([a-zA-Z_]+)\s*:\s*$")

    for line in lines:
        if not line.strip():
            continue  # skip empty lines
        sec_m = section_pattern.match(line)
        if sec_m:
            # a new section like “intervention:” or “comparator:”
            sec = sec_m.group(1)
            current_section = sec
            if sec in outcome:
                # duplicate section – override or skip
                pass
            else:
                outcome[sec] = {}
        else:
            # must be a subfield line under current_section
            if current_section is None:
                # could not parse this line
                continue
            m = kv_pattern.match(line)
            if m:
                key = m.group(1)
                val_str = m.group(2)
                # convert to float or int
                if "." in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str)
                outcome[current_section][key] = val
            else:
                # line didn’t match numeric field; skip or warn
                # you could add fallback logic here
                pass

    return outcome

'''
def partial_numeric_match_from_texts(
    pred_texts: list[str],
    ref_texts: list[str],
    float_tolerance: float = 1,
    threshold_counts: list[int] = None
) -> dict:
    """
    pred_texts: list of free-text outcomes from model
    ref_texts: list of free-text outcomes from ground truth

    Returns metrics:
      - partial_match_frac: average fraction of numeric fields matched
      - and optionally partial_match_atleast_K for thresholds
    """
    n = len(pred_texts)
    if threshold_counts is None:
        threshold_counts = []

    # helper flatten as before
    def flatten(outcome: dict) -> dict:
        flat = {}
        for section, sub in outcome.items():
            for field, val in sub.items():
                flat[f"{section}.{field}"] = val
        return flat

    # accumulate scores
    frac_scores = []
    threshold_correct = {k: 0 for k in threshold_counts}

    for ptxt, rtxt in zip(pred_texts, ref_texts):
        p = parse_outcome_text(ptxt)
        r = parse_outcome_text(rtxt)
        pflat = flatten(p)
        rflat = flatten(r)

        keys = set(pflat.keys()) & set(rflat.keys())
        if not keys:
            frac_scores.append(0.0)
            continue

        matched = 0
        total = 0
        for k in keys:
            pv = pflat[k]
            rv = rflat[k]
            # float tolerance
            if isinstance(pv, float) or isinstance(rv, float):
                if abs(pv - rv) <= float_tolerance:
                    matched += 1
            else:
                if pv == rv:
                    matched += 1
            total += 1

        frac = matched / total if total > 0 else 0.0
        frac_scores.append(frac)

        for th in threshold_counts:
            if matched >= th:
                threshold_correct[th] += 1

    out = {"partial_match_frac": sum(frac_scores) / n}
    for th, cnt in threshold_correct.items():
        out[f"partial_match_atleast_{th}"] = cnt / n
    return out

@register_metric(
    metric="partial_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def partial_match_fn(**kwargs):
    return partial_numeric_match_from_texts(**kwargs)
'''
#-------------------------------------------------------#

# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("nanmean")
def nanmean(arr):
    if len(arr) == 0 or all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    from sklearn.metrics import f1_score
    
    unzipped_list = list(zip(*items))
    golds = [1 if g > 0 else 0 for g in unzipped_list[0]]
    preds = [1 if p > 0 else 0 for p in unzipped_list[1]]

    return f1_score(golds, preds, zero_division=0)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("brier_score")
def brier_score(items):
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr: Sequence[T]) -> float:
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return None


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type=["multiple_choice", "generate_until"],
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    """
    Pool worker: `(i, xs)` → `n` bootstrap replicates
    of `f(xs)`using a RNG seeded with `i`.
    """

    def __init__(self, f: Callable[[Sequence[T]], float], n: int) -> None:
        self.f = f
        self.n = n

    def __call__(self, v: tuple[int, Sequence[T]]) -> list[float]:
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def _bootstrap_internal_no_mp(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> list[float]:
    """
    Single-process fallback: compute `iters` bootstrap replicates
    of statistic`f(xs)`, chunked (≤ 1000 draws).
    """
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print(f"bootstrapping for stddev: {f.__name__}")

    # A single loop replaces the multiprocessing pool.
    for i in tqdm(range(iters // chunk_size)):
        rnd = random.Random(i)
        for _ in range(chunk_size):
            res.append(f(rnd.choices(xs, k=len(xs))))

    return res


def bootstrap_stderr(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> float:
    """
    Bootstrap estimate of the standard error of statistic `f(xs)`
    using up to `iters` resamples, chunked (≤ 1000 draws)

    Executes in parallel unless the env-var `DISABLE_MULTIPROC` is set;
    """
    if not os.getenv("DISABLE_MULTIPROC"):
        import multiprocessing as mp

        # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
        # equivalent to stderr calculated without Bessel's correction in the stddev.
        # Unfortunately, I haven't been able to figure out what the right correction is
        # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
        # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
        # Thankfully, shouldn't matter because our samples are pretty big usually anyways
        res = []
        chunk_size = min(1000, iters)
        from tqdm import tqdm

        print("bootstrapping for stddev:", f.__name__)
        with mp.Pool(mp.cpu_count()) as pool:
            for bootstrap in tqdm(
                pool.imap(
                    _bootstrap_internal(f, chunk_size),
                    [(i, xs) for i in range(iters // chunk_size)],
                ),
                total=iters // chunk_size,
            ):
                # sample w replacement
                res.extend(bootstrap)
    else:
        res = _bootstrap_internal_no_mp(f, xs, iters)

    return sample_stddev(res)


def stderr_for_metric(
    metric: Callable[[Sequence[T]], float], bootstrap_iters: int
) -> Optional[Callable[[Sequence[T]], float]]:
    """
    Return a function that estimates the standard error of `metric(xs)`.

    * If `bootstrap_iters > 0` and the metric is in the pre-approved
      bootstrappable list, use `bootstrap_stderr` with that many draws.
    * If the metric has a closed-form SE (e.g. `mean`, `acc_all`), use it.
    * Otherwise, return `None`.
    """

    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    bootstrappable = [
        median,
        matthews_corrcoef,
        perplexity,
        bleu,
        chrf,
        ter,
        nanmean,
        #f1_score,
        #P10_score, # TO:DO Custom Metrics Added
        #R_prec_score,
        #nDCG_score,
        #MAP_score,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert metrics is not None, (
        "Need to pass a list of each subtask's metric for this stderr aggregation"
    )
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = ((curr_score * curr_size) + (score * size)) / (
            curr_size + size
        )  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (
            curr_size + size - 1
        ) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (
            curr_score - score
        ) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)


#from lm_eval.api.registry import METRIC_AGGREGATION_REGISTRY, AGGREGATION_REGISTRY, HIGHER_IS_BETTER_REGISTRY, METRIC_REGISTRY
#print(f'METRIC_REGISTRY: {METRIC_REGISTRY}')
#print(f'HIGHER_IS_BETTER_REGISTRY: {HIGHER_IS_BETTER_REGISTRY}')
#print(f'AGGREGATION_REGISTRY: {AGGREGATION_REGISTRY}')
#print(f'METRIC_AGGREGATION_REGISTRY: {METRIC_AGGREGATION_REGISTRY}')