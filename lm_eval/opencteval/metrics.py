"""OpenCTEval metrics, separated from upstream metrics without changing scores.

The core module imports this module after registering standard aggregations.
Legacy imports from lm_eval.api.metrics remain supported.
"""

__all__ = [
    "_binary_label",
    "_normalize_class_label",
    "_prepare_classification_labels",
    "items_to_query_id_dict",
    "score_per_query_id",
    "P10_score",
    "P10_fn",
    "P5_score",
    "P5_fn",
    "P15_score",
    "P15_fn",
    "R_prec_score",
    "R_prec_fn",
    "Precision_score",
    "Precision_fn",
    "Recall_score",
    "Recall_fn",
    "MAP_score",
    "MAP_fn",
    "PR_AUC_score",
    "PR_AUC_fn",
    "ROC_AUC_score",
    "ROC_AUC_fn",
    "calculate_nDCG",
    "nDCG_score",
    "nDCG_fn",
    "nDCG5_score",
    "nDCG5_fn",
    "nDCG10_score",
    "nDCG10_fn",
    "RecRank_score",
    "RecRank_fn",
    "extract_numeric_value",
    "rouge_l",
    "rouge_l_fn",
    "_normalize_regression_pairs",
    "_valid_regression_errors",
    "mean_absolute_error_fn",
    "mean_squared_error_fn",
    "parse_outcome_text",
    "partial_numeric_match_from_texts",
    "partial_match_fn",
    "normalize_span",
    "span_prf_single",
    "span_precision_agg",
    "span_recall_agg",
    "span_f1_agg",
    "span_precision_fn",
    "span_recall_fn",
    "span_f1_fn",
    "filter_by_id",
    "_base_intervention_id",
    "_item_identifier",
    "_paired_intervention_scores",
    "acc_original_agg",
    "acc_original_fn",
    "acc_paraphrase_agg",
    "acc_paraphrase_fn",
    "f1_original_agg",
    "f1_original_fn",
    "f1_paraphrase_agg",
    "f1_paraphrase_fn",
    "faithfulness_agg",
    "faithfulness_fn",
    "augmentation_consistency_agg",
    "augmentation_consistency_fn",
    "consistency_agg",
    "consistency_fn",
]

import logging
import numbers
import re
import sys
from collections import defaultdict

import numpy as np

from lm_eval.api.metrics import mean
from lm_eval.api.registry import register_aggregation, register_metric


eval_logger = logging.getLogger(__name__)


def _binary_label(value):
    """Coerce labels/predictions into 0/1 for binary aggregations.

    Handles ints/floats, numeric strings, and common textual labels such as
    included/excluded, yes/no, and relevance judgments.
    """
    # Normalize numpy scalar types (e.g., np.int64, np.float32, np.bool_)
    # into native Python scalars so type comparisons remain consistent.
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return 1 if value > 0 else 0

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {
            "1",
            "true",
            "yes",
            "y",
            "included",
            "include",
            "positive",
            "pos",
            "entailment",
            "entailed",
            "possibly relevant",
            "definitely relevant",
        }:
            return 1
        if v in {
            "0",
            "false",
            "no",
            "n",
            "excluded",
            "exclude",
            "negative",
            "neg",
            "contradiction",
            "contradicted",
            "not relevant",
        }:
            return 0
        print(
            f"[WARNING] Unrecognized binary label '{value}' (normalized: '{v}'), defaulting to 0",
            file=sys.stderr,
        )
        try:
            return 1 if float(v) > 0 else 0
        except ValueError:
            return 0

    return 0


def _normalize_class_label(value):
    """Normalize labels to a stable representation for class counting/scoring."""
    # Normalize numpy scalar types (e.g., np.int64, np.float32, np.bool_)
    # into native Python scalars so type comparisons remain consistent.
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, numbers.Integral):
        return value

    if isinstance(value, numbers.Real):
        return int(value) if float(value).is_integer() else value

    if isinstance(value, str):
        v = value.strip().lower()
        if not v:
            return v

        # Try numeric labels encoded as strings first.
        try:
            return int(v)
        except ValueError:
            pass

        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v

    return str(value).strip().lower()


def _prepare_classification_labels(items):
    """Return normalized labels plus task type flag for classification metrics.

    Returns:
        (golds, preds, is_multiclass)
    """
    unzipped_list = list(zip(*items, strict=False))
    golds_raw = list(unzipped_list[0])
    preds_raw = list(unzipped_list[1])

    golds_norm = [_normalize_class_label(g) for g in golds_raw]
    preds_norm = [_normalize_class_label(p) for p in preds_raw]

    n_classes = len(set(golds_norm) | set(preds_norm))
    is_multiclass = n_classes > 2

    # Fail fast on ambiguous multiclass mixtures (e.g., 1 vs "entailment").
    # Binary tasks are handled by _binary_label with explicit coercion rules.
    if is_multiclass:
        has_text = any(isinstance(x, str) for x in (golds_norm + preds_norm))
        has_numeric = any(
            isinstance(x, (int, float)) for x in (golds_norm + preds_norm)
        )
        if has_text and has_numeric:
            # Include compact examples to make debugging task-specific label
            # normalization issues straightforward.
            gold_examples_raw = list(dict.fromkeys(repr(g) for g in golds_raw))[:5]
            pred_examples_raw = list(dict.fromkeys(repr(p) for p in preds_raw))[:5]
            gold_examples_norm = list(dict.fromkeys(repr(g) for g in golds_norm))[:5]
            pred_examples_norm = list(dict.fromkeys(repr(p) for p in preds_norm))[:5]
            eval_logger.error(
                "Ambiguous multiclass labels detected. "
                "gold_raw=%s pred_raw=%s gold_norm=%s pred_norm=%s",
                gold_examples_raw,
                pred_examples_raw,
                gold_examples_norm,
                pred_examples_norm,
            )
            raise ValueError(
                "Ambiguous multiclass labels: found mixed numeric and text labels "
                "(e.g., 1 vs 'entailment'). Please normalize task labels/predictions "
                "to one representation before metric aggregation. "
                f"Examples -> gold_raw={gold_examples_raw}, pred_raw={pred_examples_raw}, "
                f"gold_norm={gold_examples_norm}, pred_norm={pred_examples_norm}"
            )

    if is_multiclass:
        return golds_norm, preds_norm, True

    # Binary path keeps legacy behavior (explicit positive-vs-negative collapse).
    return (
        [_binary_label(g) for g in golds_norm],
        [_binary_label(p) for p in preds_norm],
        False,
    )


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
        score = (
            prob_norm[pos_label_index]
            if isinstance(pos_label_index, int)
            else sum(prob_norm[i] for i in pos_label_index)
        )
        scores_by_query_id[doc["query_id"]].append((doc, gold, pred, score))
    return scores_by_query_id


def score_per_query_id(items, score_function_fn, cutoff_fn=None):
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
    grouped = items_to_query_id_dict(
        items, pos_label_index=len(items[0][3]) - 1
    )  # <query_id, (doc, gold, pred, prob_norm[pos_label_index])>
    scores = []

    # print(f"[DEBUG] Using function {score_function_fn.__name__} with cutoff {cutoff_fn.__name__ if cutoff_fn else 'None'}")
    # print(f"[DEBUG] Total queries: {len(grouped)}, Total items: {len(items)}")

    for _qid, docs in grouped.items():
        sorted_items = sorted(docs, key=lambda x: x[3], reverse=True)

        if cutoff_fn:
            sorted_items = sorted_items[: cutoff_fn(sorted_items)]

        # In TREC, 1 (maybe) or 2 (yes) are consolidated the positive labels
        y_true = [_binary_label(gold) for _, gold, _, _ in sorted_items]
        y_pred = [_binary_label(pred) for _, _, pred, _ in sorted_items]

        scores.append(score_function_fn(y_true, y_pred, zero_division=0))

        # TO:DO Optional debug prints (remove later)
        # print(f"[DEBUG] Query {qid} – y_true: {y_true}, y_pred: {y_pred}")
        # for doc, gold, pred, prob in sorted_items:
        #    print(f"[DEBUG] doc_id: {doc['doc_id']}, Gold: {gold}, Pred: {pred}, Prob: {prob}")
        # print(f"[DEBUG] Query {qid} – Score: {scores[-1]}")

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

    return score_per_query_id(
        items, score_function_fn=precision_score, cutoff_fn=lambda x: min(10, len(x))
    )


@register_metric(
    metric="P@10",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="P@10",
)
def P10_fn(items):  # This is a passthrough function
    return items


@register_aggregation("P@5")
def P5_score(items):
    from sklearn.metrics import precision_score

    return score_per_query_id(
        items, score_function_fn=precision_score, cutoff_fn=lambda x: min(5, len(x))
    )


@register_metric(
    metric="P@5",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="P@5",
)
def P5_fn(items):  # This is a passthrough function
    return items


@register_aggregation("P@15")
def P15_score(items):
    from sklearn.metrics import precision_score

    return score_per_query_id(
        items, score_function_fn=precision_score, cutoff_fn=lambda x: min(15, len(x))
    )


@register_metric(
    metric="P@15",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="P@15",
)
def P15_fn(items):  # This is a passthrough function
    return items


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
        return sum(1 for _, gold, *_ in docs if _binary_label(gold) > 0)

    return score_per_query_id(
        items, score_function_fn=precision_score, cutoff_fn=cutoff_fn
    )


@register_metric(
    metric="R-Prec",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="R-Prec",
)
def R_prec_fn(items):  # This is a passthrough function
    return items


@register_aggregation("Precision")
def Precision_score(items):
    from sklearn.metrics import precision_score

    golds, preds, is_multiclass = _prepare_classification_labels(items)

    if is_multiclass:
        return precision_score(golds, preds, average="macro", zero_division=0)
    return precision_score(golds, preds, zero_division=0)


@register_metric(
    metric="Precision",
    higher_is_better=True,
    output_type=[
        "generate_until",
        "multiple_choice",
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="Precision",
)
def Precision_fn(items):  # This is a passthrough function
    return items


@register_aggregation("Recall")
def Recall_score(items):
    from sklearn.metrics import recall_score

    golds, preds, is_multiclass = _prepare_classification_labels(items)

    if is_multiclass:
        return recall_score(golds, preds, average="macro", zero_division=0)
    return recall_score(golds, preds, zero_division=0)


@register_metric(
    metric="Recall",
    higher_is_better=True,
    output_type=[
        "generate_until",
        "multiple_choice",
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="Recall",
)
def Recall_fn(items):  # This is a passthrough function
    return items


@register_aggregation("MAP")
def MAP_score(items):
    from sklearn.metrics import average_precision_score

    grouped = items_to_query_id_dict(
        items, pos_label_index=1 if len(items[0][3]) == 2 else [1, 2]
    )
    ap_scores = []
    for _qid, docs in grouped.items():
        # Sort by descending model confidence score
        sorted_items = sorted(docs, key=lambda x: x[1], reverse=True)

        y_true = [_binary_label(gold) for _, gold, _, _ in sorted_items]
        y_score = [score for _, _, _, score in sorted_items]

        ap = average_precision_score(y_true, y_score)
        ap_scores.append(ap)
        # print(f"[DEBUG] Query {qid} – AP: {ap}")
    return mean(ap_scores) if ap_scores else 0.0


@register_metric(
    metric="MAP",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="MAP",
)
def MAP_fn(items):  # This is a passthrough function
    return items


@register_aggregation("PR-AUC")
def PR_AUC_score(items):
    """
    Calculate Precision-Recall Area Under Curve (PR-AUC) as an aggregation metric.
    PR-AUC measures the area under the precision-recall curve, useful for imbalanced datasets.

    Args:
        items (list): A list of tuples, where each tuple contains:
            - doc (dict): Document information.
            - gold (int): The ground truth label.
            - pred (int): The predicted label.
            - prob_norm (list[float]): Normalized probability score for each multiple choice option.

    Returns:
        float: The PR-AUC score. If computation fails, returns 0.0.
    """
    from sklearn.metrics import auc, precision_recall_curve

    # Extract labels and scores
    # Assuming the last label in available choices is the positive label
    pos_label_index = len(items[0][3]) - 1

    y_true = []
    y_scores = []

    for _doc, gold, _pred, prob_norm in items:
        # Binary classification: 1 if positive class, 0 otherwise
        y_true.append(1 if gold > 0 else 0)
        y_scores.append(prob_norm[pos_label_index])

    # Check if we have both classes
    if len(set(y_true)) < 2:
        # If only one class present, PR-AUC is not well-defined
        return 0.0

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Calculate area under the precision-recall curve
    pr_auc = auc(recall, precision)

    return pr_auc


@register_metric(
    metric="PR-AUC",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="PR-AUC",
)
def PR_AUC_fn(items):  # This is a passthrough function
    return items


@register_aggregation("ROC-AUC")
def ROC_AUC_score(items):
    """
    Calculate Receiver Operating Characteristic Area Under Curve (ROC-AUC) as an aggregation metric.
    ROC-AUC measures the area under the ROC curve, indicating the model's ability to distinguish between classes.

    Args:
        items (list): A list of tuples, where each tuple contains:
            - doc (dict): Document information.
            - gold (int): The ground truth label.
            - pred (int): The predicted label.
            - prob_norm (list[float]): Normalized probability score for each multiple choice option.

    Returns:
        float: The ROC-AUC score. If computation fails, returns 0.0.
    """
    from sklearn.metrics import roc_auc_score

    # Extract labels and scores
    # Assuming the last label in available choices is the positive label
    pos_label_index = len(items[0][3]) - 1

    y_true = []
    y_scores = []

    for _doc, gold, _pred, prob_norm in items:
        # Binary classification: 1 if positive class, 0 otherwise
        y_true.append(1 if gold > 0 else 0)
        y_scores.append(prob_norm[pos_label_index])

    # Check if we have both classes
    if len(set(y_true)) < 2:
        # If only one class present, ROC-AUC is not well-defined
        return 0.0

    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle any edge cases
        return 0.0

    return roc_auc


@register_metric(
    metric="ROC-AUC",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="ROC-AUC",
)
def ROC_AUC_fn(items):  # This is a passthrough function
    return items


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
        scores_by_query_id[doc["query_id"]].append((gold, prob_norm[pos_label_index]))

    ndcg_scores = []
    for _, docs in scores_by_query_id.items():
        if len(docs) < 2:
            ndcg_scores.append(0.0)
            continue

        golds, scores = zip(*docs, strict=False)
        y_true = [list(golds)]
        y_score = [list(scores)]

        if sum(y_true[0]) < 1:
            ndcg_scores.append(0.0)

        else:
            ndcg_scores.append(ndcg_score(y_true, y_score, k=k))
    return mean(ndcg_scores) if ndcg_scores else 0.0


@register_aggregation("nDCG")
def nDCG_score(items):
    return calculate_nDCG(items)


@register_metric(
    metric="nDCG",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="nDCG",
)
def nDCG_fn(items):  # This is a passthrough function
    return items


@register_aggregation("nDCG@5")
def nDCG5_score(items):
    return calculate_nDCG(items, k=5)


@register_metric(
    metric="nDCG@5",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
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
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="nDCG@10",
)
def nDCG10_fn(items):  # This is a passthrough function
    return items


@register_aggregation("RecRank")
def RecRank_score(items):
    def reciprocal_rank_fn(y_true, y_pred=None, zero_division=0):
        """Compute Reciprocal Rank: inverse of first relevant item's rank."""
        try:
            rank = next(i for i, v in enumerate(y_true, start=1) if v == 1)
            return 1.0 / rank
        except StopIteration:
            return 0.0

    return score_per_query_id(
        items, score_function_fn=reciprocal_rank_fn, cutoff_fn=None
    )


@register_metric(
    metric="RecRank",
    higher_is_better=True,
    output_type=[
        "multiple_choice"
    ],  # TO:DO to implement to other types, need to set inputs in api.task.py
    aggregation="RecRank",
)
def RecRank_fn(items):  # This is a passthrough function
    return items


def extract_numeric_value(text):
    """
    Extract a numeric value from text. Attempts to find and parse the first
    numeric value (integer or float) in the given text.

    Args:
        text: String that may contain a numeric value

    Returns:
        float: The extracted numeric value, or np.nan if extraction fails
    """
    if isinstance(text, (int, float)):
        return float(text)

    if not isinstance(text, str):
        return np.nan

    # Remove common text prefixes/suffixes and extract number
    text = text.strip()

    # Try to match a number (including decimals and negatives)
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except (ValueError, AttributeError):
            return np.nan

    return np.nan


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
    refs = list(zip(*items, strict=False))[0]
    preds = list(zip(*items, strict=False))[1]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Compute average Rouge-L F1 across all examples
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(refs, preds, strict=False)
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


def _normalize_regression_pairs(items=None, references=None, predictions=None):
    pairs = []

    if references is not None or predictions is not None:
        refs = references if references is not None else []
        preds = predictions if predictions is not None else []

        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        pairs = list(zip(refs, preds, strict=False))

    elif items is not None:
        if isinstance(items, (list, tuple)):
            if (
                len(items) == 2
                and not isinstance(items[0], (list, tuple))
                and not isinstance(items[1], (list, tuple))
            ):
                pairs = [(items[0], items[1])]
            else:
                pairs = [
                    (item[0], item[1])
                    for item in items
                    if isinstance(item, (list, tuple)) and len(item) >= 2
                ]

    return pairs


def _valid_regression_errors(pairs, squared=False):
    errors = []

    for ref, pred in pairs:
        ref_val = extract_numeric_value(ref)
        pred_val = extract_numeric_value(pred)

        if not np.isnan(ref_val) and not np.isnan(pred_val):
            diff = pred_val - ref_val
            errors.append(diff**2 if squared else abs(diff))

    return errors


@register_metric(
    metric="mean_absolute_error",
    higher_is_better=False,
    output_type="generate_until",
    aggregation="mean",
)
def mean_absolute_error_fn(items=None, references=None, predictions=None, **kwargs):
    """
    Calculate Mean Absolute Error (MAE) for regression tasks.
    MAE measures the average magnitude of errors between predictions and actual values.

    Args:
        items: List of tuples (gold, pred) where gold is the true value and
               pred is the predicted value (both may be strings or numeric)

    Returns:
        Mean absolute error across valid extracted numeric pairs
    """
    pairs = _normalize_regression_pairs(
        items=items, references=references, predictions=predictions
    )
    errors = _valid_regression_errors(pairs, squared=False)
    return float(np.mean(errors)) if errors else 0.0


@register_metric(
    metric="mean_squared_error",
    higher_is_better=False,
    output_type="generate_until",
    aggregation="mean",
)
def mean_squared_error_fn(items=None, references=None, predictions=None, **kwargs):
    """
    Calculate Mean Squared Error (MSE) for regression tasks.
    MSE measures the average of squared errors between predictions and actual values.

    Args:
        items: List of tuples (gold, pred) where gold is the true value and
               pred is the predicted value (both may be strings or numeric)

    Returns:
        Mean squared error across valid extracted numeric pairs
    """
    pairs = _normalize_regression_pairs(
        items=items, references=references, predictions=predictions
    )
    errors = _valid_regression_errors(pairs, squared=True)
    return float(np.mean(errors)) if errors else 0.0


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
                val = float(val_str) if "." in val_str else int(val_str)
                outcome[current_section][key] = val
            else:
                # line didn’t match numeric field; skip or warn
                # you could add fallback logic here
                pass

    return outcome


def partial_numeric_match_from_texts(
    ref_texts: list[str],
    pred_texts: list[str],
    float_tolerance: float = 1,
    threshold_counts: list[int] = None,
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

    for ptxt, rtxt in zip(pred_texts, ref_texts, strict=False):
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
def partial_match_fn(references, predictions, **kwargs):
    return partial_numeric_match_from_texts(references, predictions)


def normalize_span(s: str):
    return " ".join(s.lower().strip().split())


def span_prf_single(
    ref_texts: list[str],
    pred_texts: list[str],
) -> dict:
    gold_spans = {normalize_span(s) for s in ref_texts.split("\n") if s.strip()}

    pred_spans = {normalize_span(s) for s in pred_texts.split("\n") if s.strip()}

    tp = len(gold_spans & pred_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)

    return tp, fp, fn


@register_aggregation("span_precision")
def span_precision_agg(items):
    total_tp = total_fp = total_fn = 0

    for ref, pred in items:
        tp, fp, fn = span_prf_single(ref, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0


@register_aggregation("span_recall")
def span_recall_agg(items):
    total_tp = total_fp = total_fn = 0

    for ref, pred in items:
        tp, fp, fn = span_prf_single(ref, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0


@register_aggregation("span_f1")
def span_f1_agg(items):
    total_tp = total_fp = total_fn = 0

    for ref, pred in items:
        tp, fp, fn = span_prf_single(ref, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )


@register_metric(
    metric="span_precision",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="span_precision",
)
def span_precision_fn(items):
    return items


@register_metric(
    metric="span_recall",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="span_recall",
)
def span_recall_fn(items):
    return items


@register_metric(
    metric="span_f1",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="span_f1",
)
def span_f1_fn(items):
    return items


def filter_by_id(items, predicate):
    return [item for item in items if predicate(_item_identifier(item))]


def _base_intervention_id(item_id: str) -> str:
    return re.sub(
        r"_(?:paraphrase|contradiction)(?:\d+|_[A-Za-z0-9_-]+)?$", "", item_id
    )


def _item_identifier(item) -> str:
    doc = item[0]
    return str(doc.get("query_id", doc.get("id")))


def _paired_intervention_scores(items, variant_suffix: str, score_fn):
    originals = {}
    variant_items = []

    for item in items:
        item_id = _item_identifier(item)
        base_id = _base_intervention_id(item_id)

        if base_id == item_id:
            originals[base_id] = item
        elif variant_suffix in item_id:
            variant_items.append((base_id, item))

    scores = []
    for base_id, variant_item in variant_items:
        original_item = originals.get(base_id)
        if original_item is None:
            continue
        scores.append(score_fn(original_item, variant_item))

    return mean(scores) if scores else 0.0


@register_aggregation("acc_original")
def acc_original_agg(items):
    items = filter_by_id(items, lambda qid: "_paraphrase" not in qid)
    golds = [item[1] for item in items]
    preds = [item[2] for item in items]
    return (
        sum(g == p for g, p in zip(golds, preds, strict=False)) / len(golds)
        if golds
        else 0.0
    )


@register_metric(
    metric="acc_original",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="acc_original",
)
def acc_original_fn(items):  # This is a passthrough function
    return items


@register_aggregation("acc_paraphrase")
def acc_paraphrase_agg(items):
    items = filter_by_id(items, lambda qid: "_paraphrase" in qid)
    golds = [item[1] for item in items]
    preds = [item[2] for item in items]
    return (
        sum(g == p for g, p in zip(golds, preds, strict=False)) / len(golds)
        if golds
        else 0.0
    )


@register_metric(
    metric="acc_paraphrase",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="acc_paraphrase",
)
def acc_paraphrase_fn(items):  # This is a passthrough function
    return items


@register_aggregation("f1_original")
def f1_original_agg(items):
    from sklearn.metrics import f1_score

    items = [
        item[1:] for item in filter_by_id(items, lambda qid: "_paraphrase" not in qid)
    ]
    golds, preds, is_multiclass = _prepare_classification_labels(items)

    if is_multiclass:
        return f1_score(golds, preds, average="macro", zero_division=0)
    return f1_score(golds, preds, zero_division=0)


@register_metric(
    metric="f1_original",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="f1_original",
)
def f1_original_fn(items):  # This is a passthrough function
    return items


@register_aggregation("f1_paraphrase")
def f1_paraphrase_agg(items):
    from sklearn.metrics import f1_score

    items = [item[1:] for item in filter_by_id(items, lambda qid: "_paraphrase" in qid)]
    golds, preds, is_multiclass = _prepare_classification_labels(items)

    if is_multiclass:
        return f1_score(golds, preds, average="macro", zero_division=0)
    return f1_score(golds, preds, zero_division=0)


@register_metric(
    metric="f1_paraphrase",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="f1_paraphrase",
)
def f1_paraphrase_fn(items):  # This is a passthrough function
    return items


@register_aggregation("faithfulness")
def faithfulness_agg(items):
    def _faithfulness_score(original_item, variant_item):
        original_gold = original_item[1]
        variant_pred = variant_item[2]
        return 1 if variant_pred != original_gold else 0

    return _paired_intervention_scores(items, "_contradiction", _faithfulness_score)


@register_metric(
    metric="faithfulness",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="faithfulness",
)
def faithfulness_fn(items):  # This is a passthrough function
    return items


@register_aggregation("augmentation_consistency")
def augmentation_consistency_agg(items):
    """Pair preserving MCQA variants by source ID and stable option identity.

    Label-changing/task-changing interventions are evaluated by accuracy, not
    prediction invariance. This is pairwise consistency, not all-variants ReCon.
    """
    originals = {
        str(item[0]["source_id"]): item
        for item in items
        if item[0].get("augmentation", {}).get("mode") == "original"
    }

    def selected_id(item):
        ids = item[0].get("augmentation", {}).get("option_ids", [])
        prediction = item[2]
        if isinstance(prediction, numbers.Integral) and 0 <= prediction < len(ids):
            return ids[prediction]
        return None

    scores = []
    for item in items:
        metadata = item[0].get("augmentation", {})
        if (
            metadata.get("effect") != "preserving"
            or metadata.get("consistency_eligible") is not True
        ):
            continue
        original = originals.get(str(item[0].get("source_id")))
        if original is None:
            continue
        original_selection = selected_id(original)
        scores.append(
            int(
                original_selection is not None
                and original_selection == selected_id(item)
            )
        )
    return mean(scores) if scores else 0.0


@register_metric(
    metric="augmentation_consistency",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="augmentation_consistency",
)
def augmentation_consistency_fn(items):
    return items


@register_aggregation("consistency")
def consistency_agg(items):
    def _consistency_score(original_item, variant_item):
        original_pred = original_item[2]
        variant_pred = variant_item[2]
        return 1 if variant_pred == original_pred else 0

    return _paired_intervention_scores(items, "_paraphrase", _consistency_score)


@register_metric(
    metric="consistency",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="consistency",
)
def consistency_fn(items):  # This is a passthrough function
    return items
