"""Payload adapters for OpenCTEval scoring; no evaluator or task imports."""

import re


_DOCUMENT_METRICS = (
    "acc_original",
    "acc_paraphrase",
    "f1_original",
    "f1_paraphrase",
    "faithfulness",
    "consistency",
    "augmentation_consistency",
)
_CLASSIFICATION_METRICS = ("Precision", "Recall")
_RANKING_METRICS = (
    "P@5",
    "P@10",
    "P@15",
    "R-Prec",
    "MAP",
    "nDCG",
    "nDCG@5",
    "nDCG@10",
    "RecRank",
)


def multiple_choice_metrics(metric_names, doc, gold, pred, probabilities):
    """Supply the original tuple contracts to document-aware aggregations."""
    return {
        **{
            name: (doc, gold, pred)
            for name in _DOCUMENT_METRICS
            if name in metric_names
        },
        **{
            name: (gold, pred)
            for name in _CLASSIFICATION_METRICS
            if name in metric_names
        },
        **{
            name: (doc, gold, pred, probabilities)
            for name in _RANKING_METRICS
            if name in metric_names
        },
    }


def normalize_text(
    s, ignore_case=False, ignore_punctuation=False, ignore_numbers=False
):
    if s is None:
        return s
    s = str(s).strip()
    if ignore_case:
        s = s.lower()
    if ignore_punctuation:
        # remove punctuation (keep alnum and whitespace and underscores)
        s = re.sub(r"[^\w\s]", " ", s)
    if ignore_numbers:
        s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
