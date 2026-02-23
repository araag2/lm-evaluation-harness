"""
strategies/chain_metrics.py — text-quality-weighted voting strategies.

Each chain is scored on surface-level text quality features (word count,
readability).  Those per-chain scores are combined into a weight vector
using one of three combination methods, then used for a weighted vote.

Registered strategies
---------------------
chain_metrics_mean        — mean of normalised metric scores
chain_metrics_rank        — mean of normalised rank scores
chain_metrics_pareto      — uniform weight to Pareto-optimal chains only

Top-k variants are automatically registered for k ∈ TOP_K_VALUES.

Each strategy reads chains from ``info["doc"]["Reasoning_Chains"]``
(populated by cross_consistency Step 3 or multi-turn SC).

Dependencies
------------
    pip install textstat
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:
    import textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _TEXTSTAT_AVAILABLE = False

from lm_eval.reasoning_modes.voting._registry import register, aggregate_metrics_per_strategy
from lm_eval.reasoning_modes.voting._utils import (
    normalize_scores,
    normalize_ranks,
    pareto_front_mask,
    weighted_vote,
    apply_top_k,
)

# Top-k values to auto-register (None = all chains)
TOP_K_VALUES: List[Optional[int]] = [None, 5, 10]

# Which metrics are "higher is better" (ascending=False for rank normalisation)
_METRIC_ASCENDING: Dict[str, bool] = {
    "word_count":           False,   # longer chains are generally richer
    "flesch_reading_ease":  False,   # higher ease = more readable
    "flesch_kincaid_grade": True,    # lower grade level = simpler (penalise over-complex)
    "avg_sentence_length":  True,    # lower = more digestible
}


# ──────────────────────────────────────────────
#  Chain metric computation
# ──────────────────────────────────────────────

def compute_chain_metrics(chain: str) -> Dict[str, float]:
    """Compute surface-level quality metrics for a single reasoning chain."""
    words = chain.split()
    sentences = [s.strip() for s in chain.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    avg_sent_len = len(words) / max(len(sentences), 1)

    metrics: Dict[str, float] = {
        "word_count":          float(len(words)),
        "avg_sentence_length": avg_sent_len,
    }

    if _TEXTSTAT_AVAILABLE:
        metrics["flesch_reading_ease"]  = float(textstat.flesch_reading_ease(chain))
        metrics["flesch_kincaid_grade"] = float(textstat.flesch_kincaid_grade(chain))
    else:
        # Fallback: rough Flesch approximation (206.835 - 1.015*ASL - 84.6*ASW)
        syllables_per_word = sum(
            max(1, sum(1 for ch in w.lower() if ch in "aeiou"))
            for w in words
        ) / max(len(words), 1)
        metrics["flesch_reading_ease"]  = 206.835 - 1.015 * avg_sent_len - 84.6 * syllables_per_word
        metrics["flesch_kincaid_grade"] = 0.39 * avg_sent_len + 11.8 * syllables_per_word - 15.59

    return metrics


def _score_chains(chains: List[str]) -> List[Dict[str, float]]:
    """Return a list of metric dicts, one per chain."""
    return [compute_chain_metrics(c) for c in chains]


# ──────────────────────────────────────────────
#  Weight computation methods
# ──────────────────────────────────────────────

def _mean_weights(all_metrics: List[Dict[str, float]]) -> List[float]:
    """
    For each chain, average the min-max-normalised scores across all metrics.
    Metrics where ascending=True are inverted before normalising.
    """
    metric_names = list(all_metrics[0].keys())
    per_metric_scores: Dict[str, List[float]] = {
        m: [entry[m] for entry in all_metrics] for m in metric_names
    }
    normalised: List[List[float]] = []
    for m, scores in per_metric_scores.items():
        # Invert metric if lower is better before normalising (so higher always = better)
        if _METRIC_ASCENDING.get(m, False):
            scores = [-s for s in scores]
        normalised.append(normalize_scores(scores))

    n = len(all_metrics)
    return [
        sum(normalised[mi][ci] for mi in range(len(metric_names))) / len(metric_names)
        for ci in range(n)
    ]


def _rank_weights(all_metrics: List[Dict[str, float]]) -> List[float]:
    """
    Average normalised rank across metrics.
    ascending=True metrics are ranked so that the lowest value = best rank.
    """
    metric_names = list(all_metrics[0].keys())
    per_metric_scores: Dict[str, List[float]] = {
        m: [entry[m] for entry in all_metrics] for m in metric_names
    }
    rank_vecs: List[List[float]] = []
    for m, scores in per_metric_scores.items():
        rank_vecs.append(
            normalize_ranks(scores, ascending=_METRIC_ASCENDING.get(m, False))
        )

    n = len(all_metrics)
    return [
        sum(rank_vecs[mi][ci] for mi in range(len(metric_names))) / len(metric_names)
        for ci in range(n)
    ]


def _pareto_weights(all_metrics: List[Dict[str, float]]) -> List[float]:
    """
    Assign weight 1.0 to Pareto-optimal chains, 0.0 to dominated ones.
    All metrics used as "higher is better" (ascending metrics are inverted first).
    """
    metric_names = list(all_metrics[0].keys())
    # Build matrix (n_chains × n_metrics) with all metrics in "higher=better" space
    matrix = [
        [
            -entry[m] if _METRIC_ASCENDING.get(m, False) else entry[m]
            for m in metric_names
        ]
        for entry in all_metrics
    ]
    mask = pareto_front_mask(matrix)
    n_front = max(sum(mask), 1)
    return [1.0 / n_front if on_front else 0.0 for on_front in mask]


# ──────────────────────────────────────────────
#  Generic chain-metrics voting step
# ──────────────────────────────────────────────

def _chain_metrics_vote(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
    weight_fn,
    k: Optional[int],
    strategy_name: str,
) -> Dict:
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        chains = info["doc"].get("Reasoning_Chains", [])

        # If chains are missing, degrade gracefully to uniform weights
        if not chains or len(chains) != len(info["preds"]):
            weights = [1.0] * len(info["preds"])
        else:
            all_metrics = _score_chains(chains)
            weights     = weight_fn(all_metrics)

        preds_k, probs_k, weights_k = apply_top_k(
            info["preds"], info["pred_probs"], weights, k
        )

        if not preds_k:
            preds_k, probs_k, weights_k = info["preds"], info["pred_probs"], weights

        winner, normalised = weighted_vote(preds_k, probs_k, weights_k, doc_to_choice)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)

    return results_per_doc_id


# ──────────────────────────────────────────────
#  Auto-register all (method × top_k) combos
# ──────────────────────────────────────────────

_METHODS = {
    "mean":   _mean_weights,
    "rank":   _rank_weights,
    "pareto": _pareto_weights,
}

for _method_name, _weight_fn in _METHODS.items():
    for _k in TOP_K_VALUES:
        _suffix        = "" if _k is None else f"_top{_k}"
        _strategy_name = f"chain_metrics_{_method_name}{_suffix}"

        # Closure captures _weight_fn, _k, _strategy_name
        def _make_fn(wfn, top_k, sname):
            @register(sname)
            def _fn(predictions_per_input_doc, doc_to_choice, task_def,
                    k=top_k, strategy_name=sname):
                return _chain_metrics_vote(
                    predictions_per_input_doc, doc_to_choice, task_def,
                    wfn, k, strategy_name,
                )
            _fn.__name__ = sname
            return _fn

        _make_fn(_weight_fn, _k, _strategy_name)
