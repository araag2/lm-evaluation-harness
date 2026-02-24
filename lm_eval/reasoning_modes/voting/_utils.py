"""
_utils.py — shared low-level utilities for all voting strategies.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
#  Score normalisation
# ──────────────────────────────────────────────

def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalise to [0, 1].  Returns uniform weights if all equal."""
    lo   = min(scores)
    span = max(scores) - lo
    if span < 1e-12:
        return [1.0 / len(scores)] * len(scores)
    return [(s - lo) / span for s in scores]


def normalize_ranks(scores: List[float], ascending: bool = False) -> List[float]:
    """
    Assign normalised rank scores based on *scores*.

    ascending=False  → highest score → rank 0 (best)
    ascending=True   → lowest score  → rank 0 (best)

    Returns values in [0, 1] where 1 = best rank.
    """
    n = len(scores)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: scores[i], reverse=not ascending)
    ranks = [0.0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = 1.0 - rank / (n - 1)
    return ranks


# ──────────────────────────────────────────────
#  Top-k selection
# ──────────────────────────────────────────────

def top_k_indices(scores: List[float], k: Optional[int]) -> List[int]:
    """Return the indices of the top-k entries by score (descending).
    If k is None or k >= len(scores), returns all indices sorted desc."""
    indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return indexed[:k] if (k is not None and k < len(scores)) else indexed


def apply_top_k(
    preds: List[int],
    pred_probs: List[List[float]],
    weights: List[float],
    k: Optional[int],
) -> Tuple[List[int], List[List[float]], List[float]]:
    """
    Select the top-k chains by *weights* and return the filtered
    (preds, pred_probs, weights) triple.  Pass k=None to keep all.
    """
    if k is None or k >= len(preds):
        return preds, pred_probs, weights
    indices = top_k_indices(weights, k)
    return (
        [preds[i]      for i in indices],
        [pred_probs[i] for i in indices],
        [weights[i]    for i in indices],
    )


# ──────────────────────────────────────────────
#  Weighted vote aggregation
# ──────────────────────────────────────────────

def weighted_vote(
    preds: List[int],
    pred_probs: List[List[float]],
    weights: List[float],
    doc_to_choice: List[str],
) -> Tuple[int, List[float]]:
    """
    Weighted majority vote.

    Accumulates ``weight * logit`` per class for the winning prediction.
    Returns ``(winning_class_index, normalised_prob_list)``.
    """
    class_scores: Dict[int, float] = {}
    for p, w in zip(preds, weights):
        class_scores[p] = class_scores.get(p, 0.0) + w
    winner = max(class_scores, key=class_scores.get)

    # Build a normalised probability vector for process_results
    pred_probs_agg = [0.0] * len(doc_to_choice)
    total_w        = sum(weights) or 1.0
    for p, probs, w in zip(preds, pred_probs, weights):
        if p == winner:
            for i, prob in enumerate(probs):
                pred_probs_agg[i] += prob * w
    normalised = [(v / total_w, False) for v in pred_probs_agg]
    return winner, normalised


# ──────────────────────────────────────────────
#  Pareto optimality
# ──────────────────────────────────────────────

def pareto_front_mask(metric_matrix: List[List[float]]) -> List[bool]:
    """
    Given *n* chains × *m* metrics (higher = better for all),
    return a boolean mask where True = chain is on the Pareto front
    (no other chain dominates it on every metric).
    """
    n = len(metric_matrix)
    on_front = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j or not on_front[j]:
                continue
            # j dominates i if it's >= on all metrics and strictly > on at least one
            if all(metric_matrix[j][m] >= metric_matrix[i][m]
                   for m in range(len(metric_matrix[i]))):
                if any(metric_matrix[j][m] > metric_matrix[i][m]
                       for m in range(len(metric_matrix[i]))):
                    on_front[i] = False
                    break
    return on_front


# ──────────────────────────────────────────────
#  Entropy
# ──────────────────────────────────────────────

def entropy(probs: List[float]) -> float:
    """Shannon entropy (nats) of a probability distribution."""
    return -sum(p * math.log(max(p, 0) + 1e-12) for p in probs)
