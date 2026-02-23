"""
strategies/simple.py — count-based and rank-based voting strategies.

Registered strategies
---------------------
majority   — plurality vote on pred indices
logits     — sum raw logits per class, pick argmax
condorcet  — pairwise head-to-head winner (falls back to majority)
borda      — Borda count on per-chain probability rankings
rrf        — Reciprocal Rank Fusion

All functions share the signature:

    fn(predictions_per_input_doc, doc_to_choice, task_def,
       k=None, strategy_name=<name>) -> results_per_doc

and are automatically discovered via the _STRATEGY_REGISTRY.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from lm_eval.reasoning_modes.voting._registry import register, aggregate_metrics_per_strategy


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def majority_vote(predictions: List[int]) -> int:
    return Counter(predictions).most_common(1)[0][0]


# ──────────────────────────────────────────────
#  Strategies
# ──────────────────────────────────────────────

@register("majority")
def majority_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def,
    k=None, strategy_name="majority",
):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        top_k_preds = info["preds"][:k] if k is not None else info["preds"]
        pred        = majority_vote(top_k_preds)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]

        pred_probs = [0.0] * len(doc_to_choice)
        for p, probs in zip(top_k_preds, info["pred_probs"][:k]):
            if p == pred:
                for i, prob in enumerate(probs):
                    pred_probs[i] += prob

        n_pred_votes = top_k_preds.count(pred)
        normalised   = [(v / n_pred_votes, False) for v in pred_probs]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)
    return results_per_doc_id


@register("logits")
def logits_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def,
    k=None, strategy_name="logits",
):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        pred_probs_slice = info["pred_probs"][:k]
        n_preds          = len(pred_probs_slice)

        summed = [
            sum(probs[i] for probs in pred_probs_slice)
            for i in range(len(doc_to_choice))
        ]
        pred       = summed.index(max(summed))
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]

        normalised = [(p / n_preds if n_preds > 0 else 0.0, False) for p in summed]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)
    return results_per_doc_id


@register("condorcet")
def condorcet_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def,
    k=None, strategy_name="condorcet",
):
    """Pairwise head-to-head winner; falls back to majority per doc if no Condorcet winner."""
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        votes       = info["preds"][:k] if k is not None else info["preds"]
        num_classes = len(doc_to_choice)

        pairwise_wins    = {c: 0 for c in range(num_classes)}
        pairwise_matches = {c: 0 for c in range(num_classes)}

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                vi = sum(1 for v in votes if v == i)
                vj = sum(1 for v in votes if v == j)
                if vi > vj:
                    pairwise_wins[i] += 1
                elif vj > vi:
                    pairwise_wins[j] += 1
                pairwise_matches[i] += 1
                pairwise_matches[j] += 1

        condorcet_winner = next(
            (c for c, wins in pairwise_wins.items() if wins == num_classes - 1),
            None,
        )

        if condorcet_winner is None:
            results_per_doc_id.update(
                majority_aggregate_votes(
                    {doc_id: info}, doc_to_choice, task_def,
                    k=k, strategy_name=strategy_name,
                )
            )
            continue

        scores = [
            pairwise_wins[c] / pairwise_matches[c] if pairwise_matches[c] > 0 else 0.0
            for c in range(num_classes)
        ]
        total  = sum(scores)
        probs  = (
            [s / total for s in scores]
            if total > 0
            else [1.0 / num_classes] * num_classes
        )

        pred_probs = [(p, (i == condorcet_winner)) for i, p in enumerate(probs)]
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[condorcet_winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id


@register("borda")
def borda_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def,
    k=None, strategy_name="borda",
):
    """Borda count: rank classes by probability, award (n_classes - rank - 1) points."""
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        num_classes      = len(doc_to_choice)
        scores           = [0.0] * num_classes
        pred_probs_list  = info["pred_probs"][:k] if k is not None else info["pred_probs"]

        for probs in pred_probs_list:
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                scores[cls] += num_classes - rank - 1

        winner      = scores.index(max(scores))
        total_score = sum(scores)
        probs       = [
            s / total_score if total_score > 0 else 1.0 / num_classes
            for s in scores
        ]

        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id


@register("rrf")
def rrf_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def,
    k=None, strategy_name="rrf", rrf_k=60,
):
    """Reciprocal Rank Fusion: score[cls] += 1 / (rrf_k + rank + 1)."""
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        num_classes     = len(doc_to_choice)
        scores          = [0.0] * num_classes
        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]

        for probs in pred_probs_list:
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                scores[cls] += 1.0 / (rrf_k + rank + 1)

        winner      = scores.index(max(scores))
        total_score = sum(scores)
        probs       = [
            s / total_score if total_score > 0 else 1.0 / num_classes
            for s in scores
        ]

        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id
