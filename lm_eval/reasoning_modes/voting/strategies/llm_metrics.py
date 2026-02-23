"""
strategies/llm_metrics.py — LLM-signal-based voting strategies.

Strategies that use model-derived scores as chain weights.

Two tiers
---------
Tier 1 — uses data already available in predictions_per_input_doc (no extra
         model calls needed):

    answer_confidence      weight = max(pred_probs) per chain
    answer_entropy_inv     weight = 1 - normalised_entropy (high confidence = high weight)

Tier 2 — requires a separate model scoring pass (chain perplexity or
         explicit self-/judge-scoring).  These are scaffolded here as
         callable builders.  Actual execution is orchestrated by
         ``multi_turn_CoT_SC.py`` / ``cross_consistency.py`` via
         ``run_llm_scored_voting()``.

    chain_loglik_self      log-likelihood of chain text under reasoning model
    chain_loglik_judge     log-likelihood of chain text under a fixed judge model
    self_score             model rates own chain quality (generative, 1-10 prompt)
    judge_score            fixed judge model rates chain quality (generative, 1-10 prompt)

Top-k variants are auto-registered for Tier-1 strategies.

Dependencies
------------
Tier 1: none beyond numpy (already in lm_eval)
Tier 2: requires lm_eval model infrastructure (passed at call time)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from lm_eval.reasoning_modes.voting._registry import register, aggregate_metrics_per_strategy
from lm_eval.reasoning_modes.voting._utils import (
    normalize_scores,
    entropy,
    weighted_vote,
    apply_top_k,
)

TOP_K_VALUES: List[Optional[int]] = [None, 5, 10]


# ══════════════════════════════════════════════
#  Tier 1 — data-derived strategies
# ══════════════════════════════════════════════

def _llm_metric_vote(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
    weight_fn,          # (preds, pred_probs) → List[float]
    k: Optional[int],
    strategy_name: str,
) -> Dict:
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        raw_weights = weight_fn(info["preds"], info["pred_probs"])
        weights     = normalize_scores(raw_weights)

        preds_k, probs_k, weights_k = apply_top_k(
            info["preds"], info["pred_probs"], weights, k
        )
        if not preds_k:
            preds_k, probs_k, weights_k = info["preds"], info["pred_probs"], weights

        winner, normalised = weighted_vote(preds_k, probs_k, weights_k, doc_to_choice)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)
    return results_per_doc_id


def _confidence_weights(preds: List[int], pred_probs: List[List[float]]) -> List[float]:
    """Weight = max probability in the chain's logit distribution."""
    return [max(probs) for probs in pred_probs]


def _entropy_inv_weights(preds: List[int], pred_probs: List[List[float]]) -> List[float]:
    """Weight = negative Shannon entropy (low entropy = high confidence = high weight)."""
    return [-entropy(probs) for probs in pred_probs]


# Auto-register Tier-1 × top_k combos
_TIER1_METHODS = {
    "answer_confidence":  _confidence_weights,
    "answer_entropy_inv": _entropy_inv_weights,
}

for _method_name, _wfn in _TIER1_METHODS.items():
    for _k in TOP_K_VALUES:
        _suffix        = "" if _k is None else f"_top{_k}"
        _strategy_name = f"{_method_name}{_suffix}"

        def _make_fn(wfn, top_k, sname):
            @register(sname)
            def _fn(predictions_per_input_doc, doc_to_choice, task_def,
                    k=top_k, strategy_name=sname):
                return _llm_metric_vote(
                    predictions_per_input_doc, doc_to_choice, task_def,
                    wfn, k, strategy_name,
                )
            _fn.__name__ = sname
            return _fn

        _make_fn(_wfn, _k, _strategy_name)


# ══════════════════════════════════════════════
#  Tier 2 — model-scored strategies (scaffolding)
# ══════════════════════════════════════════════
#
# These functions accept pre-computed per-chain scores (obtained externally
# by running a model scoring pass) and produce a voted result.
# They are NOT registered in _STRATEGY_REGISTRY because they require
# extra arguments beyond the standard signature.
# Callers should invoke them directly after obtaining scores.
#
# Expected score format: Dict[doc_id, List[float]]  (one score per chain)

def vote_with_external_scores(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
    chain_scores: Dict[Any, List[float]],   # doc_id → [score_per_chain]
    strategy_name: str,
    k: Optional[int] = None,
) -> Dict:
    """
    Generic weighted vote using externally computed chain scores.

    Use this for Tier-2 strategies (chain_loglik_self, chain_loglik_judge,
    self_score, judge_score) after obtaining scores from a model run.

    Parameters
    ----------
    chain_scores : {doc_id: [score_per_chain]}
        Higher score = better chain.  Missing docs fall back to uniform weights.

    Returns
    -------
    {strategy_name: {metric: score}}
    """
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        raw = chain_scores.get(doc_id)
        if raw is None or len(raw) != len(info["preds"]):
            weights = [1.0] * len(info["preds"])
        else:
            weights = normalize_scores(raw)

        preds_k, probs_k, weights_k = apply_top_k(
            info["preds"], info["pred_probs"], weights, k
        )
        if not preds_k:
            preds_k, probs_k, weights_k = info["preds"], info["pred_probs"], weights

        winner, normalised = weighted_vote(preds_k, probs_k, weights_k, doc_to_choice)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)

    from lm_eval.reasoning_modes.voting._registry import aggregate_metrics_per_strategy
    return {strategy_name: aggregate_metrics_per_strategy(results_per_doc_id, task_def)}


# ──────────────────────────────────────────────
#  Scoring prompt templates (Tier 2)
# ──────────────────────────────────────────────
#
# These are the prompts used when running a model-based scoring pass.
# Import and use from the pipeline orchestration code.

SELF_SCORE_PROMPT = (
    "You are evaluating the quality of a reasoning chain for a medical question.\n"
    "Rate the following reasoning chain on a scale from 1 to 10, where:\n"
    "  1 = completely wrong, incoherent, or irrelevant\n"
    "  10 = thorough, accurate, and well-structured\n\n"
    "Reasoning chain:\n{chain}\n\n"
    "Respond with a single integer between 1 and 10. Score:"
)

JUDGE_SCORE_PROMPT = (
    "You are an expert medical evaluator.\n"
    "Rate the following reasoning chain for the given question on a scale from 1 to 10, where:\n"
    "  1 = completely wrong, incoherent, or irrelevant\n"
    "  10 = thorough, accurate, and consistent with medical knowledge\n\n"
    "Question: {question}\n\n"
    "Reasoning chain:\n{chain}\n\n"
    "Respond with a single integer between 1 and 10. Score:"
)
