"""
strategies/llm_metrics.py — LLM-signal-based voting strategies.

Strategies that use model-derived scores as chain weights.

Two tiers
---------
Tier 1 — uses data already available in predictions_per_input_doc (no extra
         model calls needed):

    answer_confidence      weight = max(pred_probs) per chain
    answer_entropy_inv     weight = 1 - normalised_entropy (high confidence = high weight)

Tier 2 — requires a separate model scoring pass.  Raw scores are obtained
         externally by the pipeline orchestration code
         (``multi_turn_CoT_SC.py`` / ``cross_consistency.py``) and then
         passed to ``vote_with_external_scores()``.

    Six strategy families, each available for "self" (reasoning model scores
    its own chains) and "judge" (a fixed external judge model scores them):

    chain_loglik_self/judge    sum of token log-likelihoods of the chain
                               (higher = more likely under the model = better)
    chain_perplexity_self/judge  exp(-mean log-likelihood)
                               (lower = more fluent = better; weights inverted)
    chain_score_self/judge     generative 1-10 quality prompt score
                               (higher = better)

    All six families are auto-registered with top-k variants (k = None, 5, 11).
    Use ``TIER2_STRATEGY_META`` to introspect source/metric/k for any name.

Top-k variants are auto-registered for both Tier-1 and Tier-2 strategies.

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
#  Tier 2 — model-scored strategies
# ══════════════════════════════════════════════
#
# Architecture overview
# ---------------------
# Tier-2 strategies require a scoring pass that runs a model over every
# (doc, reasoning_chain) pair to produce a scalar weight per chain.
# Three distinct score types are defined:
#
#   metric="loglik"      sum of token log-likelihoods of the chain text.
#                        Obtained via output_type=loglikelihood_rolling.
#                        resps[0] = (sum_loglik, n_tokens)
#                        Higher is better.
#
#   metric="perplexity"  mean token log-likelihood = sum_loglik / n_tokens,
#                        also from loglikelihood_rolling.
#                        scores_to_weights() converts: weight = -exp(-mean_loglik)
#                        Lower perplexity = higher weight.
#
#   metric="score"       a generative 1-10 quality rating obtained by running
#                        CHAIN_SCORE_PROMPT through the model with
#                        output_type=generate_until, then parsing with
#                        parse_chain_score().
#
# Two model sources:
#   source="self"    the same model that generated the chains scores them.
#   source="judge"   a separate fixed judge model scores all chains.
#
# Connection to orchestrators
# ---------------------------
# The scoring pass must happen BEFORE voting.
# The entry point is run_tier2_chain_scoring() in voting_modes.py, which:
#   1. Builds a flat list of (context_prompt, chain) pairs per doc.
#   2. Runs a single simple_evaluate pass with the appropriate output_type.
#   3. Returns Dict[doc_id, List[float]] — one raw score per chain per doc.
#   4. Calls vote_with_external_scores() with that dict.
#
# This function is NOT YET IMPLEMENTED in voting_modes.py and requires:
#   - A scoring task helper that wraps (context, chain) pairs as lm-eval docs
#     with output_type=loglikelihood_rolling or generate_until.
#   - Access to the context prompt for each doc (stored in predictions[doc_id]).
#
# Family and metadata definitions
# --------------------------------

_TIER2_FAMILIES: Dict[str, Dict] = {
    "chain_loglik_self":      {"source": "self",  "metric": "loglik"},
    "chain_loglik_judge":     {"source": "judge", "metric": "loglik"},
    "chain_perplexity_self":  {"source": "self",  "metric": "perplexity"},
    "chain_perplexity_judge": {"source": "judge", "metric": "perplexity"},
    "chain_score_self":       {"source": "self",  "metric": "score"},
    "chain_score_judge":      {"source": "judge", "metric": "score"},
}

# Maps every Tier-2 strategy name (including top-k variants) → {source, metric, k}.
# Orchestration code should query this to determine what scoring pass is needed.
TIER2_STRATEGY_META: Dict[str, Dict] = {}
for _family, _fmeta in _TIER2_FAMILIES.items():
    for _k in TOP_K_VALUES:
        _suffix = "" if _k is None else f"_top{_k}"
        TIER2_STRATEGY_META[f"{_family}{_suffix}"] = {**_fmeta, "k": _k}


# ── Weight conversion ──────────────────────────────────────────────────────

def scores_to_weights(metric: str, raw_scores: List[float]) -> List[float]:
    """
    Convert raw per-chain metric values to voting weights (higher = better).

    Expected raw_scores format per metric
    --------------------------------------
    "loglik"     : sum of token log-likelihoods for the chain (one float per chain,
                   typically a large negative number; higher = more likely = better).
                   weight = raw_score  (used directly)

    "perplexity" : mean token log-likelihood, i.e. sum(logprobs) / n_tokens
                   (one float per chain, negative; less negative = better).
                   Perplexity is defined as exp(-mean_loglik).  Lower perplexity
                   = more fluent chain = better.
                   weight = -exp(-mean_loglik)  (negate so higher weight = lower PPL)

    "score"      : generative 1-10 quality rating (one float per chain;
                   higher = better).
                   weight = raw_score  (used directly)

    Returns
    -------
    List[float] ready to be passed to normalize_scores().
    """
    if metric == "loglik":
        return list(raw_scores)
    elif metric == "perplexity":
        # raw_scores = mean log-likelihood (negative). PPL = exp(-mean_loglik).
        # Lower PPL is better → negate so that higher weight = lower perplexity.
        return [-math.exp(-s) for s in raw_scores]
    elif metric == "score":
        return list(raw_scores)
    else:
        raise ValueError(f"Unknown metric type {metric!r}. Expected 'loglik', 'perplexity', or 'score'.")


# ── Generic voting entry-point ─────────────────────────────────────────────

def vote_with_external_scores(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
    chain_scores: Dict[Any, List[float]],
    strategy_name: str,
    metric: str = "loglik",
    k: Optional[int] = None,
) -> Dict:
    """
    Weighted vote using externally computed per-chain scores.

    Use this for all Tier-2 strategies after obtaining raw scores from a
    model scoring pass.

    Parameters
    ----------
    chain_scores : {doc_id: [raw_score_per_chain]}
        One scalar per chain.  Expected format depends on *metric*:
          "loglik"     → sum of token log-likelihoods
          "perplexity" → mean token log-likelihood  (sum / n_tokens)
          "score"      → generative 1-10 quality rating
        Missing docs or length mismatches fall back to uniform weights.
    metric : "loglik" | "perplexity" | "score"
        Controls how raw scores are converted to weights via scores_to_weights().
        Must match the Tier-2 family being run.
    k : Optional[int]
        Top-k chain selection before voting.  None = use all chains.

    Returns
    -------
    {strategy_name: {metric_name: aggregated_score}}
    """
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        raw = chain_scores.get(doc_id)
        if raw is None or len(raw) != len(info["preds"]):
            weights = [1.0] * len(info["preds"])
        else:
            weights = normalize_scores(scores_to_weights(metric, raw))

        preds_k, probs_k, weights_k = apply_top_k(
            info["preds"], info["pred_probs"], weights, k
        )
        if not preds_k:
            preds_k, probs_k, weights_k = info["preds"], info["pred_probs"], weights

        winner, normalised = weighted_vote(preds_k, probs_k, weights_k, doc_to_choice)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)

    return {strategy_name: aggregate_metrics_per_strategy(results_per_doc_id, task_def)}


# ──────────────────────────────────────────────
#  Scoring prompt template + utilities (Tier 2 — score metric)
# ──────────────────────────────────────────────
#
# Used for both chain_score_self and chain_score_judge passes.
# Placeholders: {{Context}}, {{Reasoning Chain}}
# Expected model completion: a single integer 1-10.
#
# Workflow:
#   1. For each (doc, chain), call build_score_prompt(context, chain).
#   2. Feed the resulting prompt to the model with output_type=generate_until,
#      stop=["\n"], max_new_tokens=8.
#   3. Parse the integer from the generated text with parse_chain_score().

import re as _re

CHAIN_SCORE_PROMPT = "Act as an impartial judge and evaluate the quality of the following reasoning chain within the context of a downstream medical task. The objective of the task is carefully explained within the context prompt.\n\nYour evaluation should primarily consider the helpfulness of the reasoning chain in reaching the correct answer for the task at hand, but also consider factors such as relevance, accuracy, depth, and level of detail.\n\nBegin your evaluation by summarizing the strengths and weaknesses of the reasoning chain in a few sentences. Avoid any position biases. Do not allow the length of the chain to influence your evaluation. Be as objective as possible.\n\nAfter providing your explanation, output your final verdict by strictly following this format: outputting a number from 1 to 10, where:\n  1 = completely wrong, incoherent, or irrelevant\n  10 = thorough, accurate, and well-structured\n\nContext:\n{{Context}}\n\nReasoning chain:\n{{Reasoning Chain}}\n\nScore: "

_SCORE_RE = _re.compile(r"\b(10|[1-9])\b")


def build_score_prompt(context: str, chain: str) -> str:
    """Fill the {{Context}} and {{Reasoning Chain}} placeholders in CHAIN_SCORE_PROMPT."""
    return CHAIN_SCORE_PROMPT.replace("{{Context}}", context).replace("{{Reasoning Chain}}", chain)


def parse_chain_score(model_output: str, default: float = 5.0) -> float:
    """
    Extract a 1-10 integer score from a model's generative output.

    Searches for the first standalone digit (1-10) in the output.
    Returns *default* if no valid score is found (e.g. model refused or
    produced free text without a number).

    Parameters
    ----------
    model_output : raw string returned by the model after "Score: "
    default      : fallback value when parsing fails (default 5.0 = neutral)
    """
    match = _SCORE_RE.search(model_output)
    return float(match.group(1)) if match else default