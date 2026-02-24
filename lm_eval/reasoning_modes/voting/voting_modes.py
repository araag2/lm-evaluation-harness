"""
voting_modes.py — thin public facade for the voting package.

All implementation lives in submodules:

    strategies/simple.py        majority, logits, condorcet, borda, rrf
    strategies/chain_metrics.py text-quality-weighted voting (word_count, readability)
    strategies/llm_metrics.py   confidence/entropy weights + Tier-2 scaffold
    strategies/mbr.py           BLEURT MBR chain selection + weighted vote
    _registry.py                _STRATEGY_REGISTRY, aggregate_metrics_per_strategy
    _utils.py                   normalize, top_k, weighted_vote, pareto helpers

Public API:

    run_voting_modes(args, predictions_per_input_doc, doc_to_choice, task_def, ...)
    simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def)
    aggregate_metrics_per_strategy(results_per_doc, task_def)

Simple voting strategies active by default
------------------------------------------
    majority, logits, condorcet, borda, rrf
    chain_metrics_mean, chain_metrics_rank, chain_metrics_pareto
    answer_confidence, answer_entropy_inv

Tier-2 LLM-scored strategies (opt-in via --tier2_strategies)
-------------------------------------------------------------
    chain_loglik_self/judge      sum of token log-likelihoods of the chain
    chain_perplexity_self/judge  exp(-mean log-likelihood); lower = better
    chain_score_self/judge       generative 1-10 quality score

    "self"  = the same model that generated the chains scores them (self_model arg)
    "judge" = a fixed external model scores all chains (args.judge_model)

    All six families have top-k variants registered (k = None, 5, 11).

Adding new strategies
---------------------
Decorate any function with ``@register("name")`` from ``_registry.py``
and import the module in ``strategies/__init__.py``.  It will appear
automatically in ``simple_voting_modes`` if added to ``_SIMPLE_STRATEGIES``.
"""

import argparse
import warnings
from typing import Any, Dict, List, Optional

from datasets import DatasetDict

# Trigger registration of all strategies into _STRATEGY_REGISTRY
import lm_eval.reasoning_modes.voting.strategies  # noqa: F401

from lm_eval.reasoning_modes.voting._registry import (
    _STRATEGY_REGISTRY,
    aggregate_metrics_per_strategy,
    get_strategy,
)


# ──────────────────────────────────────────────
#  Strategy schedule for simple_voting_modes
# ──────────────────────────────────────────────

_SIMPLE_STRATEGIES: List[tuple] = [
    # Count / rank baselines
    ("majority",  None),
    ("logits",    None),
    ("condorcet", None),
    ("borda",     None),
    ("rrf",       None),
    # Text-quality weighted
    ("chain_metrics_mean",   None),
    ("chain_metrics_rank",   None),
    ("chain_metrics_pareto", None),
    ("chain_metrics_mean",   5),
    ("chain_metrics_rank",   5),
    ("chain_metrics_pareto", 5),
    # LLM-signal weighted (Tier 1 — no extra model calls)
    ("answer_confidence",  None),
    ("answer_entropy_inv", None),
]


# ──────────────────────────────────────────────
#  Tier-2 scoring pass
# ──────────────────────────────────────────────

def _load_scoring_model(provider: str, model_args: str, batch_size):
    """Load a model directly via the lm-eval API for a scoring pass."""
    from lm_eval.api.registry import get_model as _get_model_cls
    lm_cls = _get_model_cls(provider)
    return lm_cls.create_from_arg_string(
        model_args,
        {"batch_size": batch_size},
    )


def run_tier2_chain_scoring(
    args: argparse.Namespace,
    scoring_model: str,
    metric: str,
    predictions_per_input_doc: Dict,
) -> Dict[Any, List[float]]:
    """
    Run a single scoring pass over all (doc, chain) pairs and return raw scores.

    Parameters
    ----------
    scoring_model : lm-eval model_args string, e.g. "pretrained=Qwen/Qwen3-8B"
    metric        : "loglik" | "perplexity" | "score"
    predictions_per_input_doc : standard predictions dict; must have
        doc["Reasoning_Chains"] (list of chain strings per doc) and
        optionally "answering_prompt" (context used as loglik prefix).

    Returns
    -------
    {doc_id: [raw_score_per_chain]}
        loglik/perplexity → negative floats (sum or mean log-likelihood)
        score             → floats in [1, 10]
    """
    from lm_eval.api.instance import Instance
    from lm_eval.reasoning_modes.voting.strategies.llm_metrics import (
        build_score_prompt, parse_chain_score,
    )

    print(
        f"[Tier-2 Scoring] model={scoring_model}  metric={metric}  "
        f"docs={len(predictions_per_input_doc)}"
    )

    lm = _load_scoring_model(args.provider, scoring_model, args.batch_size)

    # Build a flat ordered list of (doc_id, chain_idx, context, chain)
    flat: List[tuple] = []
    for doc_id, info in predictions_per_input_doc.items():
        chains  = info["doc"].get("Reasoning_Chains", [])
        context = info.get("answering_prompt", "")
        for chain_idx, chain in enumerate(chains):
            flat.append((doc_id, chain_idx, context, chain))

    if not flat:
        warnings.warn("[Tier-2 Scoring] No Reasoning_Chains found in predictions — returning empty scores.")
        return {}

    # Build lm-eval Instance objects
    if metric in ("loglik", "perplexity"):
        instances = [
            Instance(
                request_type="loglikelihood",
                doc={},
                arguments=(context, chain),
                idx=i,
            )
            for i, (_, _, context, chain) in enumerate(flat)
        ]
        responses = lm.loglikelihood(instances)
        # responses[i] = (sum_loglik, is_greedy)

    elif metric == "score":
        instances = [
            Instance(
                request_type="generate_until",
                doc={},
                arguments=(build_score_prompt(context, chain), {"until": ["\n"], "max_gen_toks": 8}),
                idx=i,
            )
            for i, (_, _, context, chain) in enumerate(flat)
        ]
        responses = lm.generate_until(instances)
        # responses[i] = generated_text string

    else:
        raise ValueError(f"Unknown metric {metric!r}.")

    # Reconstruct per-doc score lists
    chain_scores: Dict[Any, List[float]] = {}
    for (doc_id, chain_idx, context, chain), resp in zip(flat, responses):
        if doc_id not in chain_scores:
            n = len(predictions_per_input_doc[doc_id]["doc"].get("Reasoning_Chains", []))
            chain_scores[doc_id] = [0.0] * n

        if metric == "loglik":
            sum_loglik, _ = resp
            chain_scores[doc_id][chain_idx] = sum_loglik

        elif metric == "perplexity":
            sum_loglik, _ = resp
            # mean log-likelihood = sum / n_tokens; approximate n_tokens via tokenizer
            try:
                n_tokens = len(lm.tok_encode(chain))
            except Exception:
                n_tokens = max(len(chain.split()), 1)
            chain_scores[doc_id][chain_idx] = sum_loglik / n_tokens

        elif metric == "score":
            chain_scores[doc_id][chain_idx] = parse_chain_score(resp)

    return chain_scores


def run_tier2_voting_modes(
    args: argparse.Namespace,
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
    self_model: str,
) -> Dict:
    """
    Run all Tier-2 strategies requested via args.tier2_strategies.

    Groups strategies by (scoring_model, metric) so each unique combination
    triggers exactly one scoring pass.

    Parameters
    ----------
    self_model : the model used for "self" scoring (the reasoning/answering
                 model that generated the chains).
    args.judge_model : model string for "judge" scoring (from --judge_model).
    args.tier2_strategies : list of Tier-2 strategy names (from --tier2_strategies).
    """
    from lm_eval.reasoning_modes.voting.strategies.llm_metrics import (
        TIER2_STRATEGY_META, vote_with_external_scores,
    )

    requested = getattr(args, "tier2_strategies", []) or []
    if not requested:
        return {}

    judge_model = getattr(args, "judge_model", None)

    # Validate requested names
    unknown = [s for s in requested if s not in TIER2_STRATEGY_META]
    if unknown:
        warnings.warn(f"[Tier-2 Voting] Unknown strategies ignored: {unknown}")
        requested = [s for s in requested if s in TIER2_STRATEGY_META]

    # Group by (scoring_model, metric) to minimise model loads
    groups: Dict[tuple, List[str]] = {}
    for strategy_name in requested:
        meta   = TIER2_STRATEGY_META[strategy_name]
        source = meta["source"]
        metric = meta["metric"]

        if source == "self":
            scoring_model = self_model
        else:  # "judge"
            if judge_model is None:
                warnings.warn(
                    f"[Tier-2 Voting] Strategy {strategy_name!r} needs a judge model "
                    "but --judge_model was not set. Skipping."
                )
                continue
            scoring_model = judge_model

        groups.setdefault((scoring_model, metric), []).append(strategy_name)

    results: Dict = {}
    for (scoring_model, metric), strategy_names in groups.items():
        print(
            f"[Tier-2 Voting] scoring pass: model={scoring_model}  metric={metric}  "
            f"strategies={strategy_names}"
        )
        chain_scores = run_tier2_chain_scoring(args, scoring_model, metric, predictions_per_input_doc)

        for strategy_name in strategy_names:
            k = TIER2_STRATEGY_META[strategy_name]["k"]
            out = vote_with_external_scores(
                predictions_per_input_doc,
                doc_to_choice,
                task_def,
                chain_scores,
                strategy_name=strategy_name,
                metric=metric,
                k=k,
            )
            results.update(out)

    return results


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def run_voting_modes(
    args: argparse.Namespace,
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
    *,
    base_dataset: Optional[DatasetDict] = None,
    answering_model: Optional[str] = None,
    answering_task: Optional[str] = None,
    doc_to_text_module: Optional[str] = None,
    self_model: Optional[str] = None,
) -> Dict:
    """
    Unified voting entry point for all SC-based pipelines.

    Flags (set via __main__.py args):
        args.simple_voting    — True by default; disable with --no-simple-voting
        args.mbr_voting       — False by default; enable with --mbr-voting
        args.tier2_voting     — True when --tier2_strategies is non-empty
        args.tier2_strategies — list of Tier-2 strategy names
        args.judge_model      — model string for "judge" Tier-2 strategies

    MBR requires base_dataset, answering_model, answering_task, doc_to_text_module;
    it is silently skipped (with a warning) if any are missing.

    Tier-2 requires self_model (the model that generated the chains) and
    optionally args.judge_model for judge-source strategies.
    """
    results: Dict = {}

    if getattr(args, "simple_voting", True):
        print(f"[Voting] Running simple voting strategies  docs={len(predictions_per_input_doc)}")
        results.update(
            simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def)
        )

    if getattr(args, "mbr_voting", False):
        _required = {
            "base_dataset":       base_dataset,
            "answering_model":    answering_model,
            "answering_task":     answering_task,
            "doc_to_text_module": doc_to_text_module,
        }
        missing = [k for k, v in _required.items() if v is None]
        if missing:
            warnings.warn(
                f"[Voting] MBR voting requested but required arguments are missing: "
                f"{missing}. Skipping MBR."
            )
        else:
            print(f"[Voting] Running MBR voting strategies  docs={len(predictions_per_input_doc)}")
            from lm_eval.reasoning_modes.voting.strategies.mbr import mbr_voting_modes
            mbr_out = mbr_voting_modes(
                args=args,
                predictions_per_input_doc=predictions_per_input_doc,
                base_dataset=base_dataset,
                answering_model=answering_model,
                answering_task=answering_task,
                doc_to_text_module=doc_to_text_module,
                doc_to_choice=doc_to_choice,
                task_def=task_def,
            )
            results.update(mbr_out["results"])

    if getattr(args, "tier2_strategies", None):
        if self_model is None:
            warnings.warn(
                "[Voting] Tier-2 strategies requested but self_model was not provided. "
                "Self-source strategies will be skipped."
            )
        results.update(
            run_tier2_voting_modes(
                args, predictions_per_input_doc, doc_to_choice, task_def,
                self_model=self_model or "",
            )
        )

    return results


def simple_voting_modes(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
) -> Dict:
    """
    Run all scheduled simple voting strategies over predictions_per_input_doc.

    Returns {strategy_name: {metric_name: score}}.
    """
    n_docs = len(predictions_per_input_doc)
    aggregated_metrics: Dict = {}

    for strategy, top_k in _SIMPLE_STRATEGIES:
        strategy_name = strategy if top_k is None else f"{strategy}_top{top_k}"
        print(
            f"[Voting] strategy={strategy_name!r}  docs={n_docs}"
            + (f"  top_k={top_k}" if top_k is not None else "")
        )

        fn = get_strategy(strategy_name)
        results_per_doc = fn(
            predictions_per_input_doc, doc_to_choice, task_def,
            k=top_k, strategy_name=strategy_name,
        )
        aggregated_metrics[strategy_name] = aggregate_metrics_per_strategy(
            results_per_doc, task_def
        )

    return aggregated_metrics
