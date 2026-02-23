"""
voting_modes.py — thin public facade for the voting package.

All implementation lives in submodules:

    strategies/simple.py        majority, logits, condorcet, borda, rrf
    strategies/chain_metrics.py text-quality-weighted voting (word_count, readability)
    strategies/llm_metrics.py   confidence/entropy weights + Tier-2 scaffold
    strategies/mbr.py           BLEURT MBR chain selection + weighted vote
    _registry.py                _STRATEGY_REGISTRY, aggregate_metrics_per_strategy
    _utils.py                   normalize, top_k, weighted_vote, pareto helpers

Public API (unchanged — all callers continue to work):

    run_voting_modes(args, predictions_per_input_doc, doc_to_choice, task_def, ...)
    simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def)
    aggregate_metrics_per_strategy(results_per_doc, task_def)

Simple voting strategies active by default
------------------------------------------
    majority, logits, condorcet, borda, rrf
    chain_metrics_mean, chain_metrics_rank, chain_metrics_pareto
    answer_confidence, answer_entropy_inv

Top-k variants (k=3, k=5) are registered automatically for all
chain_metrics_* and answer_* strategies.

Adding new strategies
---------------------
Decorate any function with ``@register("name")`` from ``_registry.py``
and import the module in ``strategies/__init__.py``.  It will appear
automatically in ``simple_voting_modes`` if added to ``_SIMPLE_STRATEGIES``.
"""

import argparse
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
#
# Each entry is (strategy_name, top_k).  Add new strategies here
# to include them in every simple-voting run.

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
    # LLM-signal weighted (Tier 1 — no extra model calls)
    ("answer_confidence",  None),
    ("answer_entropy_inv", None),
]


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
) -> Dict:
    """
    Unified voting entry point for all SC-based pipelines.

    Dispatched by two flags (set by __main__.py):
        args.simple_voting  — True by default; set False with --no-simple-voting
        args.mbr_voting     — True by default; set False with --no-mbr-voting

    MBR additionally requires base_dataset, answering_model, answering_task,
    and doc_to_text_module; a ValueError is raised if any are missing.
    """
    results: Dict = {}

    if getattr(args, "simple_voting", True):
        print(f"[Voting] Running simple voting strategies  docs={len(predictions_per_input_doc)}")
        results.update(
            simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def)
        )

    if getattr(args, "mbr_voting", True):
        _required = {
            "base_dataset":       base_dataset,
            "answering_model":    answering_model,
            "answering_task":     answering_task,
            "doc_to_text_module": doc_to_text_module,
        }
        missing = [k for k, v in _required.items() if v is None]
        if missing:
            raise ValueError(
                f"[Voting] MBR voting is enabled (use --no-mbr-voting to disable) "
                f"but required arguments are missing: {missing}"
            )
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

    return results


def simple_voting_modes(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
) -> Dict:
    """
    Run all scheduled simple voting strategies over predictions_per_input_doc.

    Returns {strategy_name: {metric_name: score}}.
    Also used directly by cross_consistency (no MBR dependency).
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
