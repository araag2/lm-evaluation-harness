"""
strategies/mbr.py — Minimum Bayes Risk (MBR) voting over reasoning chains.

Moved from voting_modes.py.  Public entry point is ``mbr_voting_modes()``,
called by ``run_voting_modes()`` in voting_modes.py when args.mbr_voting=True.

MBR selects the best reasoning chain per document under one or more
sequence-similarity metrics (currently BLEURT; BLEU commented out),
then re-runs answering with those chains and applies a weighted vote.

Dependencies
------------
    pip install mbrs
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict
from tqdm import tqdm

from lm_eval.reasoning_modes.reasoning_utils import (
    inject_reasoning_into_dataset,
    run_answering_for_dataset,
    format_results_dict,
)
from lm_eval.reasoning_modes.voting._registry import aggregate_metrics_per_strategy
from lm_eval.reasoning_modes.voting._utils import normalize_scores, weighted_vote


# ──────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────

def mbr_voting_modes(
    args: argparse.Namespace,
    predictions_per_input_doc: Dict,
    base_dataset: DatasetDict,
    answering_model: str,
    answering_task: str,
    doc_to_text_module: str,
    doc_to_choice: List[str],
    task_def: Any,
) -> Dict:
    """
    MBR-based voting over accumulated reasoning chains.

    For each active MBR metric:
      1. Select the best reasoning chain per document via MBR decoding.
      2. Re-run answering with those chains injected.
      3. Run a weighted-majority vote using the normalised MBR scores.

    Returns ``{"results": {strategy_name: {metric: score, ...}, ...}}``.
    """
    from mbrs.metrics import MetricBLEURT  # lazy import — only when MBR is used
    from mbrs.decoders import DecoderMBR

    reasoning_chains_per_document: Dict = {
        doc_id: info["doc"]["Reasoning_Chains"]
        for doc_id, info in predictions_per_input_doc.items()
    }
    n_docs = len(reasoning_chains_per_document)
    print(f"[MBR] Selecting best reasoning chains for {n_docs} documents.")

    mbr_best_chains, mbr_chain_scores = _get_mbr_reasoning_chains(
        reasoning_chains_per_document
    )

    metrics_results: Dict = {"results": {}}

    effective_dataset = base_dataset
    if args.limit is not None:
        effective_dataset = DatasetDict(
            {split: ds.select(range(min(args.limit, len(ds))))
             for split, ds in base_dataset.items()}
        )

    for mbr_metric, chain_list in mbr_best_chains.items():
        print(
            f"[MBR] Answering with MBR-selected chains  "
            f"metric={mbr_metric!r}  docs={len(chain_list)}"
        )
        dataset_with_reasoning = inject_reasoning_into_dataset(
            effective_dataset, chain_list
        )
        raw_output = run_answering_for_dataset(
            args=args,
            answering_model=answering_model,
            answering_task_name=answering_task,
            dataset_with_reasoning=dataset_with_reasoning,
            doc_to_text_module=doc_to_text_module,
        )
        metrics_results["results"][mbr_metric] = format_results_dict(
            raw_output["results"][answering_task]
        )

    print(
        f"[MBR] Computing weighted-majority vote  "
        f"metrics={list(mbr_chain_scores)}"
    )
    weighted_scores = _vote_weighted_chains(
        mbr_chain_scores, predictions_per_input_doc, doc_to_choice, task_def
    )
    metrics_results["results"].update(weighted_scores)

    return metrics_results


# ──────────────────────────────────────────────
#  MBR chain selection
# ──────────────────────────────────────────────

def _get_mbr_reasoning_chains(
    reasoning_chains_per_document: Dict[Any, List[str]],
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Decode the best reasoning chain per document for each active MBR metric.

    To enable BLEU, uncomment the bleu lines in ``mbr_metrics``.

    Returns
    -------
    best_chains  : {metric_name: [best_chain_per_doc, ...]}
    chain_scores : {metric_name: [[normalised_score_list], ...]}
    """
    from mbrs.metrics import MetricBLEURT
    from mbrs.decoders import DecoderMBR

    # bleu   = MetricBLEU(MetricBLEU.Config(num_workers=32))
    bleurt = MetricBLEURT(MetricBLEURT.Config(batch_size=32, model="lucadiliello/bleurt-tiny-128"))

    mbr_metrics = [
        ("bleurt", bleurt, bleurt.Config),
        # ("bleu",   bleu,   bleu.Config),
    ]

    best_chains:  Dict[str, list] = {name: [] for name, _, _ in mbr_metrics}
    chain_scores: Dict[str, list] = {name: [] for name, _, _ in mbr_metrics}

    for name, metric, config in mbr_metrics:
        decoder = DecoderMBR(config, metric)
        n_docs  = len(reasoning_chains_per_document)
        print(f"[MBR] Decoding  metric={name!r}  docs={n_docs}")

        for doc_id, chains in tqdm(
            reasoning_chains_per_document.items(), desc=name.upper(), total=n_docs
        ):
            output = decoder.decode(chains, chains, nbest=len(chains))
            best_chains[name].append(chains[output.idx[0]])
            chain_scores[name].append(normalize_scores(list(output.score)))

    return best_chains, chain_scores


# ──────────────────────────────────────────────
#  Weighted vote using MBR scores
# ──────────────────────────────────────────────

def _vote_weighted_chains(
    mbr_chain_scores: Dict[str, List[list]],
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
) -> Dict:
    """
    Weighted-majority vote using normalised MBR scores as per-chain weights.

    Returns ``{"{metric}_weight_logits": {metric: score, ...}, ...}``.
    """
    res_per_metric: Dict = {}

    for metric, weight_scores_by_doc in mbr_chain_scores.items():
        strategy_name = f"{metric}_weight_logits"
        res: Dict = {}

        for (doc_id, info), weights in zip(
            predictions_per_input_doc.items(), weight_scores_by_doc
        ):
            winner, normalised = weighted_vote(
                info["preds"], info["pred_probs"], weights, doc_to_choice
            )
            predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
            res[doc_id] = task_def.process_results(info["doc"], normalised)

        res_per_metric[strategy_name] = aggregate_metrics_per_strategy(res, task_def)

    return res_per_metric
