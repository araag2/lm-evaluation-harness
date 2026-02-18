"""
voting_modes.py — unified voting module.

Public API
----------
run_voting_modes(args, predictions_per_input_doc, doc_to_choice, task_def, ...)
    Flags-driven entry point for all SC-based pipelines.  Controlled by two
    CLI flags defined in __main__.py:
        --no-simple-voting    skip majority / logits / condorcet / borda / rrf
        --no-mbr-voting       skip BLEURT-weighted MBR voting
    MBR requires base_dataset / answering_model / answering_task /
    doc_to_text_module to be supplied; a clear ValueError is raised otherwise.

simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def)
    Run simple strategies only (also used directly by cross_consistency).

aggregate_metrics_per_strategy(results_per_doc, task_def)
    Helper used by cross_consistency and MBR internally.
"""

import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict
from mbrs.metrics import MetricBLEU, MetricBLEURT
from mbrs.decoders import DecoderMBR
from tqdm import tqdm

from lm_eval.reasoning_modes.reasoning_utils import (
    inject_reasoning_into_dataset,
    run_answering_for_dataset,
    format_results_dict,
)


# ============================================================
#  Unified entry point
# ============================================================

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

    Dispatches to simple and/or MBR voting based on two flags set by
    ``__main__.py`` (and forwarded via ``args``):

    - ``args.simple_voting`` — ``True`` by default, set ``False`` with ``--no-simple-voting``
    - ``args.mbr_voting``    — ``True`` by default, set ``False`` with ``--no-mbr-voting``

    MBR voting additionally requires the keyword-only arguments
    *base_dataset*, *answering_model*, *answering_task*, and
    *doc_to_text_module*; a ``ValueError`` is raised if any are missing
    when MBR is enabled.

    Returns a flat ``{strategy_name: {metric: score, ...}}`` dict.
    """
    do_simple = args.simple_voting
    do_mbr    = args.mbr_voting

    results: Dict = {}

    if do_simple:
        print(
            f"[Voting] Running simple voting strategies  "
            f"docs={len(predictions_per_input_doc)}"
        )
        results.update(
            simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def)
        )

    if do_mbr:
        _required = {
            "base_dataset":      base_dataset,
            "answering_model":   answering_model,
            "answering_task":    answering_task,
            "doc_to_text_module": doc_to_text_module,
        }
        missing = [k for k, v in _required.items() if v is None]
        if missing:
            raise ValueError(
                f"[Voting] MBR voting is enabled (use --no-mbr-voting to disable) "
                f"but the following required arguments were not passed to "
                f"run_voting_modes: {missing}"
            )
        print(
            f"[Voting] Running MBR voting strategies  "
            f"docs={len(predictions_per_input_doc)}"
        )
        mbr_out = _mbr_voting_modes(
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


# ============================================================
#  Simple voting strategies
# ============================================================

# Dispatch table — maps strategy name → aggregation function.
# Register new strategies at the bottom of this file.
_STRATEGY_REGISTRY: Dict[str, Any] = {}


def simple_voting_modes(
    predictions_per_input_doc: Dict,
    doc_to_choice: List[str],
    task_def: Any,
) -> Dict:
    """
    Run all registered simple voting strategies over *predictions_per_input_doc*.

    Returns ``{strategy_name: {metric_name: score}}``.
    Also usable directly when MBR is not needed (e.g. cross_consistency).
    """
    n_docs = len(predictions_per_input_doc)
    aggregated_metrics: Dict = {}

    for strategy, top_k in [
        ("majority",  None),
        ("logits",    None),
        ("condorcet", None),
        ("borda",     None),
        ("rrf",       None),
    ]:
        strategy_name = strategy if top_k is None else f"{strategy}_top{top_k}"
        print(
            f"[Voting] strategy={strategy_name!r}  docs={n_docs}"
            + (f"  top_k={top_k}" if top_k is not None else "")
        )

        fn = _STRATEGY_REGISTRY.get(strategy)
        if fn is None:
            raise ValueError(
                f"Unknown voting strategy {strategy!r}. "
                f"Registered: {list(_STRATEGY_REGISTRY)}"
            )

        results_per_doc = fn(
            predictions_per_input_doc, doc_to_choice, task_def,
            k=top_k, strategy_name=strategy_name,
        )
        aggregated_metrics[strategy_name] = aggregate_metrics_per_strategy(
            results_per_doc, task_def
        )

    return aggregated_metrics


# ------------------------------------
# Metric aggregation helper
# ------------------------------------

def aggregate_metrics_per_strategy(results_per_doc: Dict, task_def: Any) -> Dict:
    """Aggregate per-document metric values using the task's own aggregation functions."""
    aggregated: Dict = {}
    for metric_name, agg_fn in task_def.aggregation().items():
        values = [
            results_per_doc[doc_id][metric_name]
            for doc_id in results_per_doc
            if metric_name in results_per_doc[doc_id]
        ]
        if values:
            aggregated[metric_name] = agg_fn(values)
    return aggregated


# ------------------------------------
# Per-strategy voting functions
# ------------------------------------

def majority_vote(predictions: list) -> int:
    """Return the most-common prediction index."""
    return Counter(predictions).most_common(1)[0][0]


def majority_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="majority"
):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        top_k_preds = info["preds"][:k] if k is not None else info["preds"]
        pred = majority_vote(top_k_preds)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]

        pred_probs = [0.0] * len(doc_to_choice)
        for p, probs in zip(top_k_preds, info["pred_probs"][:k]):
            if p == pred:
                for i, prob in enumerate(probs):
                    pred_probs[i] += prob

        n_pred_votes = top_k_preds.count(pred)
        normalised = [(v / n_pred_votes, False) for v in pred_probs]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)
    return results_per_doc_id


def logits_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="logits"
):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        pred_probs_slice = info["pred_probs"][:k]
        n_preds = len(pred_probs_slice)

        summed_logits = [
            sum(probs[i] for probs in pred_probs_slice)
            for i in range(len(doc_to_choice))
        ]
        pred = summed_logits.index(max(summed_logits))
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]

        normalised = [(p / n_preds if n_preds > 0 else 0.0, False) for p in summed_logits]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], normalised)
    return results_per_doc_id


def condorcet_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="condorcet"
):
    """
    Condorcet voting:
    - Each pair of candidates is compared by raw vote counts.
    - Winner must beat every other candidate head-to-head.
    - If no Condorcet winner exists, falls back to majority per doc.
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        votes = info["preds"][:k] if k is not None else info["preds"]
        num_classes = len(doc_to_choice)

        pairwise_wins    = {c: 0 for c in range(num_classes)}
        pairwise_matches = {c: 0 for c in range(num_classes)}

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                votes_i = sum(1 for v in votes if v == i)
                votes_j = sum(1 for v in votes if v == j)
                if votes_i > votes_j:
                    pairwise_wins[i] += 1
                elif votes_j > votes_i:
                    pairwise_wins[j] += 1
                pairwise_matches[i] += 1
                pairwise_matches[j] += 1

        condorcet_winner = next(
            (c for c, wins in pairwise_wins.items() if wins == num_classes - 1),
            None,
        )

        if condorcet_winner is None:
            # Per-doc fallback to majority — do not short-circuit the whole loop
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
        total_score = sum(scores)
        probs = (
            [s / total_score for s in scores]
            if total_score > 0
            else [1.0 / num_classes] * num_classes
        )

        pred_probs = [(p, (i == condorcet_winner)) for i, p in enumerate(probs)]
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[condorcet_winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id


def borda_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="borda"
):
    """
    Borda count:
    - Each prediction ranks classes by probability (descending).
    - Class at rank r earns (num_classes - r - 1) points.
    - Winner = highest total points.
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        num_classes = len(doc_to_choice)
        scores = [0.0] * num_classes

        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]
        for probs in pred_probs_list:
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                scores[cls] += num_classes - rank - 1

        winner      = scores.index(max(scores))
        total_score = sum(scores)
        probs = [
            s / total_score if total_score > 0 else 1.0 / num_classes
            for s in scores
        ]

        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id


def rrf_aggregate_votes(
    predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="rrf", rrf_k=60
):
    """
    Reciprocal Rank Fusion (RRF):
    - Each ranked list contributes 1 / (rrf_k + rank + 1) to a class score.
    - Winner = highest fused score.
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        num_classes = len(doc_to_choice)
        scores = [0.0] * num_classes

        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]
        for probs in pred_probs_list:
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                scores[cls] += 1.0 / (rrf_k + rank + 1)

        winner      = scores.index(max(scores))
        total_score = sum(scores)
        probs = [
            s / total_score if total_score > 0 else 1.0 / num_classes
            for s in scores
        ]

        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id


# Strategy registry — add new strategies here only.
_STRATEGY_REGISTRY.update({
    "majority":  majority_aggregate_votes,
    "logits":    logits_aggregate_votes,
    "condorcet": condorcet_aggregate_votes,
    "borda":     borda_aggregate_votes,
    "rrf":       rrf_aggregate_votes,
})


# ============================================================
#  MBR voting  (private — called only via run_voting_modes)
# ============================================================

def _mbr_voting_modes(
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

    # Limit applied locally — never mutate the caller's DatasetDict
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


def _get_mbr_reasoning_chains(
    reasoning_chains_per_document: Dict[Any, List[str]],
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Decode the best reasoning chain per document for each active MBR metric.

    To re-enable BLEU, uncomment the bleu lines in ``mbr_metrics`` below.

    Returns:
        best_chains  : {metric_name: [best_chain_per_doc, ...]}
        chain_scores : {metric_name: [normalised_score_list_per_doc, ...]}
    """
    def _normalize(scores: list) -> list:
        lo   = min(scores)
        span = max(scores) - lo + 1e-8
        return [(s - lo) / span for s in scores]

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
            chain_scores[name].append(_normalize(output.score))

    return best_chains, chain_scores


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
    def _weighted_majority(predictions: list, weights: list) -> int:
        totals: Dict[int, float] = {}
        for p, w in zip(predictions, weights):
            totals[p] = totals.get(p, 0.0) + w
        return max(totals, key=totals.get)

    res_per_metric: Dict = {}

    for metric, weight_scores_by_doc in mbr_chain_scores.items():
        strategy_name = f"{metric}_weight_logits"
        res: Dict = {}

        for (doc_id, info), weight_scores in zip(
            predictions_per_input_doc.items(), weight_scores_by_doc
        ):
            pred = _weighted_majority(info["preds"], weight_scores)
            predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]

            pred_probs = [0.0] * len(doc_to_choice)
            for p, probs, score in zip(info["preds"], info["pred_probs"], weight_scores):
                if p == pred:
                    for i, prob in enumerate(probs):
                        pred_probs[i] += prob * score

            sum_w      = sum(weight_scores)
            normalised = [(v / sum_w if sum_w != 0.0 else v, False) for v in pred_probs]
            res[doc_id] = task_def.process_results(info["doc"], normalised)

        res_per_metric[strategy_name] = aggregate_metrics_per_strategy(res, task_def)

    return res_per_metric
