import json
from lm_eval.reasoning_modes.reasoning_utils import *

from collections import Counter
from nltk.util import ngrams
from joblib import Parallel, delayed
from mbrs.metrics import MetricBLEU, MetricBLEURT
from mbrs.decoders import DecoderMBR
from tqdm import tqdm

def mode_multi_turn_CoT_SC(args: argparse.Namespace) -> Dict:
    if args.vote_file is not None:
        return only_vote(args)

    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning model and one answering model. {args.reasoning_models=} {args.answering_models=}")

    if len(args.reasoning_tasks) != 1 or len(args.answering_tasks) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning task and one answering task. {args.reasoning_tasks=} {args.answering_tasks=}")

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]

    reasoning_task = args.reasoning_tasks[0]
    answering_task = args.answering_tasks[0]

    reasoning_outputs = run_reasoning(args)[reasoning_model][reasoning_task]
    reasoning_chains_per_document = extract_multiple_reasoning_chains_per_document(reasoning_outputs)

    full_task_name = answering_task.replace(":", "_")
    task_def = tasks.get_task_dict([full_task_name])[full_task_name]
    doc_to_choice = task_def.config.doc_to_choice
    doc_to_text_module = f"lm_eval.tasks.{parse_task_spec(answering_task)[0]}.utils"

    predictions_per_input_doc = {}
    base_dataset = load_base_dataset_from_task(answering_task.replace(":", "_"))

    for reasoning_chain_list in reasoning_chains_per_document:

        dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_chain_list)

        raw_output = run_answering_for_dataset(
            args=args,
            answering_model= answering_model,
            answering_task_name= full_task_name,
            dataset_with_reasoning= dataset_with_reasoning,
            doc_to_text_module= doc_to_text_module
        )

        for sample in raw_output["samples"][full_task_name]:
            doc_id = sample["doc_id"]
            if doc_id not in predictions_per_input_doc:
                predictions_per_input_doc[doc_id] = {
                    "doc" : copy.deepcopy(sample["doc"]),
                    "preds": [],
                    "pred_probs": [],
                }

                predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"] = []

            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(sample["doc"]["Reasoning_Chain"])
            #predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Reasoning_Chain"], 100))

            pred_probs = [prob[0][0] for prob in sample["resps"]]

            predictions_per_input_doc[doc_id]["pred_probs"].append(pred_probs)    
            predictions_per_input_doc[doc_id]["preds"].append(pred_probs.index(max(pred_probs)))


    aggregated_metrics = {}

    for strat in [("majority", None), ("logits", None), ("condorcet", None), ("borda", None), ("rrf", None)]:
        strategy, top_k = strat
        aggregated_metrics[strategy + (f"_top{top_k}" if top_k is not None else "")] = aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, strategy=strategy, top_k=top_k)

    #reasoning_chains_per_document = {i : reasoning_chains_per_document[i] for i in range(len(reasoning_chains_per_document))}
    #mbr_metrics_results = MBR_process_reasoning(args, MBR_reasoning_chains(reasoning_chains_per_document), base_dataset, answering_model, full_task_name, doc_to_text_module)

    #aggregated_metrics.update(mbr_metrics_results["results"])


    return {
        "mode": "multi-turn_CoT-SC",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }

def only_vote(args: argparse.Namespace) -> Dict:
    pass
#    reasoning_model = args.reasoning_models[0]
#    answering_model = args.answering_models[0]
#
#    reasoning_task = args.reasoning_tasks[0]
#    answering_task = args.answering_tasks[0]
#
#    full_task_name = answering_task.replace(":", "_")
#    task_def = tasks.get_task_dict([full_task_name])[full_task_name]
#
#    full_task_name = answering_task.replace(":", "_")
#    task_def = tasks.get_task_dict([full_task_name])[full_task_name]
#    doc_to_choice = task_def.config.doc_to_choice
#    doc_to_text_module = f"lm_eval.tasks.{parse_task_spec(answering_task)[0]}.utils"
#    base_dataset = load_base_dataset_from_task(answering_task.replace(":", "_"))
#
#    predictions_per_input_doc = json.load(open(args.vote_file, "r"))["samples"]
#    aggregated_metrics = {}
#
#    for strat in [("majority", None), ("logits", None), ("condorcet", None), ("borda", None), ("rrf", None)]:
#        strategy, top_k = strat
#        aggregated_metrics[strategy + (f"_top{top_k}" if top_k is not None else "")] = aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, strategy=strategy, top_k=top_k)
#
#    mbr_metrics_results = MBR_process_reasoning(args, MBR_reasoning_chains(predictions_per_input_doc), base_dataset, answering_model, full_task_name, doc_to_text_module)
#    aggregated_metrics.update(mbr_metrics_results["results"])
#
#    return {
#        "mode": "multi-turn_CoT-SC",
#        "reasoning_model": reasoning_model,
#        "answering_model": answering_model,
#        "reasoning_task": reasoning_task,
#        "answering_task": answering_task,
#        "results": aggregated_metrics,
#        "samples" : predictions_per_input_doc
#    }


def aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, strategy="majority", top_k=None):
    """
    Aggregate predictions for a document using different strategies.

    Args:
        predictions_per_input_doc: dict mapping document IDs to their predicted class indices and probabilities
        doc_to_choice: mapping of class indices -> class names
        strategy: "majority" | "logits" | "topk_majority"
        top_k: number of first predictions to use (for "topk_majority")
    """

    results_per_doc = {}

    match strategy:
        case "majority":
            strategy_name = "majority" + (f"_top{top_k}" if top_k is not None else "")
            results_per_doc = majority_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=top_k, strategy_name=strategy_name)

        case "logits":
            strategy_name = "logits" + (f"_top{top_k}" if top_k is not None else "")
            results_per_doc = logits_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=top_k, strategy_name="logits")

        case "condorcet":
            strategy_name = "condorcet"
            results_per_doc = condorcet_aggregate_votes(
                predictions_per_input_doc, doc_to_choice, task_def,
                k=top_k, strategy_name=strategy_name
            )

        case "borda":
            strategy_name = "borda"
            results_per_doc = borda_aggregate_votes(
                predictions_per_input_doc, doc_to_choice, task_def,
                k=top_k, strategy_name=strategy_name
            )

        case "rrf":
            strategy_name = "rrf"
            results_per_doc = rrf_aggregate_votes(
                predictions_per_input_doc, doc_to_choice, task_def,
                k=top_k, strategy_name=strategy_name
            )

        case _:
            raise ValueError(f"Unknown strategy {strategy}")

    # Aggregate metrics across all docs using the task aggregation
    aggregated_metrics = {}
    for metric_name, agg_fn in task_def.aggregation().items():
        all_values = [results_per_doc[doc_id][metric_name] for doc_id in results_per_doc if metric_name in results_per_doc[doc_id]]
        if all_values:
            aggregated_metrics[metric_name] = agg_fn(all_values)
    return aggregated_metrics

def majority_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="majority"):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        top_k_preds = [p for p in (info["preds"][:k] if k is not None else info["preds"])]
        pred = majority_vote(top_k_preds)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]
        pred_probs = [0.0 for _ in doc_to_choice]

        # Aggrate probabilities by averaging them but only from the majority voted class
        for p, probs in zip(info["preds"][:k], info["pred_probs"][:k]):
            if p == pred:
                for i in range(len(doc_to_choice)):
                    pred_probs[i] += probs[i]
        pred_probs = [(p / top_k_preds.count(pred), False) for p in pred_probs]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)
    return results_per_doc_id

def logits_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="logits"):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        # Sum logits across all predictions
        summed_logits = [0.0 for _ in doc_to_choice]

        for probs in info["pred_probs"][:k]:
            for i in range(len(doc_to_choice)):
                summed_logits[i] += probs[i]

        pred = summed_logits.index(max(summed_logits))

        predictions_per_input_doc[doc_id]["logits"] = doc_to_choice[pred]

        pred_probs = [(p / len(info["pred_probs"]), False) for p in summed_logits]

        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)
    return results_per_doc_id

def condorcet_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="condorcet"):
    """
    Implements Condorcet voting:
    - Each candidate compared pairwise against others
    - Winner must beat every other candidate in head-to-head matchups
    - If no winner exists, fallback to majority voting
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        votes = info["preds"][:k] if k is not None else info["preds"]

        # Count pairwise wins
        num_classes = len(doc_to_choice)
        pairwise_wins = {c: 0 for c in range(num_classes)}
        pairwise_matches = {c: 0 for c in range(num_classes)}

        # Count pairwise outcomes
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                votes_i = sum(1 for v in votes if v == i)
                votes_j = sum(1 for v in votes if v == j)
                if votes_i > votes_j:
                    pairwise_wins[i] += 1
                elif votes_j > votes_i:
                    pairwise_wins[j] += 1
                # Every comparison counts as a match
                pairwise_matches[i] += 1
                pairwise_matches[j] += 1

        # Check for Condorcet winner
        condorcet_winner = None
        for c, wins in pairwise_wins.items():
            if wins == num_classes - 1:  # beat all others
                condorcet_winner = c
                break

        if condorcet_winner is None:
            # Fallback *per-doc*, not return for all docs
            majority_results = majority_aggregate_votes(
                {doc_id: info}, doc_to_choice, task_def, k=k, strategy_name="condorcet_fallback"
            )
            results_per_doc_id.update(majority_results)
            continue

        # Build probability distribution from pairwise win ratio
        scores = [
            pairwise_wins[c] / pairwise_matches[c] if pairwise_matches[c] > 0 else 0.0
            for c in range(num_classes)
        ]

        # Normalize scores to sum=1
        total_score = sum(scores)
        if total_score == 0:
            probs = [1.0 / num_classes] * num_classes  # fallback uniform
        else:
            probs = [s / total_score for s in scores]

        # Convert into (prob, is_predicted) tuples
        pred_probs = [(p, (i == condorcet_winner)) for i, p in enumerate(probs)]

        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[condorcet_winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id

def borda_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="borda"):
    """
    Implements Borda count aggregation:
    - Each prediction provides a ranking of classes based on probs
    - Assign points = (num_classes - rank)
    - Sum points across all predictions
    - Highest scoring candidate wins
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        num_classes = len(doc_to_choice)
        scores = [0.0 for _ in range(num_classes)]

        # Use top_k predictions if specified
        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]

        for probs in pred_probs_list:
            # Sort classes by probability (descending)
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                # Higher rank → more points (num_classes - rank - 1)
                scores[cls] += num_classes - rank - 1

        # Winner = argmax of scores
        winner = scores.index(max(scores))

        # Normalize scores to probability-like distribution
        total_score = sum(scores)
        probs = [s / total_score if total_score > 0 else 1.0 / num_classes for s in scores]

        # Convert into (prob, is_predicted) tuples
        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]

        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id

def rrf_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="rrf", rrf_k=60):
    """
    Implements Reciprocal Rank Fusion (RRF) aggregation:
    - Each ranked list contributes 1 / (rrf_k + rank) to a class score
    - Sum across all predictions
    - Winner = class with highest fused score
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        num_classes = len(doc_to_choice)
        scores = [0.0 for _ in range(num_classes)]

        # Use top_k predictions if specified
        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]

        for probs in pred_probs_list:
            # Sort classes by probability (descending)
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                scores[cls] += 1.0 / (rrf_k + rank + 1)  # rank is 0-based → use (rank+1)

        # Winner = argmax of scores
        winner = scores.index(max(scores))

        # Normalize scores into probability-like distribution
        total_score = sum(scores)
        probs = [s / total_score if total_score > 0 else 1.0 / num_classes for s in scores]

        # Convert into (prob, is_predicted) tuples
        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]

        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id

def MBR_reasoning_chains(reasoning_chains_per_document: Dict[str, List[str]]) -> Dict[str, List[List[str]]]:
    def precalc_bleu_stats(sentence, max_order=4):
        tokens = sentence.split()
        ngram_counts = {}
        for n in range(1, max_order + 1):
            ngram_counts[n] = Counter(ngrams(tokens, n))
        return ngram_counts, len(tokens)

    def process_doc(doc_id, decoder, candidates):
        precomputed = {i: precalc_bleu_stats(c) for i, c in enumerate(candidates[doc_id])}

        metric_output = decoder.decode(
            candidates[doc_id],
            candidates[doc_id],
            precomputed_stats=precomputed
        )

        return candidates[doc_id][metric_output.idx[0]]

    bleu = MetricBLEU(MetricBLEU.Config(num_workers=4))
    bleurt = MetricBLEURT(MetricBLEURT.Config(batch_size=64, model="lucadiliello/bleurt-tiny-128"))

    mbr_metrics = [("bleu", bleu, bleu.Config), ("bleurt", bleurt, bleurt.Config)]

    res = {"bleu" : [], "bleurt" : []}

    for name, metric, config in mbr_metrics:
        decoder = DecoderMBR(config, metric)

        if name == "bleu":
            for doc_id in tqdm(reasoning_chains_per_document, desc="BLEU"):
                metric_output = decoder.decode(
                    reasoning_chains_per_document[doc_id],
                    reasoning_chains_per_document[doc_id],
                )
                res[name].append(
                    reasoning_chains_per_document[doc_id][metric_output.idx[0]]
                )

        elif name == "bleurt":

            for doc_id in tqdm(reasoning_chains_per_document, desc="BLEURT"):
                # still per-doc, but BLEURT batches internally with batch_size=64
                metric_output = decoder.decode(
                    reasoning_chains_per_document[doc_id],
                    reasoning_chains_per_document[doc_id],
                )

                res[name].append(
                    reasoning_chains_per_document[doc_id][metric_output.idx[0]]
                )

    return res


def MBR_process_reasoning(args: argparse.Namespace, mbr_reasoning_chains: Dict[str, List[str]], base_dataset, answering_model, full_task_name, doc_to_text_module) -> Dict:

    metrics_results = {"results" : {mbr_metric: {} for mbr_metric in mbr_reasoning_chains}}

    for mbr_metric in mbr_reasoning_chains:
        reasoning_chain_list = mbr_reasoning_chains[mbr_metric]

        dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_chain_list)

        raw_output = run_answering_for_dataset(
            args=args,
            answering_model= answering_model,
            answering_task_name= full_task_name,
            dataset_with_reasoning= dataset_with_reasoning,
            doc_to_text_module= doc_to_text_module
        )

        metrics_results["results"][mbr_metric] = format_results_dict(raw_output["results"][full_task_name])

    return metrics_results