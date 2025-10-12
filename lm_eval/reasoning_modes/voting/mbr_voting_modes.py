from lm_eval.reasoning_modes.reasoning_utils import *
from mbrs.metrics import MetricBLEU, MetricBLEURT
from mbrs.decoders import DecoderMBR
from tqdm import tqdm

from lm_eval.reasoning_modes.voting.simple_voting_modes import aggregate_metrics_per_strategy

def mbr_voting_modes(args: argparse.Namespace, predictions_per_input_doc : Dict, base_dataset : Dataset, answering_model : str, answering_task : str, doc_to_text_module : object, doc_to_choice : Dict, task_def : object) -> Dict:
    reasoning_chains_per_document = {}
    for doc_id, info in predictions_per_input_doc.items():
        reasoning_chains_per_document[doc_id] = info["doc"]["Reasoning_Chains"]

    mbr_reasoning_chains, mbr_weighted_chains_by_metric = get_mbr_reasoning_chains(reasoning_chains_per_document)

    metrics_results = {"results" : {mbr_metric: {} for mbr_metric in mbr_reasoning_chains}}

    for mbr_metric in mbr_reasoning_chains:
        reasoning_chain_list = mbr_reasoning_chains[mbr_metric]

        if args.limit is not None:
            base_dataset["test"] = base_dataset["test"].select(range(args.limit))
        dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_chain_list)

        raw_output = run_answering_for_dataset(
            args=args,
            answering_model= answering_model,
            answering_task_name= answering_task,
            dataset_with_reasoning= dataset_with_reasoning,
            doc_to_text_module= doc_to_text_module
        )

        metrics_results["results"][mbr_metric] = format_results_dict(raw_output["results"][answering_task])

    mbr_weighted_chains_scores = vote_weighted_chains_by_metric(mbr_weighted_chains_by_metric, predictions_per_input_doc, doc_to_choice, task_def)
    metrics_results["results"].update(mbr_weighted_chains_scores)

    return metrics_results

def get_mbr_reasoning_chains(reasoning_chains_per_document):
    def normalize_scores(scores : list) -> list:
        min_scores = min(scores)
        norm_factor = max(scores) - min_scores + 1e-8
        return [(score - min_scores) / norm_factor for score in scores]

    bleu = MetricBLEU(MetricBLEU.Config(num_workers=32))
    bleurt = MetricBLEURT(MetricBLEURT.Config(batch_size=32, model="lucadiliello/bleurt-tiny-128"))

    #mbr_metrics = [("bleu", bleu, bleu.Config), ("bleurt", bleurt, bleurt.Config)]
    #res = {"bleu" : [], "bleurt" : []}
    #mbr_aggregate_vote_scores = {"bleu" : [], "bleurt" : []}
    mbr_metrics = [("bleurt", bleurt, bleurt.Config)]
    res = {"bleurt" : []}
    mbr_aggregate_vote_scores = {"bleurt" : []}

    for name, metric, config in mbr_metrics:
        decoder = DecoderMBR(config, metric)

        if name == "bleu":
            for doc_id in tqdm(reasoning_chains_per_document, desc="BLEU"):

                metric_output = decoder.decode(
                    reasoning_chains_per_document[doc_id],
                    reasoning_chains_per_document[doc_id],
                    nbest=len(reasoning_chains_per_document[doc_id])
                )
                res[name].append(
                    reasoning_chains_per_document[doc_id][metric_output.idx[0]]
                )

                mbr_aggregate_vote_scores[name].append(normalize_scores(metric_output.score))


        elif name == "bleurt":

            for doc_id in tqdm(reasoning_chains_per_document, desc="BLEURT"):
                metric_output = decoder.decode(
                    reasoning_chains_per_document[doc_id],
                    reasoning_chains_per_document[doc_id],
                    nbest=len(reasoning_chains_per_document[doc_id])
                )

                res[name].append(
                    reasoning_chains_per_document[doc_id][metric_output.idx[0]]
                )

                mbr_aggregate_vote_scores[name].append(normalize_scores(metric_output.score))

    return res, mbr_aggregate_vote_scores

def vote_weighted_chains_by_metric(mbr_weighted_chains_scores : Dict, predictions_per_input_doc : Dict, doc_to_choice : List, task_def : object) -> Dict:
    def weighted_majority_vote(predictions: list, weights : list) -> int:
        """Aggregate predictions using weightedmajority voting and return the winning index."""
        weight_per_prec = {}
        for p, w in zip(predictions, weights):
            if p not in weight_per_prec:
                weight_per_prec[p] = 0.0
            weight_per_prec[p] += w
        return max(weight_per_prec, key=weight_per_prec.get)

    res_per_metric = {}
    for metric in mbr_weighted_chains_scores:
        res = {}
        weight_scores_by_doc = mbr_weighted_chains_scores[metric]

        # need to iterate over weighted scores and predictions together

        for ((doc_id, info), weight_scores) in zip(predictions_per_input_doc.items(), weight_scores_by_doc):
            top_k_preds = [p for p in info["preds"]]
            pred = weighted_majority_vote(top_k_preds, weight_scores)

            predictions_per_input_doc[doc_id][f"{metric}_weight_logits"] = doc_to_choice[pred]
            pred_probs = [0.0 for _ in doc_to_choice]

            # Aggregate probabilities by averaging them but only from the majority voted class
            for p, probs, score in zip(info["preds"], info["pred_probs"], weight_scores):
                if p == pred:
                    for i in range(len(doc_to_choice)):
                        pred_probs[i] += probs[i] * score
            sum_weight_scores = sum(weight_scores)
            pred_probs = [(p / sum_weight_scores if sum_weight_scores != 0 else p, False) for p in pred_probs]
            res[doc_id] = task_def.process_results(info["doc"], pred_probs)

        res_per_metric[f"{metric}_weight_logits"] = aggregate_metrics_per_strategy(res, task_def)

    return res_per_metric


def weighted_majority_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="majority"):
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

