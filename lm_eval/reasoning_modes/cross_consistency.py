import copy
import argparse
from email.policy import default
import json
from typing import Dict, List, Any


from lm_eval import tasks
from lm_eval.reasoning_modes.reasoning_utils import *


def mode_cross_consistency(args: argparse.Namespace) -> Dict:
    """Cross-consistency pipeline:
    1) Run each reasoning model to generate a reasoning chain for every doc.
    2) For every (origin_model, doc) chain, run every model as a *verifier* over that chain.
    (So with M models you get MxM verifications per doc.)
    3) Collect all MxM predicted distributions and perform an aggregation + majority voting.


    This function assumes the same task names as other modes and relies on helpers in
    reasoning_utils.py (run_reasoning, inject_reasoning_into_dataset, run_answering_for_dataset, etc.).
    """

    if args.vote_file is not None:
        print(f"[INFO] Using vote_file {args.vote_file} to load predictions and skip reasoning/answering steps.")
        predictions_per_input_doc = json.load(open(args.vote_file, "r"))["samples"]

        return {
            "mode": "cross_consistency",
            "reasoning_models": args.reasoning_models,
            "verifier_models": args.answering_models,
            "answering_models": args.answering_models,
            "reasoning_task": args.reasoning_tasks[0],
            "answering_task": args.answering_tasks[0],
            "results": only_vote(args, predictions_per_input_doc),
            "samples" : predictions_per_input_doc
        }


    if len(args.reasoning_models) < 1 or len(args.answering_models) < 1:
        print("[WARNING] Provide at least one reasoning model and one answering model.")

    # For cross-consistency the answering models list is unused (we verify using reasoning models)


    if len(args.reasoning_tasks) != 1 or len(args.answering_tasks) != 1:
        print(f"[WARNING] For cross-consistency please provide exactly one reasoning task and one answering task. {args.reasoning_tasks=} {args.answering_tasks=}")


    reasoning_task = args.reasoning_tasks[0]
    reasoning_full_task_name = reasoning_task.replace(":", "_")

    answering_task = args.answering_tasks[0]
    answering_task_full_task_name = answering_task.replace(":", "_")

    print(f"[Step 1: Reasoning] Running reasoning models: {args.reasoning_models} on task {reasoning_task}")
    reasoning_outputs_per_model = run_reasoning(args) # dict: model -> task -> [samples]
    
    per_model_datasets_inject = {}

    # Step 1: Run reasoning for every reasoning model and get datasets with reasoning chains injected.
    base_dataset = load_base_dataset_from_task(reasoning_full_task_name)
    for model_name in args.reasoning_models:
        samples = reasoning_outputs_per_model[model_name][reasoning_task]

        per_model_datasets_inject[model_name] = inject_reasoning_into_dataset(base_dataset, samples)


    print(f"[Step 2: Verification] Running verification for models: {args.reasoning_models} on task {reasoning_task}")

    per_model_reasoning_verification_outputs = {model_name: {} for model_name in args.reasoning_models}
    
    # Step 2: For each (model, doc) chain, run every model as a verifier.
    for reasoning_model in args.reasoning_models:
        reasoning_dataset = per_model_datasets_inject[reasoning_model]

        for answering_model in args.answering_models:
            print(f"\n[Verification] Running answering model {answering_model} to verify reasoning chains from model {reasoning_model}.")

            raw_output = run_answering_for_dataset(
                args=args,
                answering_model= answering_model,
                answering_task_name= reasoning_full_task_name,
                dataset_with_reasoning= reasoning_dataset,
                doc_to_text_module=  f"lm_eval.tasks.{parse_task_spec(reasoning_task)[0]}.utils",
                doc_to_text_func_name= "doc_to_text_verify_reasoning"
            )

            #TO:DO Need to wrap this dataset into a "test" split dataset dict so it work correctly with inject_reasoning_into_dataset. Also double check the reasoning outputs to see if they work correctly.

            print(f'{reasoning_dataset}, {raw_output["samples"][reasoning_full_task_name]}')

            per_model_reasoning_verification_outputs[reasoning_model][answering_model] = inject_reasoning_into_dataset(reasoning_dataset, raw_output["samples"][reasoning_full_task_name], reasoning_field="Verified_Reasoning_Chain")

    print(f"[Step 3: Aggregation] Aggregating {len(args.reasoning_models)}x{len(args.answering_models)} verification outputs per document.")

    # Step 3: Collect all MxM predicted distributions and perform an aggregation + majority voting.

    predictions_per_input_doc = {}

    for reasoning_model in args.reasoning_models:
        for answering_model in args.answering_models:
            print(f"\n[Aggregation] Using {reasoning_model} to verify reasoning chain with original reasoning from {reasoning_model} and verification from {answering_model}.")

            dataset_with_reasoning_verification = per_model_reasoning_verification_outputs[reasoning_model][answering_model]

            raw_output = run_answering_for_dataset(
                args=args,
                answering_model= reasoning_model,
                answering_task_name= answering_task_full_task_name,
                dataset_with_reasoning= dataset_with_reasoning_verification,
                doc_to_text_module= f"lm_eval.tasks.{parse_task_spec(answering_task)[0]}.utils",
                doc_to_text_func_name= "doc_to_text_answer_selection_after_verify_reasoning"
            )

            for sample in raw_output["samples"][answering_task_full_task_name]:
                doc_id = sample["doc_id"]

                if doc_id not in predictions_per_input_doc:

                    predictions_per_input_doc[doc_id] = {
                        "doc": copy.deepcopy(sample["doc"]),
                        "preds": {}
                    }

                    predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"] = []
                    predictions_per_input_doc[doc_id]["doc"]["Verified_Reasoning_Chains"] = []

                predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Reasoning_Chain"], 100))

                predictions_per_input_doc[doc_id]["doc"]["Verified_Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Verified_Reasoning_Chain"], 100))

                pred_probs = [prob[0][0] for prob in sample["resps"]]

                if reasoning_model not in predictions_per_input_doc[doc_id]["preds"]:
                    predictions_per_input_doc[doc_id]["preds"][reasoning_model] = []

                predictions_per_input_doc[doc_id]["preds"][reasoning_model].append((pred_probs.index(max(pred_probs)), pred_probs))


    print(f"[Step 4: Metrics] Aggregating metrics for task {answering_task} using {args.reasoning_models} reasoning models as verifiers.")

    return {
        "mode": "cross_consistency",
        "reasoning_models": args.reasoning_models,
        "verifier_models": args.answering_models,
        "answering_models": args.answering_models,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        #"results": only_vote(args, predictions_per_input_doc),
        "samples" : predictions_per_input_doc
    }

# -------------------------
# Vote Wrappers
# -------------------------

def only_vote(args: argparse.Namespace, predictions_per_input_doc: Dict) -> Dict:
    if args.vote_file is not None:
        predictions_per_input_doc = json.load(open(args.vote_file, "r"))["samples"]

    answering_task_full_task_name = args.answering_tasks[0].replace(":", "_")
    task_def = tasks.get_task_dict([answering_task_full_task_name])[answering_task_full_task_name]
    results_per_doc = aggregate_doc_predictions(predictions_per_input_doc, task_def)

    # Aggregate metrics across all docs using the task aggregation
    aggregated_metrics = {}
    for metric_name, agg_fn in task_def.aggregation().items():

        if metric_name not in results_per_doc["flat_preds_maj"]["0"]:
            continue

        for result_key in results_per_doc:
            all_values = []
            for doc_id in results_per_doc[result_key]:
                all_values.append(results_per_doc[result_key][doc_id][metric_name])

            if result_key not in aggregated_metrics:
                aggregated_metrics[result_key] = {}
            aggregated_metrics[result_key][metric_name] = agg_fn(all_values)

    return aggregated_metrics


# -------------------------
# Process Votes
# -------------------------

def single_doc_aggregate_votes(preds: List[int], doc_to_choice : str, strategy: str = "majority") -> list:
    win_pred = None

    match strategy:
        case "majority":
            win_pred = majority_vote([p for p, _ in preds])

        case "logits":
            # Sum logits across predictions
            summed = [0.0 for _ in doc_to_choice]
            for _, probs in preds:
                for i, p in enumerate(probs):
                    summed[i] += p if isinstance(p, float) else p[0]
            win_pred = max(range(len(summed)), key=lambda i: summed[i])
            vote_probs = [(s / len(preds), i == win_pred) for i, s in enumerate(summed)]
            return (win_pred, vote_probs)

        case "condorcet":
            win_pred = condorcet_vote([probs for _, probs in preds], len(doc_to_choice))

        case "borda":
            win_pred = borda_vote([probs for _, probs in preds], len(doc_to_choice))

        case "rrf":
            win_pred = rrf_vote([probs for _, probs in preds], len(doc_to_choice))

        case _:
            raise ValueError(f"Unknown strategy {strategy}")

    # Default majority-style probs for majority/condorcet/borda
    vote_probs = [0.0 for _ in doc_to_choice]
    for pred, pred_probs in preds:
        if pred == win_pred:
            for i in range(len(doc_to_choice)):
                vote_probs[i] += pred_probs[i] if isinstance(pred_probs[i], float) else pred_probs[i][0]

    right_preds = [pred for pred, _ in preds if pred == win_pred]
    vote_probs = [(p / len(right_preds) if right_preds != [] else 0, i == win_pred) 
                  for i, p in enumerate(vote_probs)]
    return (win_pred, vote_probs)

def condorcet_vote(pred_probs_list, num_classes):
    pairwise_wins = [0] * num_classes
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            wins_i = sum(1 for probs in pred_probs_list if probs[i] > probs[j])
            wins_j = len(pred_probs_list) - wins_i
            if wins_i > wins_j:
                pairwise_wins[i] += 1
            elif wins_j > wins_i:
                pairwise_wins[j] += 1
    return max(range(num_classes), key=lambda c: pairwise_wins[c])


def borda_vote(pred_probs_list, num_classes):
    scores = [0] * num_classes
    for probs in pred_probs_list:
        ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
        for rank, cls in enumerate(ranked):
            scores[cls] += num_classes - rank - 1
    return max(range(num_classes), key=lambda i: scores[i])

def rrf_vote(pred_probs_list, num_classes, rrf_k=60):
    scores = [0.0] * num_classes

    for probs in pred_probs_list:
        # Rank classes descending by probability
        ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
        for rank, cls in enumerate(ranked):
            scores[cls] += 1.0 / (rrf_k + rank + 1)  # rank is 0-based

    winner = max(range(num_classes), key=lambda i: scores[i])
    return winner

def aggregate_votes(preds: List[int], doc_to_choice: str, strategy="majority") -> list:
    return single_doc_aggregate_votes(preds, doc_to_choice, strategy=strategy)

def aggregate_doc_predictions(predictions_per_input_doc: Dict[str, Dict], task_def) -> None:
    # predictions_per_input_doc[doc_id] = {
    #     "preds": {model_name: [pred1, pred2, ...], ...},
    #     "pred_probs": {model_name: [[prob1, prob2, ...

    doc_to_choice = task_def.config.doc_to_choice

    results_per_doc = { "flat_preds_maj": {doc_id : None for doc_id in predictions_per_input_doc},
                        "flat_preds_logits": {doc_id : None for doc_id in predictions_per_input_doc},
                        "flat_preds_condorcet": {doc_id : None for doc_id in predictions_per_input_doc},
                        "flat_preds_borda": {doc_id : None for doc_id in predictions_per_input_doc}, 
                        "flat_preds_rrf": {doc_id : None for doc_id in predictions_per_input_doc}, 
                        #"per_reasoning_model_preds": {doc_id : None for doc_id in predictions_per_input_doc},
                        #"per_answering_model_preds": {doc_id : None for doc_id in predictions_per_input_doc}
                    }

    for doc_id, info in predictions_per_input_doc.items():
        flat_res = []

        # TODO: Why is this crashing

        per_reasoning_model_res = [[] for _ in range(len(info["preds"]))]
        per_answering_model_res = [[] for _ in range(len(next(iter(info["preds"].values()))))]

        reasoning_models = list(info["preds"].keys())

        # Group results correctly
        for i, reasoning_model in enumerate(reasoning_models):
            for j in range(len(info["preds"][reasoning_model])):
                flat_res.append(info["preds"][reasoning_model][j])
                per_reasoning_model_res[i].append(info["preds"][reasoning_model][j])
                per_answering_model_res[j].append(info["preds"][reasoning_model][j])

        results_per_doc["flat_preds_maj"][doc_id] = task_def.process_results(info["doc"], aggregate_votes(flat_res, doc_to_choice, strategy="majority")[1])
        results_per_doc["flat_preds_logits"][doc_id] = task_def.process_results(info["doc"], aggregate_votes(flat_res, doc_to_choice, strategy="logits")[1])
        results_per_doc["flat_preds_condorcet"][doc_id] = task_def.process_results(info["doc"], aggregate_votes(flat_res, doc_to_choice, strategy="condorcet")[1])
        results_per_doc["flat_preds_borda"][doc_id] = task_def.process_results(info["doc"], aggregate_votes(flat_res, doc_to_choice, strategy="borda")[1])
        results_per_doc["flat_preds_rrf"][doc_id] = task_def.process_results(info["doc"], aggregate_votes(flat_res, doc_to_choice, strategy="rrf")[1])
        
        #results_per_doc["per_reasoning_model_preds_maj"][doc_id] = aggregate_votes(per_reasoning_model_res, doc_to_choice, strategy="majority")
        #results_per_doc["per_answering_model_preds_maj"][doc_id] = aggregate_votes(per_answering_model_res, doc_to_choice, strategy="majority")
                                   
    return results_per_doc