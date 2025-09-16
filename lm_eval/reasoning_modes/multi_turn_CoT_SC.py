
from lm_eval.reasoning_modes.reasoning_utils import *

def mode_multi_turn_CoT_SC(args: argparse.Namespace) -> Dict:
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

    doc_to_text_module = f"lm_eval.tasks.{parse_task_spec(answering_task)[0]}.utils"

    predictions_per_input_doc = {}
    doc_to_choice = task_def.config.doc_to_choice

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

            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Reasoning_Chain"], 100))

            pred_probs = [prob[0][0] for prob in sample["resps"]]

            predictions_per_input_doc[doc_id]["pred_probs"].append(pred_probs)    
            predictions_per_input_doc[doc_id]["preds"].append(pred_probs.index(max(pred_probs)))

        aggregated_metrics = {}

        for strat in [("majority", None), ("majority", 3)]:
            strategy, top_k = strat
            aggregated_metrics[strategy + (f"_top{top_k}" if top_k is not None else "")] = aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, strategy=strategy, top_k=top_k)


    return {
        "mode": "multi-turn_CoT-SC",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }

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
            pass

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