
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

    print(f'Reasoning chain lists: {reasoning_chains_per_document}')

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

            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Reasoning_Chain"]), 100)

            pred_probs = [prob[0][0] for prob in sample["resps"]]

            predictions_per_input_doc[doc_id]["pred_probs"].append(pred_probs)    
            predictions_per_input_doc[doc_id]["preds"].append(pred_probs.index(max(pred_probs)))

    results_per_doc = {}
    for doc_id, info in predictions_per_input_doc.items():
        pred = majority_vote(info["preds"])
        predictions_per_input_doc[doc_id]["majority_pred"] = doc_to_choice[pred]
        pred_probs = [0.0 for _ in doc_to_choice]

        # Aggrate probabilities by averaging them but only from the majority voted class
        for p, probs in zip(info["preds"], info["pred_probs"]):
            if p == pred:
                for i in range(len(doc_to_choice)):
                    pred_probs[i] += probs[i]
        pred_probs = [(p / info["preds"].count(pred), False) for p in pred_probs]

        results_per_doc[doc_id] = task_def.process_results(info["doc"], pred_probs)

    # Aggregate metrics across all docs using the task aggregation
    aggregated_metrics = {}
    for metric_name, agg_fn in task_def.aggregation().items():
        all_values = [results_per_doc[doc_id][metric_name] for doc_id in results_per_doc if metric_name in results_per_doc[doc_id]]
        if all_values:
            aggregated_metrics[metric_name] = agg_fn(all_values)

    return {
        "mode": "multi-turn_CoT-SC",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }