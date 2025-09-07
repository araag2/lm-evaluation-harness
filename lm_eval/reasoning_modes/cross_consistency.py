import copy
import argparse
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
            print(f"\n[Verification] Running reasoning model {reasoning_model} as verifier for chains from {answering_model}")

            raw_output = run_answering_for_dataset(
                args=args,
                answering_model= reasoning_model,
                answering_task_name= reasoning_full_task_name,
                dataset_with_reasoning= reasoning_dataset,
                doc_to_text_module=  f"lm_eval.tasks.{parse_task_spec(reasoning_task)[0]}.utils",
                doc_to_text_func_name= "doc_to_text_verify_reasoning"
            )

            per_model_reasoning_verification_outputs[reasoning_model][answering_model] = inject_reasoning_into_dataset(reasoning_dataset, raw_output["samples"][reasoning_full_task_name], reasoning_field="Verified_Reasoning_Chain")

    print(f"[Step 3: Aggregation] Aggregating {len(args.reasoning_models)}x{len(args.answering_models)} verification outputs per document.")

    # Step 3: Collect all MxM predicted distributions and perform an aggregation + majority voting.

    predictions_per_input_doc = {}

    for reasoning_model in args.reasoning_models:
        for answering_model in args.answering_models:
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
                        "preds": [],
                        "pred_probs": [],
                    }

                    predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"] = []
                    predictions_per_input_doc[doc_id]["doc"]["Verified_Reasoning_Chains"] = []

                predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Reasoning_Chain"]), 100)

                predictions_per_input_doc[doc_id]["doc"]["Verified_Reasoning_Chains"].append(shorten_reasoning_chain(sample["doc"]["Verified_Reasoning_Chain"]), 100)

                pred_probs = [prob[0][0] for prob in sample["resps"]]

                predictions_per_input_doc[doc_id]["pred_probs"].append(pred_probs)    
                predictions_per_input_doc[doc_id]["preds"].append(pred_probs.index(max(pred_probs)))


    print(f"[Step 4: Metrics] Aggregating metrics for task {answering_task} using {args.reasoning_models} reasoning models as verifiers.")

    task_def = tasks.get_task_dict([answering_task_full_task_name])[answering_task_full_task_name]
    doc_to_choice = task_def.config.doc_to_choice

    # Step 4: Aggregate metrics across all docs using the task aggregation

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
        "mode": "cross_consistency",
        "reasoning_models": args.reasoning_models,
        "verifier_models": args.answering_models,
        "answering_models": args.answering_models,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }