import copy
import argparse
import json
# typing aliases (Dict, List, Any, Tuple) are re-exported from reasoning_utils via *-import
from lm_eval import tasks
from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.voting_modes import simple_voting_modes


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

        doc_info = json.load(open(args.vote_file, "r"))
        predictions_per_input_doc = doc_info["samples"]
        answering_task = doc_info.get("answering_task", args.answering_tasks[0])
        full_task_name = answering_task.replace(":", "_")
        task_def = tasks.get_task_dict([full_task_name])[full_task_name]
        doc_to_choice = task_def.config.doc_to_choice
        flat_preds = flatten_cross_consistency_preds(predictions_per_input_doc)

        return {
            "mode": "cross_consistency",
            "reasoning_models": args.reasoning_models,
            "verifier_models": args.answering_models,
            "answering_models": args.answering_models,
            "reasoning_task": args.reasoning_tasks[0],
            "answering_task": answering_task,
            "results": simple_voting_modes(flat_preds, doc_to_choice, task_def),
            "samples": predictions_per_input_doc
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

    task_def = tasks.get_task_dict([answering_task_full_task_name])[answering_task_full_task_name]
    doc_to_choice = task_def.config.doc_to_choice
    flat_preds = flatten_cross_consistency_preds(predictions_per_input_doc)

    return {
        "mode": "cross_consistency",
        "reasoning_models": args.reasoning_models,
        "verifier_models": args.answering_models,
        "answering_models": args.answering_models,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": simple_voting_modes(flat_preds, doc_to_choice, task_def),
        "samples": predictions_per_input_doc
    }


# -------------------------
# Cross-consistency helpers
# -------------------------

def flatten_cross_consistency_preds(predictions_per_input_doc: Dict) -> Dict:
    """
    Convert cross-consistency predictions_per_input_doc (where preds is a
    dict keyed by reasoning_model, each value a list of (pred_idx, pred_probs))
    into the standard flat format expected by simple_voting_modes:
      doc["preds"]      -> flat list of pred indices
      doc["pred_probs"] -> flat list of prob lists
    Returns a new dict; does not mutate the input.
    """
    flat = {}
    for doc_id, info in predictions_per_input_doc.items():
        flat_preds = []
        flat_probs = []
        for model_preds in info["preds"].values():
            for pred_idx, pred_probs in model_preds:
                flat_preds.append(pred_idx)
                flat_probs.append(pred_probs)
        flat[doc_id] = {
            **info,
            "preds": flat_preds,
            "pred_probs": flat_probs,
        }
    return flat