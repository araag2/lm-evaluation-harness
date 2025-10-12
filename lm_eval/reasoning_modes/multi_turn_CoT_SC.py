from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.simple_voting_modes import simple_voting_modes
from lm_eval.reasoning_modes.voting.mbr_voting_modes import mbr_voting_modes

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

            pred_probs = [prob[0][0] for prob in sample["resps"]]

            predictions_per_input_doc[doc_id]["pred_probs"].append(pred_probs)    
            predictions_per_input_doc[doc_id]["preds"].append(pred_probs.index(max(pred_probs)))


    aggregated_metrics = simple_voting_modes(
        predictions_per_input_doc=predictions_per_input_doc,
        doc_to_choice=doc_to_choice,
        task_def=task_def,
    )

    #mbr_metrics = mbr_voting_modes(
    #    args=args,
    #    predictions_per_input_doc=predictions_per_input_doc,
    #    base_dataset=base_dataset,
    #    answering_model=answering_model,
    #    answering_task=full_task_name,
    #    doc_to_text_module=doc_to_text_module,
    #    doc_to_choice=doc_to_choice,
    #    task_def=task_def
    #)
    #
    #aggregated_metrics.update(mbr_metrics["results"])


    return {
        "mode": "multi-turn_CoT-SC",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }