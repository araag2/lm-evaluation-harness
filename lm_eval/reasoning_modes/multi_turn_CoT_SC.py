from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.voting_modes import run_voting_modes

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

        samples = raw_output["samples"][full_task_name]
        extract_predictions_from_samples(samples, doc_to_choice, existing=predictions_per_input_doc)

        # Accumulate reasoning chains (SC-specific: one chain per iteration)
        for sample in samples:
            doc_id = sample["doc_id"]
            if "Reasoning_Chains" not in predictions_per_input_doc[doc_id]["doc"]:
                predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"] = []
            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(sample["doc"]["Reasoning_Chain"])

    aggregated_metrics = run_voting_modes(
        args,
        predictions_per_input_doc,
        doc_to_choice,
        task_def,
        base_dataset=base_dataset,
        answering_model=answering_model,
        answering_task=full_task_name,
        doc_to_text_module=doc_to_text_module,
    )

    return {
        "mode": "multi-turn_CoT-SC",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }