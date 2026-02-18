from lm_eval.reasoning_modes.reasoning_utils import *

def mode_multi_turn_CoT(args: argparse.Namespace) -> Dict:
    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning model and one answering model. {args.reasoning_models=} {args.answering_models=}")

    if len(args.answering_tasks) != 1:
        print(f"[WARNING] For multi-turn_CoT mode exactly one answering task is supported. Using first: {args.answering_tasks[0]}")

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]
    reasoning_task  = args.reasoning_tasks[0]
    answering_task_spec = args.answering_tasks[0]

    task_base_name, _ = parse_task_spec(answering_task_spec)
    full_task_name    = answering_task_spec.replace(":", "_")
    doc_to_text_module = f"lm_eval.tasks.{task_base_name}.utils"

    reasoning_outputs      = run_reasoning(args)[reasoning_model][reasoning_task]
    base_dataset           = load_base_dataset_from_task(reasoning_task.replace(":", "_"))
    dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_outputs)

    raw_output = run_answering_for_dataset(
        args=args,
        answering_model=answering_model,
        answering_task_name=full_task_name,
        dataset_with_reasoning=dataset_with_reasoning,
        doc_to_text_module=doc_to_text_module
    )

    task_def      = tasks.get_task_dict([full_task_name])[full_task_name]
    doc_to_choice = task_def.config.doc_to_choice

    predictions_per_input_doc = extract_predictions_from_samples(
        raw_output["samples"][full_task_name], doc_to_choice
    )

    return {
        "mode":           "multi-turn_CoT",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task":  reasoning_task,
        "answering_task":  answering_task_spec,
        "results":         format_results_dict(raw_output["results"][full_task_name]),
        "samples":         predictions_per_input_doc,
    }