from lm_eval.reasoning_modes.reasoning_utils import *

def mode_multi_turn_CoT(args: argparse.Namespace):
    """Multi-turn CoT pipeline.

    Supports batched evaluation: all task pairs are processed with a single
    reasoning model load and a single answering model load, regardless of how
    many task pairs are provided.

    Returns a list of result dicts (one per task pair) when multiple pairs are
    given, or a single dict for backward compatibility when only one pair is given.
    """
    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning model and one answering model. {args.reasoning_models=} {args.answering_models=}")

    if len(args.reasoning_tasks) != len(args.answering_tasks):
        raise ValueError(
            f"reasoning_tasks and answering_tasks must have the same length. "
            f"Got {len(args.reasoning_tasks)} vs {len(args.answering_tasks)}."
        )

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]

    # --- Reasoning step: ALL tasks in ONE model load ---
    all_reasoning_outputs = run_reasoning(args)[reasoning_model] 

    # Build per-task metadata and (task_name, dataset) pairs for batch answering
    tasks_and_datasets = []
    doc_to_text_modules = []
    task_meta = []

    for reasoning_task, answering_task_spec in zip(args.reasoning_tasks, args.answering_tasks):
        task_base_name, _ = parse_task_spec(answering_task_spec)
        full_task_name    = answering_task_spec.replace(":", "_")
        doc_to_text_module = f"lm_eval.tasks.{task_base_name}.utils"

        reasoning_outputs      = all_reasoning_outputs[reasoning_task]
        base_dataset           = load_base_dataset_from_task(answering_task_spec.replace(":", "_"))
        dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_outputs)

        tasks_and_datasets.append((full_task_name, dataset_with_reasoning))
        doc_to_text_modules.append(doc_to_text_module)
        task_meta.append((reasoning_task, answering_task_spec, full_task_name))

    # --- Answering step: ALL tasks in ONE model load ---
    raw_output = run_answering_for_datasets(
        args=args,
        answering_model=answering_model,
        tasks_and_datasets=tasks_and_datasets,
        doc_to_text_module=doc_to_text_modules,
    )

    results_list = []
    for reasoning_task, answering_task_spec, full_task_name in task_meta:
        task_def      = get_task(full_task_name)
        doc_to_choice = task_def.config.doc_to_choice

        predictions_per_input_doc = extract_predictions_from_samples(
            raw_output["samples"][full_task_name], doc_to_choice
        )
        results_list.append({
            "mode":            "multi-turn_CoT",
            "reasoning_model": reasoning_model,
            "answering_model": answering_model,
            "reasoning_task":  reasoning_task,
            "answering_task":  answering_task_spec,
            "results":         format_results_dict(raw_output["results"][full_task_name]),
            "samples":         predictions_per_input_doc,
        })

    # Backward compatibility: return a plain dict when only one pair was given
    return results_list[0] if len(results_list) == 1 else results_list