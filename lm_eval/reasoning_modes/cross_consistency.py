from lm_eval.reasoning_modes.reasoning_utils import *

def mode_cross_consistency(args: argparse.Namespace) -> Dict:
    """
    N models produce reasoning for each reasoning task.
    M models answer each answering task consuming the reasoning produced by every (model, task) from Reasoning Generation.
    """
    reasoning_outputs = run_reasoning(args)

    dataset_with_reasoning_per_source = {}

    # Step 1: Generate Reasoning Source
    for reasoning_task in args.reasoning_tasks:
        # Load base dataset once for this task
        base_dataset = load_base_dataset_from_task(reasoning_task)

        for reasoning_model in args.reasoning_models:
            samples = reasoning_outputs[reasoning_model][reasoning_task]
            enriched = inject_reasoning_into_dataset(base_dataset, samples)

            tbase, _ = parse_task_spec(reasoning_task)

            if tbase not in dataset_with_reasoning_per_source:
                dataset_with_reasoning_per_source[tbase] = {}

            dataset_with_reasoning_per_source[tbase][(reasoning_model, reasoning_task)] = enriched

    # Step 2: Run Answering
    all_results = {}

    for answering_model in args.answering_models:
        all_results[answering_model] = {}
        for answering_task_spec in args.answering_tasks:
            tbase, _ = parse_task_spec(answering_task_spec)
            full_task_name = answering_task_spec.replace(":", "_")

            doc_to_text_module = f"lm_eval.tasks.{tbase}.utils"

            all_results[answering_model][full_task_name] = {}

            for (reasoning_task, reasoning_model), enriched in dataset_with_reasoning_per_source[tbase].items():

                out = run_answering_for_dataset(
                    args=args,
                    answering_model=answering_model,
                    answering_task_name=full_task_name,
                    enriched_dataset=enriched,
                    doc_to_text_module=doc_to_text_module
                )

                all_results[answering_model][full_task_name][(reasoning_model, reasoning_task)] = out

    return {
        "mode": "cross-consistency",
        "reasoning_models": args.reasoning_models,
        "answering_models": args.answering_models,
        "reasoning_tasks": args.reasoning_tasks,
        "answering_tasks": args.answering_tasks,
        "results": all_results,
    }