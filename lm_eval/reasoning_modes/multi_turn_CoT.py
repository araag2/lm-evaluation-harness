from lm_eval.reasoning_modes.reasoning_utils import *

def mode_multi_turn_CoT(args: argparse.Namespace) -> Dict:
    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning model and one answering model. {args.reasoning_models=} {args.answering_models=}")

    if len(args.reasoning_tasks) != 1 or len(args.answering_tasks) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning task and one answering task. {args.reasoning_tasks=} {args.answering_tasks=}")

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]

    reasoning_task = args.reasoning_tasks[0]

    reasoning_outputs = run_reasoning(args)[reasoning_model][reasoning_task]
    base_dataset = load_base_dataset_from_task(reasoning_task.replace(":", "_"))
    dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_outputs)

    results = {}
    for answering_task_spec in args.answering_tasks:
        task_base_name, _ = parse_task_spec(answering_task_spec)
        full_task_name = answering_task_spec.replace(":", "_")
        doc_to_text_module = f"lm_eval.tasks.{task_base_name}.utils"

        output = run_answering_for_dataset(
            args=args,
            answering_model= answering_model,
            answering_task_name= full_task_name,
            dataset_with_reasoning= dataset_with_reasoning,
            doc_to_text_module= doc_to_text_module
        )

        results[full_task_name] = output

    return {
        "mode": "multi-turn_CoT",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "results": results,
    }