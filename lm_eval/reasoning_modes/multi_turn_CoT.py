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

    task_names = list(results.keys())
    doc_to_choice = tasks.get_task_dict([full_task_name])[full_task_name].config.doc_to_choice

    output_results = None

    if len(task_names) == 1:
        results = results[task_names[0]]

        output_results = {"results" : format_results_dict(results["results"][task_names[0]]), "samples": {}}
        predictions_per_input_doc = {}

        for sample in results["samples"][task_names[0]]:
            doc_id = sample["doc_id"]

            predictions_per_input_doc[doc_id] = {
                "doc" : sample["doc"],
                "preds": [],
                "pred_probs": [],
            }

            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chain"] = shorten_reasoning_chain(sample["doc"]["Reasoning_Chain"], 100)
            predictions_per_input_doc[doc_id]["pred_probs"] = [prob[0][0] for prob in sample["resps"]]
            predictions_per_input_doc[doc_id]["preds"] = predictions_per_input_doc[doc_id]["pred_probs"].index(max(predictions_per_input_doc[doc_id]["pred_probs"]))

            predictions_per_input_doc[doc_id]["pred_label"] = doc_to_choice[predictions_per_input_doc[doc_id]["preds"]]
            
        
        output_results["samples"] = predictions_per_input_doc
        
    return {
        "mode": "multi-turn_CoT",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_tasks": answering_task_spec,
        **output_results
    }