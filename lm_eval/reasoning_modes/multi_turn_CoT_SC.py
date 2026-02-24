from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.voting_modes import run_voting_modes

def mode_multi_turn_CoT_SC(args: argparse.Namespace):
    """Multi-turn CoT with Self-Consistency (SC) pipeline.

    Batched: all task pairs share a single reasoning model load and the answering
    model is loaded once per SC chain index (not once per task pair per chain).
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
    all_reasoning_outputs = run_reasoning(args)[reasoning_model]  # {task: [samples]}

    # Build per-task metadata
    task_meta = []
    for reasoning_task, answering_task_spec in zip(args.reasoning_tasks, args.answering_tasks):
        task_base_name, _ = parse_task_spec(answering_task_spec)
        full_task_name    = answering_task_spec.replace(":", "_")
        doc_to_text_module = f"lm_eval.tasks.{task_base_name}.utils"
        reasoning_outputs  = sorted(all_reasoning_outputs[reasoning_task], key=lambda s: s["doc_id"])
        chains_per_doc     = extract_multiple_reasoning_chains_per_document(reasoning_outputs)
        task_def           = tasks.get_task_dict([full_task_name])[full_task_name]
        base_dataset       = load_base_dataset_from_task(answering_task_spec.replace(":", "_"))
        task_meta.append({
            "reasoning_task":    reasoning_task,
            "answering_task":    answering_task_spec,
            "full_task_name":    full_task_name,
            "doc_to_text_module": doc_to_text_module,
            "task_def":          task_def,
            "doc_to_choice":     task_def.config.doc_to_choice,
            "base_dataset":      base_dataset,
            "chains_per_doc":    chains_per_doc,
            "predictions":       {},
        })

    # All tasks should have the same number of SC chains
    n_chains = len(task_meta[0]["chains_per_doc"])

    # --- SC loop: one answering model load per chain index across ALL tasks ---
    for chain_idx in range(n_chains):
        tasks_and_datasets = []
        modules = []
        for meta in task_meta:
            chain_list = meta["chains_per_doc"][chain_idx]
            dataset    = inject_reasoning_into_dataset(meta["base_dataset"], chain_list)
            tasks_and_datasets.append((meta["full_task_name"], dataset))
            modules.append(meta["doc_to_text_module"])

        print(f"[SC chain {chain_idx + 1}/{n_chains}] answering {len(tasks_and_datasets)} tasks in one model load")
        raw_output = run_answering_for_datasets(
            args=args,
            answering_model=answering_model,
            tasks_and_datasets=tasks_and_datasets,
            doc_to_text_module=modules,
        )

        for meta in task_meta:
            samples = raw_output["samples"][meta["full_task_name"]]
            extract_predictions_from_samples(samples, meta["doc_to_choice"], existing=meta["predictions"])
            for sample in samples:
                doc_id = sample["doc_id"]
                if "Reasoning_Chains" not in meta["predictions"][doc_id]["doc"]:
                    meta["predictions"][doc_id]["doc"]["Reasoning_Chains"] = []
                meta["predictions"][doc_id]["doc"]["Reasoning_Chains"].append(
                    sample["doc"]["Reasoning_Chain"]
                )

    # --- Voting + aggregate results per task ---
    results_list = []
    for meta in task_meta:
        aggregated_metrics = run_voting_modes(
            args,
            meta["predictions"],
            meta["doc_to_choice"],
            meta["task_def"],
            base_dataset=meta["base_dataset"],
            answering_model=answering_model,
            answering_task=meta["full_task_name"],
            doc_to_text_module=meta["doc_to_text_module"],
            self_model=reasoning_model,
        )
        
        results_list.append({
            "mode":            "multi-turn_CoT-SC",
            "reasoning_model": reasoning_model,
            "answering_model": answering_model,
            "reasoning_task":  meta["reasoning_task"],
            "answering_task":  meta["answering_task"],
            "results":         aggregated_metrics,
            "samples":         meta["predictions"],
        })

    return results_list[0] if len(results_list) == 1 else results_list