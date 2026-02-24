import json

from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.voting_modes import run_voting_modes


def mode_only_vote(args: argparse.Namespace) -> Dict:
    if args.vote_file is None:
        raise ValueError(
            "Vote-only mode requires a samples JSON file — pass it with --vote_file."
        )

    doc_info = json.load(open(args.vote_file, "r"))
    answering_model               = doc_info.get("answering_model", "unknown")
    answering_task_with_delimiter = doc_info.get("answering_task",  "unknown")
    answering_task                = answering_task_with_delimiter.replace(":", "_")

    predictions_per_input_doc = doc_info.get("samples", {})
    if args.limit is not None:
        predictions_per_input_doc = dict(
            list(predictions_per_input_doc.items())[:args.limit]
        )
    print(
        f"[only-vote] task={answering_task}  model={answering_model}  "
        f"docs={len(predictions_per_input_doc)}"
    )

    task_def      = tasks.get_task_dict([answering_task])[answering_task]
    doc_to_choice = task_def.config.doc_to_choice

    doc_to_text_module = (
        f"lm_eval.tasks.{parse_task_spec(answering_task_with_delimiter)[0]}.utils"
    )
    base_dataset = load_base_dataset_from_task(answering_task)

    aggregated_metrics = run_voting_modes(
        args,
        predictions_per_input_doc,
        doc_to_choice,
        task_def,
        base_dataset=base_dataset,
        answering_model=answering_model,
        answering_task=answering_task,
        doc_to_text_module=doc_to_text_module,
    )

    return {
        "mode":            doc_info.get("mode",            "unknown"),
        "reasoning_model": doc_info.get("reasoning_model", "unknown"),
        "answering_model": answering_model,
        "reasoning_task":  doc_info.get("reasoning_task",  "unknown"),
        "answering_task":  answering_task,
        "results":         aggregated_metrics,
        "samples":         predictions_per_input_doc,
    }