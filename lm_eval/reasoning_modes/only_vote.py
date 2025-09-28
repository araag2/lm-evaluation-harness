import json
import os

from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.simple_voting_modes import simple_voting_modes
from lm_eval.reasoning_modes.voting.mbr_voting_modes import mbr_voting_modes

def mode_only_vote(args: argparse.Namespace, simple_voting : bool = True, mbr_voting : bool = True) -> Dict:
    if args.vote_file is None:
        raise ValueError("For Vote-only mode, please provide a vote_file with --vote_file")
    
    doc_info = json.load(open(args.vote_file, "r"))
    answering_model = doc_info.get("answering_model", "unknown")
    answering_task_with_delimiter = doc_info.get("answering_task", "unknown")
    answering_task = answering_task_with_delimiter.replace(":", "_")

    predictions_per_input_doc = doc_info.get("samples", {})
    if args.limit is not None:
        predictions_per_input_doc = dict(list(predictions_per_input_doc.items())[:args.limit])


    task_def = tasks.get_task_dict([answering_task])[answering_task]
    doc_to_choice = task_def.config.doc_to_choice

    doc_to_text_module = f"lm_eval.tasks.{parse_task_spec(answering_task_with_delimiter)[0]}.utils"
    base_dataset = load_base_dataset_from_task(answering_task)

    aggregated_metrics = {}

    if simple_voting:
        aggregated_metrics_simple = simple_voting_modes(
            predictions_per_input_doc=predictions_per_input_doc,
            doc_to_choice=doc_to_choice,
            task_def=task_def,
        )

        print(aggregated_metrics_simple)

    if mbr_voting:
        aggregated_metrics_mbr = mbr_voting_modes(
            args=args,
            predictions_per_input_doc=predictions_per_input_doc,
            base_dataset=base_dataset,
            answering_model=answering_model,
            answering_task=answering_task,
            doc_to_text_module=doc_to_text_module,
            doc_to_choice=doc_to_choice,
            task_def=task_def
        )

    aggregated_metrics.update(aggregated_metrics_simple)
    aggregated_metrics.update(aggregated_metrics_mbr["results"])

    args.output_path = os.path.join(args.output_path, doc_info.get("reasoning_task", "unknown").replace(":", "_"), f"{answering_model.split(',')[0]}")

    return {
        "mode": doc_info.get("mode", "unknown"),
        "reasoning_model": doc_info.get("reasoning_model", "unknown"),
        "answering_model": answering_model,
        "reasoning_task": doc_info.get("reasoning_task", "unknown"),
        "answering_task": answering_task,
        "results": aggregated_metrics,
        "samples" : predictions_per_input_doc
    }