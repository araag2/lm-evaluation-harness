import copy
import argparse
import json
from lm_eval import tasks
from lm_eval.reasoning_modes.reasoning_utils import *
from lm_eval.reasoning_modes.voting.voting_modes import simple_voting_modes


def mode_cross_consistency(args: argparse.Namespace) -> Dict:
    """Cross-consistency pipeline:
    1) Run each reasoning model to generate a reasoning chain for every doc.
    2) For every (origin_model, doc) chain, run every verifier model over that chain,
       producing MxM verification outputs (M reasoning models × M verifier models).
    3) For each reasoning_model, run it as the final answering model over all its
       verified datasets in a single model load (one load per unique answering model).
    4) Aggregate all MxM prediction distributions and apply majority voting.
    """

    if args.vote_file is not None:
        print(f"[CrossConsistency] Loading predictions from {args.vote_file}, skipping reasoning/answering.")

        doc_info = json.load(open(args.vote_file, "r"))
        predictions_per_input_doc = doc_info["samples"]
        answering_task = doc_info.get("answering_task", args.answering_tasks[0])
        full_task_name = answering_task.replace(":", "_")
        task_def = tasks.get_task_dict([full_task_name])[full_task_name]
        doc_to_choice = task_def.config.doc_to_choice
        flat_preds = flatten_cross_consistency_preds(predictions_per_input_doc)

        return {
            "mode": "cross_consistency",
            "reasoning_models": args.reasoning_models,
            "verifier_models": args.answering_models,
            "answering_models": args.answering_models,
            "reasoning_task": args.reasoning_tasks[0],
            "answering_task": answering_task,
            "results": simple_voting_modes(flat_preds, doc_to_choice, task_def),
            "samples": predictions_per_input_doc
        }

    if len(args.reasoning_models) < 1 or len(args.answering_models) < 1:
        print("[CrossConsistency][WARNING] At least one reasoning model and one answering model are required.")

    if len(args.reasoning_tasks) != 1 or len(args.answering_tasks) != 1:
        print(
            f"[CrossConsistency][WARNING] Expected exactly one reasoning task and one answering task; "
            f"got reasoning_tasks={args.reasoning_tasks}, answering_tasks={args.answering_tasks}."
        )

    reasoning_task = args.reasoning_tasks[0]
    reasoning_full_task_name = reasoning_task.replace(":", "_")
    answering_task = args.answering_tasks[0]
    answering_task_full_task_name = answering_task.replace(":", "_")
    reasoning_module = f"lm_eval.tasks.{parse_task_spec(reasoning_task)[0]}.utils"
    answering_module = f"lm_eval.tasks.{parse_task_spec(answering_task)[0]}.utils"

    # ------------------------------------------------------------------
    # Step 1: Generate one reasoning chain per model per doc.
    # ------------------------------------------------------------------
    print(f"[CrossConsistency] Step 1 — Reasoning: {args.reasoning_models} on {reasoning_task}")
    reasoning_outputs_per_model = run_reasoning(args)

    base_dataset = load_base_dataset_from_task(reasoning_full_task_name)
    per_model_reasoning_dataset: Dict[str, Any] = {
        model: inject_reasoning_into_dataset(base_dataset, reasoning_outputs_per_model[model][reasoning_task])
        for model in args.reasoning_models
    }

    # ------------------------------------------------------------------
    # Step 2: Verification — each verifier model runs over every
    # reasoning dataset.  Group by verifier to load each model once.
    # ------------------------------------------------------------------
    print(
        f"[CrossConsistency] Step 2 — Verification: {len(args.answering_models)} verifier(s) × "
        f"{len(args.reasoning_models)} reasoning chain(s)"
    )

    # per_model_reasoning_verification_outputs[reasoning_model][verifier_model] = Dataset
    per_model_reasoning_verification_outputs: Dict[str, Dict[str, Any]] = {
        m: {} for m in args.reasoning_models
    }

    for verifier_model in args.answering_models:
        tasks_and_datasets = [
            (reasoning_full_task_name, per_model_reasoning_dataset[reasoning_model])
            for reasoning_model in args.reasoning_models
        ]
        raw_output = run_answering_for_datasets(
            args=args,
            answering_model=verifier_model,
            tasks_and_datasets=tasks_and_datasets,
            doc_to_text_module=reasoning_module,
            doc_to_text_func_name="doc_to_text_verify_reasoning",
        )
        for reasoning_model in args.reasoning_models:
            verification_samples = raw_output["samples"][reasoning_full_task_name]
            per_model_reasoning_verification_outputs[reasoning_model][verifier_model] = inject_reasoning_into_dataset(
                per_model_reasoning_dataset[reasoning_model],
                verification_samples,
                reasoning_field="Verified_Reasoning_Chain",
            )

    # Resolve label list once — needed in both Step 3 and Step 4.
    task_def      = tasks.get_task_dict([answering_task_full_task_name])[answering_task_full_task_name]
    doc_to_choice = task_def.config.doc_to_choice

    # ------------------------------------------------------------------
    # Step 3: Final answer selection — each reasoning model reads its own
    # verified chains.  Group by answering (reasoning) model to load once.
    # ------------------------------------------------------------------
    print(
        f"[CrossConsistency] Step 3 — Answer selection: {len(args.reasoning_models)} reasoning model(s) × "
        f"{len(args.answering_models)} verifier dataset(s)"
    )

    predictions_per_input_doc: Dict[int, Dict] = {}

    for reasoning_model in args.reasoning_models:
        tasks_and_datasets = [
            (answering_task_full_task_name, per_model_reasoning_verification_outputs[reasoning_model][verifier_model])
            for verifier_model in args.answering_models
        ]
        raw_output = run_answering_for_datasets(
            args=args,
            answering_model=reasoning_model,
            tasks_and_datasets=tasks_and_datasets,
            doc_to_text_module=answering_module,
            doc_to_text_func_name="doc_to_text_answer_selection_after_verify_reasoning",
        )

        for sample in raw_output["samples"][answering_task_full_task_name]:
            doc_id = sample["doc_id"]

            if doc_id not in predictions_per_input_doc:
                predictions_per_input_doc[doc_id] = {
                    "doc": copy.deepcopy(sample["doc"]),
                    "preds": {},
                }
                predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"] = []
                predictions_per_input_doc[doc_id]["doc"]["Verified_Reasoning_Chains"] = []

            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(
                sample["doc"]["Reasoning_Chain"]
            )
            predictions_per_input_doc[doc_id]["doc"]["Verified_Reasoning_Chains"].append(
                sample["doc"]["Verified_Reasoning_Chain"]
            )

            pred_probs = [prob[0][0] for prob in sample["resps"]]
            pred_idx   = pred_probs.index(max(pred_probs))
            pred_label = doc_to_choice[pred_idx]
            predictions_per_input_doc[doc_id]["preds"].setdefault(reasoning_model, []).append(
                (pred_idx, pred_probs, pred_label)
            )

    # ------------------------------------------------------------------
    # Step 4: Aggregate metrics across all MxM predictions.
    # ------------------------------------------------------------------
    print(f"[CrossConsistency] Step 4 — Voting: aggregating {len(args.reasoning_models)}×{len(args.answering_models)} predictions for {answering_task}")

    flat_preds = flatten_cross_consistency_preds(predictions_per_input_doc)

    return {
        "mode": "cross_consistency",
        "reasoning_models": args.reasoning_models,
        "verifier_models": args.answering_models,
        "answering_models": args.answering_models,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        "results": simple_voting_modes(flat_preds, doc_to_choice, task_def),
        "samples": predictions_per_input_doc
    }


# -------------------------
# Cross-consistency helpers
# -------------------------

def flatten_cross_consistency_preds(predictions_per_input_doc: Dict) -> Dict:
    """
    Convert cross-consistency predictions_per_input_doc (where preds is a
    dict keyed by reasoning_model, each value a list of (pred_idx, pred_probs))
    into the standard flat format expected by simple_voting_modes:
      doc["preds"]      -> flat list of pred indices
      doc["pred_probs"] -> flat list of prob lists
    Returns a new dict; does not mutate the input.
    """
    flat = {}
    for doc_id, info in predictions_per_input_doc.items():
        flat_preds  = []
        flat_probs  = []
        flat_labels = []
        for model_preds in info["preds"].values():
            for entry in model_preds:
                pred_idx, pred_probs = entry[0], entry[1]
                flat_preds.append(pred_idx)
                flat_probs.append(pred_probs)
                flat_labels.append(entry[2] if len(entry) > 2 else None)
        flat[doc_id] = {
            **info,
            "preds":       flat_preds,
            "pred_probs":  flat_probs,
            "pred_labels": flat_labels,
        }
    return flat