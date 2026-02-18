import collections
import copy
import importlib
import argparse

from typing import Dict, List, Tuple, Any
from datasets import Dataset, DatasetDict
from lm_eval import evaluator, tasks


def shorten_reasoning_chain(chain: str, edge_length: int = 100) -> str:
    """Shorten a reasoning chain for logging purposes."""
    return f'{chain[:edge_length]}  +  ..........  +  {chain[-edge_length:]}'

def format_results_dict(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format results dictionary by rounding floats and converting lists to strings."""
    formatted = {}
    for key, value in results.items():
        if key in ("alias", "acc_stderr,none", "acc_norm_stderr,none", "f1_stderr,none", "f1_norm_stderr,none"):
            continue
        elif key in ("acc,none", "acc_norm,none", "f1,none", "f1_norm,none"):
            formatted[key.split(",")[0]] = round(value, 4) if isinstance(value, float) else 0.0
        else:
            formatted[key] = round(value, 4) if isinstance(value, float) else 0.0
    return formatted

# -------------------------
# Dataset Helpers
# -------------------------

def dataset_list_to_dataset_dict(dataset_list: list[dict], split_name: str = "test") -> DatasetDict:
    """Convert a list of dictionaries into a DatasetDict."""
    if isinstance(dataset_list, DatasetDict):
        return dataset_list
    return DatasetDict({split_name: Dataset.from_list(dataset_list)})


def parse_task_spec(task_spec: str) -> Tuple[str, str]:
    """Parse a task specification string into a tuple of (task_name, sub_task_name)."""
    task_parts = task_spec.split(":")
    if len(task_parts) == 2:
        return task_parts[0], task_parts[1]
    else:
        raise ValueError(f"Invalid task spec: {task_spec}")

def load_base_dataset_from_task(task_name: str) -> DatasetDict:
    """Load the base dataset for a given task name."""
    task_def = tasks.get_task_dict([task_name])[task_name]
    return dataset_list_to_dataset_dict(list(task_def.dataset[task_def.config.test_split]), split_name="test")

# -------------------------
# Reasoning Chain Helpers
# -------------------------

def extract_multiple_reasoning_chains_per_document(reasoning_outputs: List[dict]) -> Dict[int, List[str]]:
    n_chains = len(reasoning_outputs[0]["resps"][0])
    reasoning_chains_per_document = [[] for _ in range(n_chains)]
    for output in reasoning_outputs:
        for i in range(n_chains):
            reasoning_chains_per_document[i].append(output["resps"][0][i])
    return reasoning_chains_per_document

def extract_reasoning_text_from_dicts(sample: dict) -> List[str]:
    """Extract reasoning text from a sample produced during reasoning step."""
    if "resps" in sample and sample["resps"] and isinstance(sample["resps"], list):
        return [resp[0] for resp in sample["resps"]]
    raise ValueError(f"Could not extract reasoning text from sample: {sample}")

def inject_reasoning_into_dataset(
    base_dataset: List[dict],
    reasoning_samples: List,
    reasoning_field: str = "Reasoning_Chain",
) -> Dataset:
    """
    Inject reasoning texts into a copy of base_dataset.

    HuggingFace Dataset objects are immutable/copy-on-write, so `.select()` is
    sufficient — no deepcopy needed (which would be very slow on large datasets).
    with_indices=True guarantees position-based alignment between the dataset
    rows and the reasoning_samples list.
    """
    test_set = dataset_list_to_dataset_dict(base_dataset, "test")["test"]
    aligned = test_set.select(range(len(reasoning_samples)))

    reasoning_texts = reasoning_samples
    if reasoning_texts and isinstance(reasoning_texts[0], dict):
        reasoning_texts = [extract_reasoning_text_from_dicts(sample)[0] for sample in reasoning_texts]

    return aligned.map(
        lambda x, i: {**x, reasoning_field: reasoning_texts[i]},
        with_indices=True,
    )

# -------------------------
# Task Patching
# -------------------------

def build_patched_task(answering_task_name:str, dataset_with_reasoning: List[dict], doc_to_text_func) -> Any:
    original_task = tasks.get_task_dict([answering_task_name])[answering_task_name]

    patched = copy.deepcopy(original_task)
    patched.dataset = dataset_list_to_dataset_dict(dataset_with_reasoning)
    patched.doc_to_text = doc_to_text_func
    patched.config.test_split = "test"

    return patched

# -------------------------
# Generate Reasoning Chains
# -------------------------

def run_reasoning(args: argparse.Namespace) -> Dict[str, Dict[str, List[dict]]]:
    """
    Run the reasoning step for every (model, task) pair.

    Returns a nested dict::

        {
            model_name: {
                reasoning_task_name: [sample, ...]
            }
        }
    """
    res_per_model_task = {
        model_name: {task: {} for task in args.reasoning_tasks}
        for model_name in args.reasoning_models
    }

    n_models = len(args.reasoning_models)
    n_tasks  = len(args.reasoning_tasks)
    total    = n_models * n_tasks
    done     = 0

    for model_name in args.reasoning_models:
        for task in args.reasoning_tasks:
            done += 1
            full_task_name = task.replace(":", "_")
            print(
                f"[Reasoning {done}/{total}] model={model_name}  task={full_task_name}"
                + (f"  limit={args.limit}" if args.limit is not None else "")
            )

            results = evaluator.simple_evaluate(
                model=args.provider,
                model_args=model_name,
                tasks=[full_task_name],
                batch_size=args.batch_size,
                limit=args.limit,
                log_samples=args.log_samples,
                random_seed=args.seed,
            )

            res_per_model_task[model_name][task] = results["samples"][full_task_name]
            print(f"[Reasoning {done}/{total}] done — {len(res_per_model_task[model_name][task])} samples collected.")

    return res_per_model_task

# -------------------------------------
# Answer queries with reasoning chain
# -------------------------------------

def run_answering_for_dataset(
    args: argparse.Namespace,
    answering_model: str,
    answering_task_name: str,
    dataset_with_reasoning: List[dict],
    doc_to_text_module: str,
    doc_to_text_func_name: str = "doc_to_text_answer_selection",
) -> dict:
    """
    Patch the task with the reasoning-augmented dataset and run answering evaluation.

    The doc_to_text function is loaded once via importlib; the patched task object is
    local to this call so concurrent callers (if any) won't interfere.
    """
    print(
        f"[Answering] model={answering_model}  task={answering_task_name}  "
        f"prompt_fn={doc_to_text_func_name}  dataset_size={len(dataset_with_reasoning)}"
        + (f"  limit={args.limit}" if args.limit is not None else "")
    )

    doc_to_text_func = getattr(importlib.import_module(doc_to_text_module), doc_to_text_func_name)
    patched_task     = build_patched_task(answering_task_name, dataset_with_reasoning, doc_to_text_func)

    results = evaluator.simple_evaluate(
        model=args.provider,
        model_args=answering_model,
        tasks=[patched_task],
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=args.log_samples,
        random_seed=args.seed,
    )

    return results

# -------------------------
# Prediction Extraction
# -------------------------

def extract_predictions_from_samples(
    samples: List[dict],
    doc_to_choice: List[str],
    existing: Dict = None
) -> Dict:
    """
    Convert raw lm-eval samples into the canonical predictions_per_input_doc format.

    If `existing` is None (single-chain modes): returns a fresh dict with scalar
    `preds` (int), flat `pred_probs` (list of floats), and `pred_label` (str).

    If `existing` is provided (SC / multi-chain accumulation): appends each
    chain's pred/pred_probs as lists into the existing dict. The caller is
    responsible for any mode-specific extra fields (e.g. Reasoning_Chains).
    """
    accumulate = existing is not None
    result = existing if accumulate else {}

    for sample in samples:
        doc_id = sample["doc_id"]
        pred_probs = [prob[0][0] for prob in sample["resps"]]
        pred_idx = pred_probs.index(max(pred_probs))

        if doc_id not in result:
            result[doc_id] = {
                "doc": copy.deepcopy(sample["doc"]),
                "preds": [] if accumulate else None,
                "pred_probs": [] if accumulate else None,
            }

        if accumulate:
            result[doc_id]["pred_probs"].append(pred_probs)
            result[doc_id]["preds"].append(pred_idx)
        else:
            result[doc_id]["pred_probs"] = pred_probs
            result[doc_id]["preds"] = pred_idx
            result[doc_id]["pred_label"] = doc_to_choice[pred_idx]

    return result