import collections
import copy
import importlib
import argparse

from typing import Dict, List, Tuple, Any
from datasets import Dataset, DatasetDict
from lm_eval import evaluator, tasks

def majority_vote(predictions: list) -> int:
    """Aggregate predictions using majority voting and return the winning index."""
    counter = collections.Counter(predictions)
    return counter.most_common(1)[0][0]

def shorten_reasoning_chain(chain: str, edge_length: int = 100) -> str:
    """Shorten a reasoning chain for logging purposes."""
    return f'{chain[:edge_length]}  +  ..........  +  {chain[-edge_length:]}'

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
    if "test" not in task_def.dataset and "labeled" in task_def.dataset:
        return dataset_list_to_dataset_dict(list(task_def.dataset["labeled"]), split_name="labeled")
    return dataset_list_to_dataset_dict(list(task_def.dataset["test"]), split_name="test")

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

def inject_reasoning_into_dataset(base_dataset: List[dict], reasoning_samples: List, reasoning_field: str = "Reasoning_Chain") -> List[dict]:
    res = copy.deepcopy(base_dataset["test"].select(range(len(reasoning_samples))))

    # If its a list, easy to inject directly
    reasoning_texts = reasoning_samples

    if type(reasoning_samples[0]) == dict:
        reasoning_texts = [extract_reasoning_text_from_dicts(sample)[0] for sample in reasoning_samples]

    it = iter(reasoning_texts)
    return res.map(lambda x: {**x, reasoning_field: next(it)})

# -------------------------
# Task Patching
# -------------------------

def build_patched_task(answering_task_name:str, dataset_with_reasoning: List[dict], doc_to_text_func) -> Any:
    original_task = tasks.get_task_dict([answering_task_name])[answering_task_name]

    patched = copy.deepcopy(original_task)
    patched.dataset = dataset_list_to_dataset_dict(dataset_with_reasoning)
    patched.doc_to_text = doc_to_text_func

    return patched

# -------------------------
# Generate Reasoning Chains
# -------------------------

def run_reasoning(args: argparse.Namespace) -> Dict[str, Dict[str, List[dict]]]: 
    """
    Run the first step (reasoning) for multiple models and tasks.
    Returns:
      {
        model_name: {
          reasoning_task_name: [sample, sample, ...]
        },
        ...
      }
    """
    res_per_model_task = {model_name: {task : {} for task in args.reasoning_tasks} for model_name in args.reasoning_models}

    for model_name in args.reasoning_models:
        for task in args.reasoning_tasks:
            full_task_name = task.replace(":", "_")

            print(f"\n[Step 1: Reasoning] Running model: {model_name} for task {task} with args: {args}")
            results = evaluator.simple_evaluate(
                model=args.provider,
                model_args=model_name,
                tasks=[full_task_name],
                batch_size=args.batch_size,
                limit=args.limit,
                log_samples=args.log_samples,
                random_seed=args.seed
            )

            res_per_model_task[model_name][task] = results["samples"][full_task_name]
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
        doc_to_text_func_name: str = "doc_to_text_answer_selection"
    ) -> dict:
        
    doc_to_text_func = getattr(importlib.import_module(doc_to_text_module), doc_to_text_func_name)

    patched_task = build_patched_task(answering_task_name, dataset_with_reasoning, doc_to_text_func)

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