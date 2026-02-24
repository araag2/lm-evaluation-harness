import collections
import copy
import importlib
import argparse
import re

from typing import Dict, List, Optional, Tuple, Any
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

def dataset_list_to_dataset_dict(dataset_list, split_name: str = "test") -> DatasetDict:
    """Convert a list of dicts, a Dataset, or a DatasetDict into a DatasetDict."""
    if isinstance(dataset_list, DatasetDict):
        return dataset_list
    if isinstance(dataset_list, Dataset):
        return DatasetDict({split_name: dataset_list})
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

# Trailing answer lines that any model might append at the end of the chain.
# Strips "Answer: ...", "Final Answer: ...", "Verified Answer: ..." regardless
# of what follows the colon, so this works across all tasks and label sets.
_ANSWER_TAIL_RE = re.compile(
    r"\n?(?:Answer|Final\s+Answer|Verified\s+Answer)\s*:[^\n]*\s*$",
    re.IGNORECASE,
)
# Junk tokens/fragments to remove from anywhere in the chain.
# Each pattern is stripped via re.sub rather than used to discard the whole chain.
_JUNK_STRIP_RE = re.compile(
    r"<\|[^|]+\|>"          # special tokens: <|im_end|>, <|END|>, etc.
    r"|```(?:\w*)?"          # code-fence markers (``` or ```python etc.)
    r"|Do not write anything[^\n]*"  # prompt-echo fragment
    ,
    re.IGNORECASE,
)
_MIN_CHAIN_LENGTH = 30

def clean_reasoning_chain(chain: str) -> str:
    """Clean a raw reasoning chain before injection into the answering dataset.

    Handles the following failure modes:

    1. **No reasoning generated**: the model output the answer directly or echoed
       part of the prompt.  Returns ``""``.

    2. **Trailing answer stubs**: strips lines like ``Answer: X`` or
       ``Final Answer: X`` that some models append after the reasoning, regardless
       of the label vocabulary used by the task.

    3. **Inline junk tokens**: removes special tokens (``<|...|>``), code-fence
       markers (`` ``` ``), and prompt-echo fragments (``Do not write anything...``)
       from anywhere in the chain rather than discarding the whole chain.

    4. **Effectively empty after cleaning**: if fewer than 30 characters remain
       after all stripping, returns ``""``.
    """
    chain = chain.strip()
    if not chain:
        return ""

    # --- Strip trailing answer lines ---
    chain = _ANSWER_TAIL_RE.sub("", chain).strip()

    # --- Strip junk tokens/fragments throughout the chain ---
    chain = _JUNK_STRIP_RE.sub("", chain).strip()

    # After all cleaning, treat very short results as empty (model produced no reasoning).
    if len(chain) < _MIN_CHAIN_LENGTH:
        return ""

    return chain

def inject_reasoning_into_dataset(
    base_dataset: List[dict],
    reasoning_samples: List,
    reasoning_field: str = "Reasoning_Chain",
) -> Dataset:
    """
    Inject reasoning texts into a copy of base_dataset.

    When ``reasoning_samples`` contains lm-eval sample dicts (each with a
    ``doc_id`` field), the list is sorted by ``doc_id`` before injection so
    that position *i* in the sorted list always corresponds to dataset row *i*.
    This is necessary because ``evaluator.simple_evaluate`` does not guarantee
    that samples are returned in the same order as the input dataset.

    When the samples are raw lm-eval dicts (with ``arguments``), the prompt
    that produced each generation is also injected as ``{reasoning_field}_Prompt``
    so it can be inspected in the output JSON.
    """
    test_set = dataset_list_to_dataset_dict(base_dataset, "test")["test"]

    reasoning_texts = reasoning_samples
    prompts: Optional[List[str]] = None

    if reasoning_texts and isinstance(reasoning_texts[0], dict):
        # Sort by doc_id to align with dataset row order (0, 1, 2, ...)
        reasoning_texts = sorted(reasoning_texts, key=lambda s: s["doc_id"])
        # Capture prompts from arguments[0][0] when available
        prompts = [
            s["arguments"][0][0]
            if "arguments" in s and s["arguments"]
            else ""
            for s in reasoning_texts
        ]
        reasoning_texts = [extract_reasoning_text_from_dicts(sample)[0] for sample in reasoning_texts]

    # Clean every chain: extract think-block content, strip answer echoes, drop junk.
    reasoning_texts = [clean_reasoning_chain(t) for t in reasoning_texts]

    aligned = test_set.select(range(len(reasoning_texts)))

    if prompts is not None:
        return aligned.map(
            lambda x, i: {**x, reasoning_field: reasoning_texts[i], f"{reasoning_field}_Prompt": prompts[i]},
            with_indices=True,
        )
    return aligned.map(
        lambda x, i: {**x, reasoning_field: reasoning_texts[i]},
        with_indices=True,
    )

# -------------------------
# Task Patching
# -------------------------

def build_patched_task(
    answering_task_name: str,
    dataset_with_reasoning: List[dict],
    doc_to_text_func,
    alias: Optional[str] = None,
) -> Any:
    """Build a patched lm-eval task with a replaced dataset and doc_to_text.

    ``alias`` renames the task in the evaluator results dict.  Pass a unique
    string when multiple patched tasks share the same base task name so that
    ``simple_evaluate`` does not overwrite earlier results with later ones.
    """
    original_task = tasks.get_task_dict([answering_task_name])[answering_task_name]

    patched = copy.deepcopy(original_task)
    patched.dataset = dataset_list_to_dataset_dict(dataset_with_reasoning)
    patched.doc_to_text = doc_to_text_func
    patched.config.test_split = "test"
    if alias is not None:
        patched.config.task = alias

    return patched

# -------------------------
# Generate Reasoning Chains
# -------------------------

def run_reasoning(args: argparse.Namespace) -> Dict[str, Dict[str, List[dict]]]:
    """
    Run the reasoning step for every (model, task) pair.

    All reasoning tasks for a given model are batched into a SINGLE
    ``simple_evaluate`` call so the model is loaded exactly once per model,
    regardless of how many reasoning tasks are requested.

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

    for model_idx, model_name in enumerate(args.reasoning_models, 1):
        all_full_task_names = [t.replace(":", "_") for t in args.reasoning_tasks]
        print(
            f"[Reasoning model {model_idx}/{n_models}] model={model_name}  "
            f"tasks={all_full_task_names}  (batched: {n_tasks} tasks, 1 model load)"
            + (f"  limit={args.limit}" if args.limit is not None else "")
        )

        results = evaluator.simple_evaluate(
            model=args.provider,
            model_args=model_name,
            tasks=all_full_task_names,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=True,      # always needed to extract samples
            random_seed=args.seed,
        )

        for task, full_task_name in zip(args.reasoning_tasks, all_full_task_names):
            res_per_model_task[model_name][task] = results["samples"][full_task_name]
            print(
                f"[Reasoning] done — task={full_task_name}  "
                f"{len(res_per_model_task[model_name][task])} samples collected."
            )

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
        log_samples=True,   # always needed — chain injection and prediction extraction require samples
        random_seed=args.seed,
    )

    return results


def run_answering_for_datasets(
    args: argparse.Namespace,
    answering_model: str,
    tasks_and_datasets: List[Tuple[str, Any]],
    doc_to_text_module,  # str (shared) or List[str] (one per task)
    doc_to_text_func_name: str = "doc_to_text_answer_selection",
    task_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run answering for *multiple* datasets under a single model load.

    ``tasks_and_datasets`` is a list of ``(task_name, dataset_with_reasoning)`` pairs.
    ``doc_to_text_module`` can be a single module path (string, shared by all tasks)
    or a list of per-task module paths.
    ``task_aliases`` is an optional list of unique name overrides for each task;
    use this when multiple tasks share the same base task name so that
    ``simple_evaluate`` does not collapse their results into a single key.

    All patched tasks are evaluated in one ``simple_evaluate`` call so the model is
    loaded only once, regardless of how many datasets are passed.

    Returns the raw ``simple_evaluate`` result dict whose ``"samples"`` key is keyed
    by task name (or alias if provided).
    """
    modules = (
        doc_to_text_module
        if isinstance(doc_to_text_module, list)
        else [doc_to_text_module] * len(tasks_and_datasets)
    )
    aliases = task_aliases if task_aliases is not None else [None] * len(tasks_and_datasets)

    patched_tasks = []
    for (task_name, dataset_with_reasoning), module, alias in zip(tasks_and_datasets, modules, aliases):
        doc_to_text_func = getattr(importlib.import_module(module), doc_to_text_func_name)
        patched_tasks.append(build_patched_task(task_name, dataset_with_reasoning, doc_to_text_func, alias=alias))

    print(
        f"[Answering] model={answering_model}  tasks={[t for t, _ in tasks_and_datasets]}  "
        f"prompt_fn={doc_to_text_func_name}  (batched: {len(tasks_and_datasets)} tasks, 1 model load)"
        + (f"  limit={args.limit}" if args.limit is not None else "")
    )

    return evaluator.simple_evaluate(
        model=args.provider,
        model_args=answering_model,
        tasks=patched_tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=True,   # always needed — chain injection and prediction extraction require samples
        random_seed=args.seed,
    )


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

        # Extract the full answering prompt from arguments[0][0] when available.
        # For loglikelihood requests, args = (context, continuation) — args[0] is the prompt.
        answering_prompt: Optional[str] = None
        if "arguments" in sample and sample["arguments"]:
            try:
                answering_prompt = sample["arguments"][0][0]
            except (IndexError, TypeError):
                pass

        if doc_id not in result:
            result[doc_id] = {
                "doc": copy.deepcopy(sample["doc"]),
                "preds": [] if accumulate else None,
                "pred_probs": [] if accumulate else None,
            }
            if answering_prompt is not None:
                result[doc_id]["answering_prompt"] = answering_prompt

        if accumulate:
            result[doc_id]["pred_probs"].append(pred_probs)
            result[doc_id]["preds"].append(pred_idx)
        else:
            result[doc_id]["pred_probs"] = pred_probs
            result[doc_id]["preds"] = pred_idx
            result[doc_id]["pred_label"] = doc_to_choice[pred_idx]

    return result