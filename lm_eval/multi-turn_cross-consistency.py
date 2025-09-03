import copy
import json
import argparse
import importlib

from datetime import datetime
from typing import Dict, List, Tuple, Any, Union
from lm_eval import evaluator, tasks
from datasets import Dataset, DatasetDict

def dataset_list_to_dataset_dict(dataset_list: list[dict]) -> DatasetDict:
    """Convert a list of dictionaries into a DatasetDict."""
    if type(dataset_list) is DatasetDict:
        return dataset_list
    return DatasetDict({"test": Dataset.from_list(dataset_list)})

def parse_task_spec(task_spec: str) -> Tuple[str, str]:
    """Parse a task specification string into a tuple of (task_name, sub_task_name)."""
    task_parts = task_spec.split(":")
    if len(task_parts) == 2:
        return task_parts[0], task_parts[1]
    else:
        raise ValueError(f"Invalid task spec: {task_spec}")
    
def load_base_dataset_from_task(task_name: str) -> List[str]:
    """Load the base dataset for a given task."""
    task_def = tasks.get_task_dict([task_name])[task_name]
    return dataset_list_to_dataset_dict(list(task_def.dataset["test"]))

def extract_reasoning_text(sample:dict) -> str:
    if "resps" in sample and sample["resps"]:
        if isinstance(sample["resps"], list) and sample["resps"]:
            return sample["resps"][0][0]

    raise ValueError(f"Could not extract reasoning text from sample: {sample}")

def inject_reasoning_into_dataset(base_dataset: List[dict], reasoning_samples: List[dict], reasoning_field: str = "Reasoning_Chain") -> List[dict]:
    res = copy.deepcopy(base_dataset["test"].select(range(len(reasoning_samples))))
    reasoning_texts = [extract_reasoning_text(sample) for sample in reasoning_samples]
    it = iter(reasoning_texts)

    return res.map(lambda x: {**x, reasoning_field: next(it)})


def build_patched_task(answering_task_name:str, dataset_with_reasoning: List[dict], doc_to_text_func) -> Any:
    original_task = tasks.get_task_dict([answering_task_name])[answering_task_name]

    patched = copy.deepcopy(original_task)
    print(dataset_with_reasoning)
    print(dataset_with_reasoning[0].keys())
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
    #print(res_per_model_task)
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

# -------------------------
# Modes
# -------------------------

def mode_multi_turn(args: argparse.Namespace) -> None:
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
        "mode": "multi-turn",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "results": results,
    }

def mode_cross_consistency(args: argparse.Namespace) -> None:
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



    



def main():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument('--provider', type=str, default="vllm")
    parser.add_argument('--reasoning_models', nargs='+', default=['pretrained=Qwen/Qwen3-0.6B'])
    parser.add_argument('--answering_models', nargs='+', default=['pretrained=Qwen/Qwen3-0.6B'])

    # Gen kwargs
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="limit #eval docs for quick tests")
    parser.add_argument('--write_out', action='store_true')
    parser.add_argument('--log_samples', action='store_true')

    # Tasks to Run
    parser.add_argument('--reasoning_tasks', nargs='+', default=['MedNLI_CoT'])
    parser.add_argument('--answering_tasks', nargs='+', default=['MedNLI:0-shot'])

    # Modes
    parser.add_argument("--mode", type=str, default="multi-turn",
                        choices=["multi-turn", "cross-consistency"])

    # Output Args
    parser.add_argument('--output_path', type=str, default="/cfs/home/u021010/PhD/active_dev/outputs/CoT-Debug/")
    args = parser.parse_args()

    reasoning_chain_outputs = run_reasoning(args)

    if args.mode == "multi-turn":
        out = mode_multi_turn(args)
    else:
        out = mode_cross_consistency(args)

    print("\n==== RESULTS (summary keys only) ====")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))

    if args.output_path:
        with open(f"{args.output_path}{datetime.now().strftime('%Y-%m-%dT%H-%M')}.json", "w") as f:
            json.dump(out, f, indent=4)
        print(f"\n✅ Results written to {args.output_path}")

if __name__ == "__main__":
    main()