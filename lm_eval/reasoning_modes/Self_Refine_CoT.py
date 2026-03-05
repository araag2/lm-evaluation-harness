"""Self-Refine CoT reasoning pipeline.

Implements the Self-Refine loop (Madaan et al., 2023 — https://arxiv.org/abs/2303.17651):

  1. Generate an initial reasoning chain (identical to multi_turn_CoT step 1).
  2. Iteratively refine it:
       a. Feedback step — the reasoning model critiques the current chain.
       b. Refinement step — the reasoning model rewrites the chain using the feedback.
  3. Select the final answer from the best accepted reasoning chain.

Stop conditions (both can be combined):
  --refine_iterations N     at most N refinement iterations  (default: 1)
  --stop_on_degradation     halt early when the answering model's mean log-likelihood
                            decreases relative to the previous accepted iteration;
                            the best-performing chain is then used for the final answer.

Required task utility functions
--------------------------------
Each task's ``utils.py`` must expose two additional ``doc_to_text`` functions:

    def doc_to_text_self_refine_feedback(doc):
        # Critique the current ``{{Reasoning_Chain}}``.
        # Returns a generative prompt; response stored as ``Feedback``.

    def doc_to_text_self_refine_refine(doc):
        # Rewrite the reasoning chain using ``{{Reasoning_Chain}}`` + ``{{Feedback}}``.
        # Returns a generative prompt; response stored as the new ``Reasoning_Chain``.

See lm_eval/tasks/MedNLI/utils.py for the expected template pattern.
"""

import re
from typing import Optional
from lm_eval.reasoning_modes.reasoning_utils import *


def _clean_feedback(text: str) -> str:
    """Strip looping repetitions from model-generated feedback.

    Some models end the feedback correctly but then repeat ``**Final Answer: ...**``
    many times.  We keep the text up to (but not including) the first occurrence of
    that marker on its own line, then strip any trailing whitespace.
    """
    # Match a double-newline followed by '**Final Answer' (the start of the loop)
    # or a standalone line that begins with '**Final Answer'
    pattern = re.compile(r"\n{1,2}\*\*Final Answer[:\s]", re.IGNORECASE)
    m = pattern.search(text)
    if m:
        text = text[: m.start()]
    return text.rstrip()


def mode_self_refine_CoT(args: argparse.Namespace) -> Dict:
    """Self-Refine CoT pipeline entry point."""

    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(
            f"[SelfRefine][WARNING] Expected exactly one reasoning model and one answering model. "
            f"Got reasoning_models={args.reasoning_models}, answering_models={args.answering_models}."
        )

    if len(args.reasoning_tasks) != len(args.answering_tasks):
        raise ValueError(
            f"reasoning_tasks and answering_tasks must have the same length. "
            f"Got {len(args.reasoning_tasks)} vs {len(args.answering_tasks)}."
        )

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]

    n_iterations      = getattr(args, "refine_iterations", 1)
    check_degradation = getattr(args, "stop_on_degradation", False)

    # ------------------------------------------------------------------
    # Step 1: Initial reasoning — ALL tasks in ONE model load
    # ------------------------------------------------------------------
    print(f"[SelfRefine] Step 1 — Initial reasoning for {len(args.reasoning_tasks)} task(s): model={reasoning_model}")
    all_reasoning_outputs = run_reasoning(args)[reasoning_model]  # {reasoning_task: [samples]}

    # Build per-task state objects
    task_states = []
    for reasoning_task, answering_task_spec in zip(args.reasoning_tasks, args.answering_tasks):
        task_base_name, _   = parse_task_spec(answering_task_spec)
        reasoning_full_task = reasoning_task.replace(":", "_")
        answering_full_task = answering_task_spec.replace(":", "_")
        doc_to_text_module  = f"lm_eval.tasks.{task_base_name}.utils"

        reasoning_outputs = all_reasoning_outputs[reasoning_task]
        base_dataset      = load_base_dataset_from_task(reasoning_full_task)
        current_dataset   = inject_reasoning_into_dataset(base_dataset, reasoning_outputs)

        initial_chains = [extract_reasoning_text_from_dicts(s)[0] for s in reasoning_outputs]
        chain_history:    Dict[int, List[str]] = {
            s["doc_id"]: [initial_chains[i]] for i, s in enumerate(reasoning_outputs)
        }
        task_states.append({
            "reasoning_task":    reasoning_task,
            "answering_task":    answering_task_spec,
            "reasoning_full_task": reasoning_full_task,
            "answering_full_task": answering_full_task,
            "doc_to_text_module":  doc_to_text_module,
            "base_dataset":        base_dataset,
            "current_dataset":     current_dataset,
            "chain_history":       chain_history,
            "feedback_history":    {},
            "best_loglik":         float("-inf"),
            "best_dataset":        current_dataset,
            "accepted_answer_raw": None,
            "stopped_early":       False,
        })

    # ------------------------------------------------------------------
    # Step 2: Iterative Self-Refine — each sub-step batches ALL active tasks
    # ------------------------------------------------------------------
    print(
        f"[SelfRefine] Step 2 — Refinement loop: {n_iterations} iteration(s)  "
        f"stop_on_degradation={check_degradation}"
    )

    for iteration in range(n_iterations):
        iter_label    = f"{iteration + 1}/{n_iterations}"
        active_states = [s for s in task_states if not s["stopped_early"]]
        if not active_states:
            print(f"[SelfRefine] All tasks stopped early — exiting loop.")
            break

        # 2a — Feedback (batch across all active tasks)
        print(f"[SelfRefine] Iteration {iter_label} — Feedback ({len(active_states)} tasks)")
        feedback_raw = run_answering_for_datasets(
            args=args,
            answering_model=reasoning_model,
            tasks_and_datasets=[
                (s["reasoning_full_task"], s["current_dataset"]) for s in active_states
            ],
            doc_to_text_module=[s["doc_to_text_module"] for s in active_states],
            doc_to_text_func_name="doc_to_text_self_refine_feedback",
        )
        for s in active_states:
            feedback_samples = sorted(feedback_raw["samples"][s["reasoning_full_task"]], key=lambda x: x["doc_id"])
            feedback_texts   = [_clean_feedback(sample["resps"][0][0]) for sample in feedback_samples]
            for sample, text in zip(feedback_samples, feedback_texts):
                s["feedback_history"].setdefault(sample["doc_id"], []).append(text)
            s["current_dataset"] = inject_reasoning_into_dataset(
                s["current_dataset"], feedback_texts, reasoning_field="Feedback"
            )

        # 2b — Refinement (batch across all active tasks)
        print(f"[SelfRefine] Iteration {iter_label} — Refinement ({len(active_states)} tasks)")
        refined_raw = run_answering_for_datasets(
            args=args,
            answering_model=reasoning_model,
            tasks_and_datasets=[
                (s["reasoning_full_task"], s["current_dataset"]) for s in active_states
            ],
            doc_to_text_module=[s["doc_to_text_module"] for s in active_states],
            doc_to_text_func_name="doc_to_text_self_refine_refine",
        )
        for s in active_states:
            refined_samples = sorted(refined_raw["samples"][s["reasoning_full_task"]], key=lambda x: x["doc_id"])
            refined_texts   = [sample["resps"][0][0] for sample in refined_samples]
            for sample, text in zip(refined_samples, refined_texts):
                s["chain_history"].setdefault(sample["doc_id"], []).append(text)
            s["current_dataset"] = inject_reasoning_into_dataset(
                s["current_dataset"], refined_texts, reasoning_field="Reasoning_Chain"
            )

        # 2c — Optional degradation check (batch across all active tasks)
        if check_degradation:
            print(f"[SelfRefine] Iteration {iter_label} — Degradation check ({len(active_states)} tasks)")
            answer_raw = run_answering_for_datasets(
                args=args,
                answering_model=answering_model,
                tasks_and_datasets=[
                    (s["answering_full_task"], s["current_dataset"]) for s in active_states
                ],
                doc_to_text_module=[s["doc_to_text_module"] for s in active_states],
            )
            for s in active_states:
                curr_loglik = _mean_max_loglik(answer_raw["samples"][s["answering_full_task"]])
                print(
                    f"[SelfRefine][{s['answering_task']}] Iteration {iter_label}  "
                    f"loglik={curr_loglik:.4f}  best_so_far={s['best_loglik']:.4f}"
                )
                if curr_loglik < s["best_loglik"]:
                    print(
                        f"[SelfRefine][{s['answering_task']}] Degradation detected "
                        f"({s['best_loglik']:.4f} → {curr_loglik:.4f}). Reverting and stopping."
                    )
                    s["current_dataset"]  = s["best_dataset"]
                    s["stopped_early"]    = True
                else:
                    s["best_loglik"]        = curr_loglik
                    s["best_dataset"]       = s["current_dataset"]
                    # Build a minimal per-task result container for caching
                    s["accepted_answer_raw"] = {
                        "samples": {s["answering_full_task"]: answer_raw["samples"][s["answering_full_task"]]},
                        "results": {s["answering_full_task"]: answer_raw["results"][s["answering_full_task"]]},
                    }
        else:
            print(f"[SelfRefine] Iteration {iter_label} complete.")

    # ------------------------------------------------------------------
    # Step 3: Final answer selection — batch across all tasks
    # ------------------------------------------------------------------
    print(f"[SelfRefine] Step 3 — Final answer: model={answering_model}")

    # Tasks with a cached accepted_answer_raw can skip the model call
    needs_final = [s for s in task_states if s["accepted_answer_raw"] is None]
    cached      = [s for s in task_states if s["accepted_answer_raw"] is not None]

    final_results = {}  # answering_full_task → raw result dict slice

    if needs_final:
        final_batch = run_answering_for_datasets(
            args=args,
            answering_model=answering_model,
            tasks_and_datasets=[
                (s["answering_full_task"], s["current_dataset"]) for s in needs_final
            ],
            doc_to_text_module=[s["doc_to_text_module"] for s in needs_final],
        )
        for s in needs_final:
            final_results[s["answering_full_task"]] = final_batch

    for s in cached:
        final_results[s["answering_full_task"]] = s["accepted_answer_raw"]

    # Build output list
    results_list = []
    for s in task_states:
        raw = final_results[s["answering_full_task"]]
        task_def      = get_task(s["answering_full_task"])
        doc_to_choice = task_def.config.doc_to_choice

        predictions_per_input_doc = extract_predictions_from_samples(
            raw["samples"][s["answering_full_task"]], doc_to_choice
        )
        for doc_id, info in predictions_per_input_doc.items():
            info["doc"]["Reasoning_Chains_History"] = s["chain_history"].get(doc_id, [])
            info["doc"]["Feedback_History"]          = s["feedback_history"].get(doc_id, [])

        results_list.append({
            "mode":              "self-refine_CoT",
            "reasoning_model":   reasoning_model,
            "answering_model":   answering_model,
            "reasoning_task":    s["reasoning_task"],
            "answering_task":    s["answering_task"],
            "refine_iterations": n_iterations,
            "results":           format_results_dict(raw["results"][s["answering_full_task"]]),
            "samples":           predictions_per_input_doc,
        })

    return results_list[0] if len(results_list) == 1 else results_list


# -------------------------
# Private helpers
# -------------------------

def _mean_max_loglik(samples: List[dict]) -> float:
    """
    Compute the mean of each doc's best log-likelihood across answer choices.

    Used as a proxy for model confidence to detect performance degradation.
    A higher (less negative) value means the model is more confident overall.
    """
    if not samples:
        return float("-inf")
    return sum(
        max(prob[0][0] for prob in sample["resps"])
        for sample in samples
    ) / len(samples)
