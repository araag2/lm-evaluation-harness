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

    if len(args.reasoning_tasks) != 1 or len(args.answering_tasks) != 1:
        print(
            f"[SelfRefine][WARNING] Expected exactly one reasoning task and one answering task. "
            f"Got reasoning_tasks={args.reasoning_tasks}, answering_tasks={args.answering_tasks}."
        )

    reasoning_model     = args.reasoning_models[0]
    answering_model     = args.answering_models[0]
    reasoning_task      = args.reasoning_tasks[0]
    answering_task_spec = args.answering_tasks[0]

    task_base_name, _      = parse_task_spec(answering_task_spec)
    reasoning_full_task    = reasoning_task.replace(":", "_")
    answering_full_task    = answering_task_spec.replace(":", "_")
    doc_to_text_module     = f"lm_eval.tasks.{task_base_name}.utils"

    # ------------------------------------------------------------------
    # Step 1: Initial reasoning chain
    # ------------------------------------------------------------------
    print(f"[SelfRefine] Step 1 — Initial reasoning: model={reasoning_model}  task={reasoning_task}")
    reasoning_outputs = run_reasoning(args)[reasoning_model][reasoning_task]

    base_dataset     = load_base_dataset_from_task(reasoning_full_task)
    current_dataset  = inject_reasoning_into_dataset(base_dataset, reasoning_outputs)

    # Seed chain history from initial outputs (one entry per doc).
    initial_chains = [extract_reasoning_text_from_dicts(s)[0] for s in reasoning_outputs]
    chain_history:    Dict[int, List[str]] = {
        s["doc_id"]: [initial_chains[i]] for i, s in enumerate(reasoning_outputs)
    }
    feedback_history: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    # Step 2: Iterative Self-Refine loop
    # ------------------------------------------------------------------
    n_iterations   = getattr(args, "refine_iterations", 1)
    check_degradation = getattr(args, "stop_on_degradation", False)

    best_dataset      = current_dataset
    best_loglik       = float("-inf")
    accepted_answer_raw: Optional[dict] = None   # cached last non-degraded answer eval

    print(
        f"[SelfRefine] Step 2 — Refinement loop: {n_iterations} iteration(s)  "
        f"stop_on_degradation={check_degradation}"
    )

    for iteration in range(n_iterations):
        iter_label = f"{iteration + 1}/{n_iterations}"

        # 2a — Feedback: critique the current Reasoning_Chain
        print(f"[SelfRefine] Iteration {iter_label} — Feedback")
        feedback_raw     = run_answering_for_dataset(
            args                  = args,
            answering_model       = reasoning_model,
            answering_task_name   = reasoning_full_task,
            dataset_with_reasoning= current_dataset,
            doc_to_text_module    = doc_to_text_module,
            doc_to_text_func_name = "doc_to_text_self_refine_feedback",
        )
        feedback_samples = feedback_raw["samples"][reasoning_full_task]
        feedback_texts   = [_clean_feedback(s["resps"][0][0]) for s in feedback_samples]

        for sample, text in zip(feedback_samples, feedback_texts):
            feedback_history.setdefault(sample["doc_id"], []).append(text)

        current_dataset = inject_reasoning_into_dataset(
            current_dataset, feedback_texts, reasoning_field="Feedback"
        )

        # 2b — Refinement: rewrite Reasoning_Chain given Feedback
        print(f"[SelfRefine] Iteration {iter_label} — Refinement")
        refined_raw      = run_answering_for_dataset(
            args                  = args,
            answering_model       = reasoning_model,
            answering_task_name   = reasoning_full_task,
            dataset_with_reasoning= current_dataset,
            doc_to_text_module    = doc_to_text_module,
            doc_to_text_func_name = "doc_to_text_self_refine_refine",
        )
        refined_samples = refined_raw["samples"][reasoning_full_task]
        refined_texts   = [s["resps"][0][0] for s in refined_samples]

        for sample, text in zip(refined_samples, refined_texts):
            chain_history.setdefault(sample["doc_id"], []).append(text)

        current_dataset = inject_reasoning_into_dataset(
            current_dataset, refined_texts, reasoning_field="Reasoning_Chain"
        )

        # 2c — Optional degradation check via answering log-likelihood
        if check_degradation:
            answer_raw  = run_answering_for_dataset(
                args                  = args,
                answering_model       = answering_model,
                answering_task_name   = answering_full_task,
                dataset_with_reasoning= current_dataset,
                doc_to_text_module    = doc_to_text_module,
            )
            curr_loglik = _mean_max_loglik(answer_raw["samples"][answering_full_task])
            print(
                f"[SelfRefine] Iteration {iter_label}  "
                f"loglik={curr_loglik:.4f}  best_so_far={best_loglik:.4f}"
            )

            if curr_loglik < best_loglik:
                print(
                    f"[SelfRefine] Degradation detected "
                    f"({best_loglik:.4f} → {curr_loglik:.4f}). "
                    f"Reverting to best chain and stopping."
                )
                current_dataset = best_dataset
                break

            best_loglik        = curr_loglik
            best_dataset       = current_dataset
            accepted_answer_raw = answer_raw
        else:
            print(f"[SelfRefine] Iteration {iter_label} complete.")

    # ------------------------------------------------------------------
    # Step 3: Final answer selection
    # Reuse the cached answer evaluation when available (avoids a redundant
    # model load); otherwise run the answering model fresh on the final chain.
    # ------------------------------------------------------------------
    print(f"[SelfRefine] Step 3 — Final answer: model={answering_model}  task={answering_full_task}")
    if accepted_answer_raw is not None:
        final_answer_raw = accepted_answer_raw
    else:
        final_answer_raw = run_answering_for_dataset(
            args                  = args,
            answering_model       = answering_model,
            answering_task_name   = answering_full_task,
            dataset_with_reasoning= current_dataset,
            doc_to_text_module    = doc_to_text_module,
        )

    task_def      = tasks.get_task_dict([answering_full_task])[answering_full_task]
    doc_to_choice = task_def.config.doc_to_choice

    predictions_per_input_doc = extract_predictions_from_samples(
        final_answer_raw["samples"][answering_full_task], doc_to_choice
    )

    # Attach full chain / feedback histories to each doc's record
    for doc_id, info in predictions_per_input_doc.items():
        info["doc"]["Reasoning_Chains_History"] = chain_history.get(doc_id, [])
        info["doc"]["Feedback_History"]          = feedback_history.get(doc_id, [])

    return {
        "mode":           "self-refine_CoT",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task":  reasoning_task,
        "answering_task":  answering_task_spec,
        "refine_iterations": n_iterations,
        "results":         format_results_dict(final_answer_raw["results"][answering_full_task]),
        "samples":         predictions_per_input_doc,
    }


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
