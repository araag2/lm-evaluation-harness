"""Experimental Self-Refine CoT — multi-checkpoint variant.

Runs the Self-Refine loop (Madaan et al., 2023 — https://arxiv.org/abs/2303.17651)
for up to ``max(CHECKPOINT_ITERS)`` iterations and saves results at every
checkpoint in ``CHECKPOINT_ITERS``.

Key differences from Self_Refine_CoT.py
----------------------------------------
* ``stop_on_degradation`` defaults to **True**.
* Feedback generation is capped to ``FEEDBACK_MAX_NEW_TOKENS`` (1 000 tokens).
* Results are saved at fixed checkpoints (1, 5, 10 iterations) rather than only
  at the end; when early-stopping fires the last accepted result is propagated to
  all remaining checkpoints.
* The function returns a **list** of result dicts, one per checkpoint × task,
  each tagged with ``refine_iterations`` = the checkpoint number.

CLI / args
----------
All args from Self_Refine_CoT.py are supported.  Extra args:

  --refine_iterations   ignored in this mode (checkpoints are fixed).
  --stop_on_degradation defaults to True here; pass --no-stop_on_degradation
                        to disable it.
  --feedback_max_tokens override the 1 000-token cap on feedback generation.
"""

import copy
import re
from typing import Optional
from lm_eval.reasoning_modes.reasoning_utils import *

# ── Experimental knobs ────────────────────────────────────────────────────────
CHECKPOINT_ITERS:       tuple = (1, 5, 10)
FEEDBACK_MAX_NEW_TOKENS: int  = 1_000
MAX_ITERS: int = max(CHECKPOINT_ITERS)
# ─────────────────────────────────────────────────────────────────────────────


def _clean_feedback(text: str) -> str:
    """Strip looping repetitions from model-generated feedback."""
    pattern = re.compile(r"\n{1,2}\*\*Final Answer[:\s]", re.IGNORECASE)
    m = pattern.search(text)
    if m:
        text = text[: m.start()]
    return text.rstrip()


def _cap_model_gen_tokens(model_args: str, max_new_tokens: int) -> str:
    """Return a copy of *model_args* with ``max_gen_toks`` set to *max_new_tokens*.

    If the key already appears it is replaced; otherwise it is appended.
    This is used to limit feedback generation length without affecting other
    model calls.
    """
    pattern = re.compile(r"(max_gen_toks\s*=\s*)[^\s,]+")
    if pattern.search(model_args):
        return pattern.sub(rf"\g<1>{max_new_tokens}", model_args)
    return f"{model_args},max_gen_toks={max_new_tokens}"


def mode_self_refine_CoT_experimental(args: argparse.Namespace) -> list:
    """Experimental Self-Refine CoT pipeline — multi-checkpoint entry point.

    Returns a flat list of result dicts, one per (checkpoint, task).
    """
    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(
            f"[SelfRefineExp][WARNING] Expected exactly one reasoning model and one "
            f"answering model.  Got reasoning_models={args.reasoning_models}, "
            f"answering_models={args.answering_models}."
        )

    if len(args.reasoning_tasks) != len(args.answering_tasks):
        raise ValueError(
            f"reasoning_tasks and answering_tasks must have the same length. "
            f"Got {len(args.reasoning_tasks)} vs {len(args.answering_tasks)}."
        )

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]

    # stop_on_degradation defaults to True in this experimental mode
    check_degradation = getattr(args, "stop_on_degradation", True)

    # Feedback-capped model args (only used for feedback generation)
    feedback_max_tokens = getattr(args, "feedback_max_tokens", FEEDBACK_MAX_NEW_TOKENS)
    reasoning_model_feedback = _cap_model_gen_tokens(reasoning_model, feedback_max_tokens)

    print(
        f"[SelfRefineExp] checkpoints={CHECKPOINT_ITERS}  "
        f"stop_on_degradation={check_degradation}  "
        f"feedback_max_new_tokens={feedback_max_tokens}"
    )

    # ------------------------------------------------------------------
    # Step 1: Initial reasoning — ALL tasks in ONE model load
    # ------------------------------------------------------------------
    print(
        f"[SelfRefineExp] Step 1 — Initial reasoning for "
        f"{len(args.reasoning_tasks)} task(s): model={reasoning_model}"
    )
    all_reasoning_outputs = run_reasoning(args)[reasoning_model]

    # Build per-task state objects
    task_states = []
    for reasoning_task, answering_task_spec in zip(args.reasoning_tasks, args.answering_tasks):
        task_base_name, _      = parse_task_spec(answering_task_spec)
        reasoning_full_task    = reasoning_task.replace(":", "_")
        answering_full_task    = answering_task_spec.replace(":", "_")
        doc_to_text_module     = f"lm_eval.tasks.{task_base_name}.utils"

        reasoning_outputs = all_reasoning_outputs[reasoning_task]
        base_dataset      = load_base_dataset_from_task(reasoning_full_task)
        current_dataset   = inject_reasoning_into_dataset(base_dataset, reasoning_outputs)

        initial_chains = [extract_reasoning_text_from_dicts(s)[0] for s in reasoning_outputs]
        chain_history: Dict[int, List[str]] = {
            s["doc_id"]: [initial_chains[i]] for i, s in enumerate(reasoning_outputs)
        }

        task_states.append({
            "reasoning_task":       reasoning_task,
            "answering_task":       answering_task_spec,
            "reasoning_full_task":  reasoning_full_task,
            "answering_full_task":  answering_full_task,
            "doc_to_text_module":   doc_to_text_module,
            "base_dataset":         base_dataset,
            "current_dataset":      current_dataset,
            "chain_history":        chain_history,
            "feedback_history":     {},
            "best_loglik":          float("-inf"),
            "best_dataset":         current_dataset,
            # checkpoint_results[iter_n] = raw result dict at that checkpoint
            "checkpoint_results":   {},
            "stopped_early":        False,
            "stopped_at_iter":      None,
        })

    # ------------------------------------------------------------------
    # Step 2: Iterative Self-Refine
    # ------------------------------------------------------------------
    print(
        f"[SelfRefineExp] Step 2 — Refinement loop: up to {MAX_ITERS} iterations  "
        f"stop_on_degradation={check_degradation}"
    )

    for iteration in range(1, MAX_ITERS + 1):
        iter_label    = f"{iteration}/{MAX_ITERS}"
        active_states = [s for s in task_states if not s["stopped_early"]]
        if not active_states:
            print(f"[SelfRefineExp] All tasks stopped early — exiting loop.")
            break

        # 2a — Feedback (capped to feedback_max_tokens)
        print(f"[SelfRefineExp] Iteration {iter_label} — Feedback ({len(active_states)} tasks, "
              f"max_gen_toks={feedback_max_tokens})")
        feedback_raw = run_answering_for_datasets(
            args=args,
            answering_model=reasoning_model_feedback,
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

        # 2b — Refinement
        print(f"[SelfRefineExp] Iteration {iter_label} — Refinement ({len(active_states)} tasks)")
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

        # 2c — Degradation check (always run so we have answer scores; only gate
        #      early-stop on check_degradation flag)
        print(f"[SelfRefineExp] Iteration {iter_label} — Answer scoring ({len(active_states)} tasks)")
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
                f"[SelfRefineExp][{s['answering_task']}] iter={iteration}  "
                f"loglik={curr_loglik:.4f}  best_so_far={s['best_loglik']:.4f}"
            )

            degraded = check_degradation and curr_loglik < s["best_loglik"]
            if degraded:
                print(
                    f"[SelfRefineExp][{s['answering_task']}] Degradation detected "
                    f"({s['best_loglik']:.4f} → {curr_loglik:.4f}). Reverting."
                )
                s["current_dataset"] = s["best_dataset"]
                s["stopped_early"]   = True
                s["stopped_at_iter"] = iteration
            else:
                s["best_loglik"]  = curr_loglik
                s["best_dataset"] = s["current_dataset"]
                s["_last_answer_raw"] = answer_raw  # cache for checkpoint use

            # Save checkpoint if this iteration is in CHECKPOINT_ITERS
            if iteration in CHECKPOINT_ITERS:
                # Use the best (non-degraded) result for this checkpoint
                checkpoint_raw = s.get("_last_answer_raw", answer_raw)
                s["checkpoint_results"][iteration] = {
                    "samples": {s["answering_full_task"]: checkpoint_raw["samples"][s["answering_full_task"]]},
                    "results": {s["answering_full_task"]: checkpoint_raw["results"][s["answering_full_task"]]},
                }
                print(f"[SelfRefineExp] Saved checkpoint @ iteration {iteration} for {s['answering_task']}")

        # For tasks that stopped early this iteration, fill forward checkpoints
        for s in [st for st in task_states if st["stopped_early"] and st["stopped_at_iter"] == iteration]:
            for ckpt in CHECKPOINT_ITERS:
                if ckpt > iteration and ckpt not in s["checkpoint_results"]:
                    # Propagate last accepted result to remaining checkpoints
                    last_raw = s.get("_last_answer_raw")
                    if last_raw is not None:
                        s["checkpoint_results"][ckpt] = {
                            "samples": {s["answering_full_task"]: last_raw["samples"][s["answering_full_task"]]},
                            "results": {s["answering_full_task"]: last_raw["results"][s["answering_full_task"]]},
                        }
                        print(
                            f"[SelfRefineExp] Propagating early-stop result to "
                            f"checkpoint {ckpt} for {s['answering_task']}"
                        )

    # ------------------------------------------------------------------
    # Step 3: Final pass for any checkpoint not yet filled
    #         (covers tasks that never triggered degradation)
    # ------------------------------------------------------------------
    print(f"[SelfRefineExp] Step 3 — Filling any missing checkpoints.")
    for s in task_states:
        missing_ckpts = [c for c in CHECKPOINT_ITERS if c not in s["checkpoint_results"]]
        if not missing_ckpts:
            continue
        # Run a final answer pass on the current (best) dataset
        final_raw = run_answering_for_datasets(
            args=args,
            answering_model=answering_model,
            tasks_and_datasets=[(s["answering_full_task"], s["current_dataset"])],
            doc_to_text_module=s["doc_to_text_module"],
        )
        for ckpt in missing_ckpts:
            s["checkpoint_results"][ckpt] = {
                "samples": {s["answering_full_task"]: final_raw["samples"][s["answering_full_task"]]},
                "results": {s["answering_full_task"]: final_raw["results"][s["answering_full_task"]]},
            }
            print(f"[SelfRefineExp] Filled missing checkpoint {ckpt} for {s['answering_task']}")

    # ------------------------------------------------------------------
    # Step 4: Build output list — one dict per (checkpoint × task)
    # ------------------------------------------------------------------
    output_list = []
    for s in task_states:
        task_def      = tasks.get_task_dict([s["answering_full_task"]])[s["answering_full_task"]]
        doc_to_choice = task_def.config.doc_to_choice

        for ckpt in sorted(CHECKPOINT_ITERS):
            raw = s["checkpoint_results"].get(ckpt)
            if raw is None:
                print(f"[SelfRefineExp][WARNING] No result for checkpoint {ckpt} / {s['answering_task']}")
                continue

            predictions_per_input_doc = extract_predictions_from_samples(
                raw["samples"][s["answering_full_task"]], doc_to_choice
            )
            for doc_id, info in predictions_per_input_doc.items():
                info["doc"]["Reasoning_Chains_History"] = s["chain_history"].get(doc_id, [])
                info["doc"]["Feedback_History"]          = s["feedback_history"].get(doc_id, [])

            output_list.append({
                "mode":              "self-refine_CoT_experimental",
                "reasoning_model":   reasoning_model,
                "answering_model":   answering_model,
                "reasoning_task":    s["reasoning_task"],
                "answering_task":    s["answering_task"],
                "refine_iterations": ckpt,
                "stopped_early":     s["stopped_early"],
                "stopped_at_iter":   s["stopped_at_iter"],
                "results":           format_results_dict(raw["results"][s["answering_full_task"]]),
                "samples":           predictions_per_input_doc,
            })

    return output_list


# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────

def _mean_max_loglik(samples: List[dict]) -> float:
    """Mean of each doc's best log-likelihood across answer choices.

    Used as a proxy for model confidence to detect performance degradation.
    A higher (less negative) value means the model is more confident overall.
    """
    if not samples:
        return float("-inf")
    return sum(
        max(prob[0][0] for prob in sample["resps"])
        for sample in samples
    ) / len(samples)
