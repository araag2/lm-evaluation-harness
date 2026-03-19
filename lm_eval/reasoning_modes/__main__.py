import copy
import json
import argparse
import os
import numpy as np

from datetime import datetime

from lm_eval.reasoning_modes.multi_turn_CoT import mode_multi_turn_CoT
from lm_eval.reasoning_modes.multi_turn_CoT_SC import mode_multi_turn_CoT_SC
from lm_eval.reasoning_modes.cross_consistency import mode_cross_consistency
from lm_eval.reasoning_modes.only_vote import mode_only_vote
from lm_eval.reasoning_modes.Self_Refine_CoT import mode_self_refine_CoT
from lm_eval.reasoning_modes.Self_Refine_CoT_Experimental import mode_self_refine_CoT_experimental

# ---------------------------------------------------------------------------
# Mode registry — add new modes here without touching main()
# ---------------------------------------------------------------------------
MODE_REGISTRY = {
    "multi-turn_CoT":                  mode_multi_turn_CoT,
    "multi-turn_CoT-SC":               mode_multi_turn_CoT_SC,
    "cross-consistency":               mode_cross_consistency,
    "only-vote":                       mode_only_vote,
    "self-refine_CoT":                 mode_self_refine_CoT,
    "self-refine_CoT_experimental":    mode_self_refine_CoT_experimental,
}


def make_summary_output(out: dict) -> dict:
    """Return a copy of ``out`` with each sample stripped to essentials.

    * ``doc`` is reduced to ``id`` and ``Label`` only.
    * Full prompt strings (``answering_prompt``, any ``*_Prompt`` doc fields)
      are removed to keep the summary file compact.
    """
    summary = copy.deepcopy(out)
    for info in summary.get("samples", {}).values():
        if "doc" in info:
            doc = info["doc"]
            info["doc"] = {k: doc[k] for k in ("id", "Label") if k in doc}
        # Drop verbose prompt strings that belong in FullSamples only
        info.pop("answering_prompt", None)
    return summary


def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')


def make_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)


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

    # Mode
    parser.add_argument(
        "--mode", type=str, default="multi-turn_CoT",
        choices=list(MODE_REGISTRY.keys()),
        help="Reasoning pipeline mode."
    )
    parser.add_argument("--vote_file", type=str, default=None)

    # Voting flags (used by only-vote and cross-consistency)
    parser.add_argument('--no-simple-voting', dest='simple_voting', action='store_false',
                        help="Skip simple voting strategies (majority, logits, etc.)")
    parser.add_argument('--no-mbr-voting', dest='mbr_voting', action='store_false',
                        help="Skip MBR voting strategies.")
    parser.set_defaults(simple_voting=True, mbr_voting=False)

    # Tier-2 LLM-scored voting (opt-in; requires explicit strategy names)
    from lm_eval.reasoning_modes.voting.strategies.llm_metrics import TIER2_STRATEGY_META
    parser.add_argument(
        '--tier2_strategies', nargs='+', default=[],
        choices=list(TIER2_STRATEGY_META.keys()),
        metavar='STRATEGY',
        help=(
            "One or more Tier-2 chain-scoring strategies to run after the main SC loop. "
            "Available: " + ", ".join(TIER2_STRATEGY_META.keys())
        ),
    )
    parser.add_argument(
        '--judge_model', type=str, default=None,
        help=(
            "lm-eval model_args string for the external judge model used by "
            "'judge'-source Tier-2 strategies (e.g. 'pretrained=Qwen/Qwen3-8B')."
        ),
    )

    # Self-Refine flags
    parser.add_argument('--refine_iterations', type=int, default=1,
                        help="Number of Self-Refine feedback+refinement iterations.")
    parser.add_argument('--stop_on_degradation', action='store_true',
                        help="Stop Self-Refine early if answer log-likelihood decreases.")
    parser.add_argument('--no_stop_on_degradation', action='store_true',
                        help="Disable degradation-based early stopping (overrides the "
                             "self-refine_CoT_experimental default of True).")
    parser.add_argument('--feedback_max_tokens', type=int, default=1000,
                        help="Max tokens for feedback generation in Self-Refine (default: 1000).")

    # Output Args
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    # self-refine_CoT_experimental defaults stop_on_degradation to True
    # unless the user explicitly opts out with --no_stop_on_degradation
    if args.mode == "self-refine_CoT_experimental" and not args.no_stop_on_degradation:
        args.stop_on_degradation = True

    mode_fn = MODE_REGISTRY.get(args.mode)
    if mode_fn is None:
        raise ValueError(f"Unknown mode '{args.mode}'. Available: {list(MODE_REGISTRY.keys())}")

    out = mode_fn(args)

    if args.output_path:
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M')
        # mode_fn may return a list (batched multi-task) or a single dict
        outputs = out if isinstance(out, list) else [out]

        for result in outputs:
            task_key   = result.get("answering_task", result.get("reasoning_task", "unknown"))
            task_name  = task_key.split(":")[0]          # "MedQA:0-shot" → "MedQA"
            # Support both singular (CoT) and plural (cross-consistency) model keys
            import re as _re
            raw_model = result.get("reasoning_model") or result.get("answering_model")
            if raw_model is None:
                model_list = result.get("reasoning_models") or result.get("answering_models") or []
                raw_model  = "_".join(
                    _re.search(r'pretrained=([^,)]+)', m).group(1).replace('/', '_')
                    for m in model_list
                    if _re.search(r'pretrained=([^,)]+)', m)
                ) or "unknown"
                model_name = raw_model
            else:
                m = _re.search(r'pretrained=([^,)]+)', raw_model)
                model_name = m.group(1).replace('/', '_') if m else raw_model

            task_output_path = os.path.join(args.output_path, task_name, model_name)
            os.makedirs(task_output_path, exist_ok=True)
            with safe_open_w(os.path.join(task_output_path, f"FullSamples_{timestamp}.json")) as f:
                json.dump(result, f, indent=4, default=make_json_serializable)
            with safe_open_w(os.path.join(task_output_path, f"Summary_{timestamp}.json")) as f:
                json.dump(make_summary_output(result), f, indent=4, default=make_json_serializable)
            print(f"\n✅ Results written to {task_output_path}")


if __name__ == "__main__":
    main()