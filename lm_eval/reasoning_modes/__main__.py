import json
import argparse
import os
import numpy as np

from datetime import datetime

from lm_eval.reasoning_modes.multi_turn_CoT import mode_multi_turn_CoT
from lm_eval.reasoning_modes.multi_turn_CoT_SC import mode_multi_turn_CoT_SC
from lm_eval.reasoning_modes.cross_consistency import mode_cross_consistency
from lm_eval.reasoning_modes.only_vote import mode_only_vote

# ---------------------------------------------------------------------------
# Mode registry — add new modes here without touching main()
# ---------------------------------------------------------------------------
MODE_REGISTRY = {
    "multi-turn_CoT":    mode_multi_turn_CoT,
    "multi-turn_CoT-SC": mode_multi_turn_CoT_SC,
    "cross-consistency": mode_cross_consistency,
    "only-vote":         mode_only_vote,
}


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

    # Output Args
    parser.add_argument('--output_path', type=str, default="/cfs/home/u021010/PhD/active_dev/outputs/CoT-Debug/")

    args = parser.parse_args()

    mode_fn = MODE_REGISTRY.get(args.mode)
    if mode_fn is None:
        raise ValueError(f"Unknown mode '{args.mode}'. Available: {list(MODE_REGISTRY.keys())}")

    out = mode_fn(args)

    if args.output_path:
        with safe_open_w(f"{args.output_path}Summary_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.json") as f:
            json.dump(out, f, indent=4, default=make_json_serializable)
        print(f"\n✅ Results written to {args.output_path}")


if __name__ == "__main__":
    main()