import copy
import json
import argparse
import importlib
import os
import pprint
import numpy as np

from datetime import datetime
from typing import Dict, List, Tuple, Any, Union
from lm_eval import evaluator, tasks
from datasets import Dataset, DatasetDict

from lm_eval.reasoning_modes.multi_turn_CoT import mode_multi_turn_CoT

from lm_eval.reasoning_modes.multi_turn_CoT_SC import mode_multi_turn_CoT_SC

from lm_eval.reasoning_modes.cross_consistency import mode_cross_consistency

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

    # Modes
    parser.add_argument("--mode", type=str, default="multi-turn",
                        choices=["multi-turn_CoT", "multi-turn_CoT-SC", "cross-consistency"])

    # Output Args
    parser.add_argument('--output_path', type=str, default="/cfs/home/u021010/PhD/active_dev/outputs/CoT-Debug/")
    args = parser.parse_args()

    out = None
    match args.mode:
        case "multi-turn_CoT":
            out = mode_multi_turn_CoT(args)
        case "multi-turn_CoT-SC":
            out = mode_multi_turn_CoT_SC(args)
        case "cross-consistency":
            out = mode_cross_consistency(args)
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")
        
    print("\n==== RESULTS (summary keys only) ====")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))

    if args.output_path:
        with safe_open_w(f"{args.output_path}Summary_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.json") as f:
            json.dump(out, f, indent=4, default=make_json_serializable)

        #with safe_open_w(f"{args.output_path}Samples_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.json") as f:
            #json.dump(out, f, indent=4, default=make_json_serializable)

        print(f"\n✅ Results written to {args.output_path}")
        
if __name__ == "__main__":
    main()