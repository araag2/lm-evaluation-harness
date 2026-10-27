"""Paraphrase-set scores with paired CIs (pre-registration 2026-10-27_gate_and_transfer_prereg.md).

For one task's logged samples per run (legacy `<id>_paraphraseN` sets or AugExp exports):
- acc original: accuracy on the originals;
- acc paraphrase: accuracy on the label-preserving variants;
- consistency: share of those variants predicted like their original.
Differences against `--baseline` get a 95% CI from resampling original items (families), paired across runs.

    python scripts/transfer_scores.py --run base-ce=SAMPLES.jsonl --run aug-ce=SAMPLES.jsonl --baseline base-ce
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from consensus_gate_stats import family_of, predicted


METRICS = ("acc original", "acc paraphrase", "consistency")
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 0


def read_run(path: Path) -> dict[str, dict]:
    """Per family: the original (gold, prediction, correct) and its label-preserving variants."""
    groups = defaultdict(lambda: {"original": None, "variants": []})
    for line in path.open():
        sample = json.loads(line)
        key, original = family_of(sample["doc"])
        row = (str(sample["target"]), predicted(sample), bool(sample["acc"]))
        if original:
            groups[key]["original"] = row
        else:
            groups[key]["variants"].append(row)
    out = {}
    for key, group in groups.items():
        original = group["original"]
        if original is None:
            continue
        out[key] = {"original": original, "variants": [v for v in group["variants"] if v[0] == original[0]]}
    return out


def scores(run: dict, keys: list[str]) -> dict[str, float]:
    originals = [run[k]["original"] for k in keys]
    pairs = [(run[k]["original"], v) for k in keys for v in run[k]["variants"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "acc original": mean([o[2] for o in originals]),
        "acc paraphrase": mean([v[2] for _, v in pairs]),
        "consistency": mean([o[1] == v[1] for o, v in pairs]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=SAMPLES.jsonl")
    parser.add_argument("--baseline")
    args = parser.parse_args()

    runs = {name: read_run(Path(path)) for name, path in (item.split("=", 1) for item in args.run)}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    table = {name: scores(run, keys) for name, run in runs.items()}
    print(f"{len(keys)} original items\n")
    print("| run | " + " | ".join(METRICS) + " |")
    print("|-|" + "-|" * len(METRICS))
    for name, s in table.items():
        print(f"| {name} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS) + " |")
    if not args.baseline:
        return
    print(f"\nDifference vs {args.baseline}, 95% CI (original items resampled, {BOOTSTRAP_ITERATIONS} draws):\n")
    print("| run | metric | diff | CI |")
    print("|-|-|-|-|")
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [rng.choices(keys, k=len(keys)) for _ in range(BOOTSTRAP_ITERATIONS)]
    base = runs[args.baseline]
    base_draws = [scores(base, d) for d in draws]
    for name, run in runs.items():
        if name == args.baseline:
            continue
        run_draws = [scores(run, d) for d in draws]
        for metric in METRICS:
            diff = table[name][metric] - table[args.baseline][metric]
            boot = sorted(b for b in (r[metric] - s[metric] for r, s in zip(run_draws, base_draws)) if b == b)
            low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
            print(f"| {name} | {metric} | {diff:+.3f} | [{low:+.3f}, {high:+.3f}] |")


if __name__ == "__main__":
    main()
