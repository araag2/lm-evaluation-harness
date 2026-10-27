"""TREC CT necessity probe scores (scripts/build_trec_probe.py): does a model drop "definitely relevant" when a
criterion now excludes the patient, but keep it when criteria are only removed?

Per run, over the eligible pairs whose original the model grades "definitely relevant":
- flip moved: flip_criterion views no longer graded "definitely relevant" (the necessity analogue);
- flip exact: flip views graded "possibly relevant" (TREC's grade for an excluded patient);
- subset moved: subset_criteria views no longer graded "definitely relevant" (control: a criteria edit that keeps
  eligibility);
- selectivity: flip moved - subset moved.
- all-pairs selectivity (secondary, added before the trained-run results): over all eligible pairs, whatever the
  original's prediction, the share of subset views graded "definitely relevant" minus the share of flip views graded
  so. Every run is scored on the same views, and pairs the model already misses are not dropped.
Differences against `--baseline` get a 95% CI from resampling patients (query_id), as the pairs of one patient are
not independent.

    python scripts/trec_probe_scores.py --run base-ce=SAMPLES.jsonl --run ect-ce=SAMPLES.jsonl --baseline base-ce
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from lm_eval.opencteval.augmentation import prediction


LABELS = ["not relevant", "possibly relevant", "definitely relevant"]
ELIGIBLE = "definitely relevant"
METRICS = ("pairs", "flip moved", "flip exact", "subset moved", "selectivity", "all-pairs selectivity")
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 0


def read_run(path: Path) -> dict[str, dict]:
    """Per eligible pair: its patient and the predicted grade of each of its views."""
    pairs = defaultdict(lambda: {"views": defaultdict(list)})
    for line in path.open():
        sample = json.loads(line)
        doc = sample["doc"]
        pred = prediction(sample["filtered_resps"], LABELS)
        entry = pairs[doc["source_id"]]
        entry["patient"] = doc["query_id"]
        entry["views"][doc["view"]].append(None if pred is None else LABELS[pred])
    return dict(pairs)


def scores(pairs: dict, keys: list[str]) -> dict[str, float]:
    every = [pairs[k]["views"] for k in keys]
    kept = [v for v in every if v["original"] == [ELIGIBLE]]
    flips = [p for v in kept for p in v.get("flip_criterion", [])]
    subsets = [p for v in kept for p in v.get("subset_criteria", [])]

    def share(preds, test):
        return sum(test(p) for p in preds) / len(preds) if preds else float("nan")

    flip_moved = share(flips, lambda p: p != ELIGIBLE)
    subset_moved = share(subsets, lambda p: p != ELIGIBLE)
    return {
        "pairs": len(kept),
        "flip moved": flip_moved,
        "flip exact": share(flips, lambda p: p == "possibly relevant"),
        "subset moved": subset_moved,
        "selectivity": flip_moved - subset_moved,
        "all-pairs selectivity": share([p for v in every for p in v.get("subset_criteria", [])], lambda p: p == ELIGIBLE)
        - share([p for v in every for p in v.get("flip_criterion", [])], lambda p: p == ELIGIBLE),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=SAMPLES.jsonl")
    parser.add_argument("--baseline")
    args = parser.parse_args()

    runs = {name: read_run(Path(path)) for name, path in (item.split("=", 1) for item in args.run)}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    first = next(iter(runs.values()))
    by_patient = defaultdict(list)
    for k in keys:
        by_patient[first[k]["patient"]].append(k)
    patients = sorted(by_patient)
    print(f"{len(keys)} eligible pairs, {len(patients)} patients\n")
    print("| run | " + " | ".join(METRICS) + " |")
    print("|-|" + "-|" * len(METRICS))
    for name, pairs in runs.items():
        s = scores(pairs, keys)
        print(f"| {name} | {s['pairs']} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS[1:]) + " |")
    if not args.baseline:
        return
    print(f"\nDifference vs {args.baseline}, 95% CI (patients resampled):\n")
    print("| run | metric | diff | CI |")
    print("|-|-|-|-|")
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        [k for p in rng.choices(patients, k=len(patients)) for k in by_patient[p]]
        for _ in range(BOOTSTRAP_ITERATIONS)
    ]
    base = runs[args.baseline]
    for name, pairs in runs.items():
        if name == args.baseline:
            continue
        for metric in METRICS[1:]:
            diff = scores(pairs, keys)[metric] - scores(base, keys)[metric]
            boot = [scores(pairs, d)[metric] - scores(base, d)[metric] for d in draws]
            boot = sorted(b for b in boot if b == b)  # a draw can leave a view type empty; drop NaN before sorting
            low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
            print(f"| {name} | {metric} | {diff:+.3f} | [{low:+.3f}, {high:+.3f}] |")


if __name__ == "__main__":
    main()
