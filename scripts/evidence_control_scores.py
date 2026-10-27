"""Is a model's necessity about the evidence, or about any removal? Scores the evidence ablation against its matched
control (scripts/build_evidence_control.py).

Per run, on Entailment statements predicted Entailment with the full section(s):
- necessity: lost Entailment once the evidence is removed (`evidence_removed`);
- random flip: lost Entailment once as many random non-evidence lines are removed (`random_removed`);
- selectivity: necessity - random flip. A model that follows the evidence has high necessity and a low random flip.
Also, on Contradiction statements, the share still predicted Contradiction after a random removal. Reported on the
statements whose control removes as many lines as the evidence has (matched) and on all 205. Differences against
`--baseline` get a 95% CI from resampling statements.

    python scripts/evidence_control_scores.py --run p2-aug-ce=ABLATION.jsonl,CONTROL.jsonl \
        --run p3-ect-ce=ABLATION.jsonl,CONTROL.jsonl --baseline p2-aug-ce
"""

import argparse
import json
import random
from pathlib import Path

from evidence_ablation_scores import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, read_views


METRICS = ("n", "necessity", "random flip", "selectivity", "random Con kept")


def read_run(ablation: Path, control: Path) -> dict[str, dict]:
    statements = read_views(ablation)
    for sid, entry in read_views(control).items():
        statements[sid]["views"].update(entry["views"])
    return {sid: s for sid, s in statements.items() if "random_removed" in s["views"]}


def scores(statements: dict, keys: list[str]) -> dict[str, float]:
    kept = [statements[k]["views"] for k in keys
            if statements[k]["gold"] == "Entailment" and statements[k]["views"].get("full") == "Entailment"]
    con = [statements[k]["views"] for k in keys if statements[k]["gold"] == "Contradiction"]

    def share(rows, view, label):
        return sum(r.get(view) == label for r in rows) / len(rows) if rows else float("nan")

    necessity = share(kept, "evidence_removed", "Contradiction")
    flip = share(kept, "random_removed", "Contradiction")
    return {
        "n": len(kept),
        "necessity": necessity,
        "random flip": flip,
        "selectivity": necessity - flip,
        "random Con kept": share(con, "random_removed", "Contradiction"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=ABLATION_SAMPLES,CONTROL_SAMPLES")
    parser.add_argument("--baseline")
    parser.add_argument("--control-data", type=Path, default=Path("data/nli4ct_evidence_ablation/control.jsonl"))
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, paths = item.split("=", 1)
        ablation, control = paths.split(",")
        runs[name] = read_run(Path(ablation), Path(control))
    matched = {row["source_id"] for row in map(json.loads, args.control_data.open()) if row["matched"]}
    every = sorted(set.intersection(*(set(r) for r in runs.values())))
    subsets = {"matched": [k for k in every if k in matched], "all": every}

    for subset, keys in subsets.items():
        print(f"\n### {subset}: {len(keys)} statements\n")
        print("| run | " + " | ".join(METRICS) + " |")
        print("|-|" + "-|" * len(METRICS))
        for name, statements in runs.items():
            s = scores(statements, keys)
            print(f"| {name} | {s['n']} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS[1:]) + " |")
        if not args.baseline:
            continue
        print(f"\nDifference vs {args.baseline}, 95% CI:\n")
        print("| run | metric | diff | CI |")
        print("|-|-|-|-|")
        rng = random.Random(BOOTSTRAP_SEED)
        draws = [rng.choices(keys, k=len(keys)) for _ in range(BOOTSTRAP_ITERATIONS)]
        base = runs[args.baseline]
        for name, statements in runs.items():
            if name == args.baseline:
                continue
            for metric in METRICS[1:]:
                diff = scores(statements, keys)[metric] - scores(base, keys)[metric]
                boot = sorted(scores(statements, d)[metric] - scores(base, d)[metric] for d in draws)
                low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
                print(f"| {name} | {metric} | {diff:+.3f} | [{low:+.3f}, {high:+.3f}] |")


if __name__ == "__main__":
    main()
