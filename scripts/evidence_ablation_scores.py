"""Evidence-ablation scores of logged runs: does a model's prediction follow the gold evidence?

Reads `samples_SemEval_NLI4CT_EvidenceAblation_*.jsonl` files. Per run:
- full acc: accuracy with the full section(s) (the SemEval-2024 base statements);
- sufficiency: evidence-only predictions equal to the full-section prediction (all, Entailment, Contradiction);
- necessity: Entailment statements predicted right with the full section that lose Entailment once the
  evidence is removed; faithfulness: the same without requiring the full prediction to be right;
- removed (Con) acc: Contradiction statements still predicted Contradiction once the evidence is removed.
With `--judge` (from `verify_evidence_views.py`), necessity is also given on the statements whose full view the
judge found supported and whose evidence-removed view it found not supported. Differences against
`--baseline` get a 95% CI from resampling statements.

    python scripts/evidence_ablation_scores.py --run base-ce=PATH --run ect=PATH --baseline base-ce \
        --judge data/nli4ct_evidence_ablation/judge.jsonl
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from lm_eval.opencteval.augmentation import _mean, prediction


BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 0
CHOICES = ["Contradiction", "Entailment"]


def read_views(path: Path) -> dict[str, dict]:
    """Per statement: gold label and the predicted label of each view."""
    statements = defaultdict(lambda: {"views": {}})
    for line in path.open():
        sample = json.loads(line)
        doc = sample["doc"]
        pred = prediction(sample["filtered_resps"], CHOICES)
        entry = statements[doc["source_id"]]
        entry["gold"] = doc["Gold_Label"]
        entry["views"][doc["view"]] = None if pred is None else CHOICES[pred]
    return dict(statements)


def verified_ids(path: Path) -> set[str]:
    verdicts = defaultdict(dict)
    for line in path.open():
        row = json.loads(line)
        verdicts[row["source_id"]][row["view"]] = row["verdict"]
    return {
        sid
        for sid, v in verdicts.items()
        if v.get("full") == "SUPPORTED" and v.get("evidence_removed") == "NOT_SUPPORTED"
    }


def scores(
    statements: dict, keys: list[str], verified: set[str] | None
) -> dict[str, float]:
    rows = [statements[k] for k in keys]
    ablated = [r for r in rows if "evidence_removed" in r["views"]]
    entail = [r for r in ablated if r["gold"] == "Entailment"]
    result = {
        "full acc": _mean([float(r["views"]["full"] == r["gold"]) for r in rows]),
        "suff": _mean(
            [float(r["views"]["evidence_only"] == r["views"]["full"]) for r in ablated]
        ),
        "suff Ent": _mean(
            [float(r["views"]["evidence_only"] == r["views"]["full"]) for r in entail]
        ),
        "suff Con": _mean(
            [
                float(r["views"]["evidence_only"] == r["views"]["full"])
                for r in ablated
                if r["gold"] == "Contradiction"
            ]
        ),
        "necessity": _mean(
            [
                float(r["views"]["evidence_removed"] == "Contradiction")
                for r in entail
                if r["views"]["full"] == "Entailment"
            ]
        ),
        "faith": _mean(
            [float(r["views"]["evidence_removed"] == "Contradiction") for r in entail]
        ),
        "removed Con acc": _mean(
            [
                float(r["views"]["evidence_removed"] == "Contradiction")
                for r in ablated
                if r["gold"] == "Contradiction"
            ]
        ),
    }
    if verified is not None:
        sure = [r for k, r in zip(keys, rows, strict=True) if k in verified]
        result["necessity (judged)"] = _mean(
            [
                float(r["views"]["evidence_removed"] == "Contradiction")
                for r in sure
                if r["views"]["full"] == "Entailment"
            ]
        )
    return result


def fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", action="append", required=True, help="NAME=samples_*.jsonl"
    )
    parser.add_argument(
        "--baseline", help="Run name that differences and CIs are taken against"
    )
    parser.add_argument(
        "--judge", type=Path, help="judge.jsonl from verify_evidence_views.py"
    )
    parser.add_argument("--output", type=Path, help="Write the Markdown here as well")
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = read_views(Path(path))
    keys = sorted(next(iter(runs.values())))
    verified = verified_ids(args.judge) if args.judge else None
    table = {
        name: scores(statements, keys, verified) for name, statements in runs.items()
    }
    metrics = list(next(iter(table.values())))

    first = next(iter(runs.values()))
    n_ablated = sum("evidence_removed" in first[k]["views"] for k in keys)
    n_entail = sum(
        "evidence_removed" in first[k]["views"] and first[k]["gold"] == "Entailment"
        for k in keys
    )
    lines = [
        f"{len(keys)} statements, {n_ablated} with partial evidence ({n_entail} Entailment)"
        + (f", {len(verified)} judge-verified" if verified is not None else ""),
        "",
        "| run | " + " | ".join(metrics) + " |",
        "|-" * (len(metrics) + 1) + "|",
    ]
    lines += [
        f"| {n} | " + " | ".join(fmt(t[m]) for m in metrics) + " |"
        for n, t in table.items()
    ]
    if args.baseline:
        rng = random.Random(BOOTSTRAP_SEED)
        draws = defaultdict(lambda: defaultdict(list))
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = [rng.choice(keys) for _ in keys]
            base = scores(runs[args.baseline], sample, verified)
            for name, statements in runs.items():
                if name != args.baseline:
                    for metric, value in scores(statements, sample, verified).items():
                        draws[name][metric].append(value - base[metric])
        low, high = (
            int(0.025 * BOOTSTRAP_ITERATIONS),
            int(0.975 * BOOTSTRAP_ITERATIONS) - 1,
        )
        lines += [
            "",
            f"Difference vs {args.baseline}, 95% CI:",
            "",
            "| run | metric | diff | CI |",
        ]
        lines += ["|-|-|-|-|"]
        for name, by_metric in draws.items():
            for metric, values in by_metric.items():
                values = sorted(v for v in values if not math.isnan(v))
                diff = table[name][metric] - table[args.baseline][metric]
                lo, hi = (
                    values[int(low * len(values) / BOOTSTRAP_ITERATIONS)],
                    values[int(high * len(values) / BOOTSTRAP_ITERATIONS)],
                )
                lines.append(
                    f"| {name} | {metric} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
                )
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
