"""SemEval-2024 NLI4CT official scores of logged runs, per intervention, with bootstrap CIs.

Re-scores `samples_SemEval_NLI4CT_2024_*.jsonl` files from their logged documents and answers, so runs
logged before the task reported Faithfulness and Consistency are scored the same way as new ones.
Differences against `--baseline` get a 95% CI from resampling base statements (each with its variants).

    python scripts/semeval2024_scores.py --run base-ce=PATH --run aug-ce=PATH --baseline base-ce
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from lm_eval.opencteval.augmentation import consistency, faithfulness, positive_f1


BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 0
INTERVENTIONS = (
    "Paraphrase",
    "Numerical_paraphrase",
    "Text_appended",
    "Contradiction",
    "Numerical_contradiction",
)
ALTERING_INTERVENTIONS = (
    "Contradiction",
    "Numerical_contradiction",
)  # the only ones with Altering rows


def load_process_results():
    # The task folder is named "0-shot", which is not importable by name.
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "lm_eval/tasks/SemEval_NLI4CT/0-shot/utils.py"
    )
    spec = importlib.util.spec_from_file_location("semeval_0shot_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_results_2024


def read_items(path: Path, process_results) -> dict[str, list[tuple]]:
    """Scored items grouped by base statement: (source_id, is_original, gold, pred, intervention)."""
    groups = defaultdict(list)
    for line in path.open():
        sample = json.loads(line)
        doc = sample["doc"]
        scores = process_results(doc, sample["filtered_resps"])
        item = scores["consistency"]
        groups[item[0]].append((*item, doc["Intervention"]))
    return groups


def scores(
    groups: dict[str, list[tuple]], keys: list[str], detail: bool = True
) -> dict[str, float]:
    items = [item for key in keys for item in groups[key]]
    core = [item[:4] for item in items]
    f1 = positive_f1([(item[2], item[3]) for item in core if item[1]])
    faith, cons = faithfulness(core), consistency(core)
    result = {"F1": f1, "Faith": faith, "Cons": cons, "avg": (f1 + faith + cons) / 3}
    if not detail:
        return result
    for name in INTERVENTIONS:
        subset = [item[:4] for item in items if item[1] or item[4] == name]
        result[f"Cons {name}"] = consistency(subset)
    for name in ALTERING_INTERVENTIONS:
        subset = [item[:4] for item in items if item[1] or item[4] == name]
        result[f"Faith {name}"] = faithfulness(subset)
    return result


def bootstrap(runs, baseline, keys, rng) -> dict[str, dict[str, tuple[float, float]]]:
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [rng.choice(keys) for _ in keys]
        base = scores(runs[baseline], sample, detail=False)
        for name, groups in runs.items():
            if name == baseline:
                continue
            for metric, value in scores(groups, sample, detail=False).items():
                draws[name][metric].append(value - base[metric])
    low, high = int(0.025 * BOOTSTRAP_ITERATIONS), int(0.975 * BOOTSTRAP_ITERATIONS) - 1
    return {
        name: {
            metric: (sorted(v)[low], sorted(v)[high]) for metric, v in metrics.items()
        }
        for name, metrics in draws.items()
    }


def fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=samples_SemEval_NLI4CT_2024_*.jsonl",
    )
    parser.add_argument(
        "--baseline", help="Run name that differences and CIs are taken against"
    )
    parser.add_argument("--output", type=Path, help="Write the Markdown here as well")
    args = parser.parse_args()

    process_results = load_process_results()
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = read_items(Path(path), process_results)
    keys = sorted(next(iter(runs.values())))
    table = {name: scores(groups, keys) for name, groups in runs.items()}
    metrics = list(next(iter(table.values())))

    lines = ["| run | " + " | ".join(metrics[:4]) + " |", "|-" * 5 + "|"]
    lines += [
        f"| {n} | " + " | ".join(fmt(t[m]) for m in metrics[:4]) + " |"
        for n, t in table.items()
    ]
    lines += [
        "",
        "| run | " + " | ".join(m.replace(" ", "<br>") for m in metrics[4:]) + " |",
    ]
    lines += ["|-" * (len(metrics) - 3) + "|"]
    lines += [
        f"| {n} | " + " | ".join(fmt(t[m]) for m in metrics[4:]) + " |"
        for n, t in table.items()
    ]
    if args.baseline:
        cis = bootstrap(runs, args.baseline, keys, random.Random(BOOTSTRAP_SEED))
        lines += [
            "",
            f"Difference vs {args.baseline}, 95% CI (bootstrap over {len(keys)} base statements):",
            "",
        ]
        lines += ["| run | metric | diff | CI |", "|-|-|-|-|"]
        for name, ci in cis.items():
            for metric in ci:
                diff = table[name][metric] - table[args.baseline][metric]
                lines.append(
                    f"| {name} | {metric} | {diff:+.3f} | [{ci[metric][0]:+.3f}, {ci[metric][1]:+.3f}] |"
                )
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
