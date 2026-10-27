"""TREC CT 2023 scores of trained models, with paired patient-bootstrap CIs (the TREC pre-registration's metrics).

Per run, from logged samples:
- TREC_CT_2023 (`--main`): 3-way macro-F1, accuracy, recall per class, precision of "possibly relevant";
- TREC_CT_AugExp_2023 (`--augexp`, optional): consistency and acc_drop (lm_eval.opencteval.augmentation).
Differences against `--baseline` get a 95% CI from resampling patients (query_id), since one patient's pairs are not
independent. Both tasks are resampled with the same patient draws.

    python scripts/trec_ct_scores.py --run aug-ce=MAIN.jsonl,AUGEXP.jsonl --run ect-ce=MAIN.jsonl,AUGEXP.jsonl \
        --baseline aug-ce
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from lm_eval.opencteval.augmentation import acc_drop, consistency, is_original, macro_f1, prediction


LABELS = ["not relevant", "possibly relevant", "definitely relevant"]
SHORT = ["not", "possibly", "definitely"]
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 0


def read_main(path: Path) -> dict[str, list[tuple[int, int | None]]]:
    """(gold, prediction) pairs per patient."""
    by_patient = defaultdict(list)
    for line in path.open():
        sample = json.loads(line)
        doc = sample["doc"]
        by_patient[str(doc["query_id"])].append((LABELS.index(doc["Label"]), prediction(sample["filtered_resps"], LABELS)))
    return dict(by_patient)


def read_augexp(path: Path) -> dict[str, list[tuple]]:
    """Consistency items (source_id, is_original, gold, prediction) per patient."""
    by_patient = defaultdict(list)
    for line in path.open():
        sample = json.loads(line)
        doc = sample["doc"]
        item = (str(doc["source_id"]), is_original(doc), LABELS.index(doc["Label"]), prediction(sample["filtered_resps"], LABELS))
        by_patient[str(doc["query_id"])].append(item)
    return dict(by_patient)


def main_scores(pairs: list[tuple[int, int | None]]) -> dict[str, float]:
    scores = {
        "macro-F1": macro_f1(pairs),
        "acc": sum(g == p for g, p in pairs) / len(pairs),
    }
    for cls, name in enumerate(SHORT):
        gold = [p for g, p in pairs if g == cls]
        scores[f"recall {name}"] = sum(p == cls for p in gold) / len(gold) if gold else float("nan")
    predicted = [g for g, p in pairs if p == 1]
    scores["precision possibly"] = sum(g == 1 for g in predicted) / len(predicted) if predicted else float("nan")
    return scores


def augexp_scores(items: list[tuple]) -> dict[str, float]:
    return {"AugExp consistency": consistency(items), "AugExp acc_drop": acc_drop(items)}


def scores(run: dict, patients: list[str]) -> dict[str, float]:
    out = main_scores([x for p in patients for x in run["main"].get(p, [])])
    if run["augexp"] is not None:
        out.update(augexp_scores([x for p in patients for x in run["augexp"].get(p, [])]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=MAIN_SAMPLES[,AUGEXP_SAMPLES]")
    parser.add_argument("--baseline")
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, paths = item.split("=", 1)
        main_path, _, augexp_path = paths.partition(",")
        runs[name] = {
            "main": read_main(Path(main_path)),
            "augexp": read_augexp(Path(augexp_path)) if augexp_path else None,
        }
    patients = sorted(set.union(*(set(r["main"]) for r in runs.values())))
    table = {name: scores(run, patients) for name, run in runs.items()}
    metrics = list(next(iter(table.values())))
    print(f"{len(patients)} patients\n")
    print("| run | " + " | ".join(metrics) + " |")
    print("|-|" + "-|" * len(metrics))
    for name, s in table.items():
        print(f"| {name} | " + " | ".join(f"{s.get(m, float('nan')):.3f}" for m in metrics) + " |")
    if not args.baseline:
        return
    print(f"\nDifference vs {args.baseline}, 95% CI (patients resampled, {BOOTSTRAP_ITERATIONS} draws):\n")
    print("| run | metric | diff | CI |")
    print("|-|-|-|-|")
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [rng.choices(patients, k=len(patients)) for _ in range(BOOTSTRAP_ITERATIONS)]
    base = runs[args.baseline]
    base_draws = [scores(base, d) for d in draws]
    for name, run in runs.items():
        if name == args.baseline:
            continue
        run_draws = [scores(run, d) for d in draws]
        for metric in metrics:
            if metric not in table[name] or metric not in table[args.baseline]:
                continue
            diff = table[name][metric] - table[args.baseline][metric]
            boot = sorted(b for b in (r[metric] - s[metric] for r, s in zip(run_draws, base_draws)) if b == b)
            low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
            print(f"| {name} | {metric} | {diff:+.3f} | [{low:+.3f}, {high:+.3f}] |")


if __name__ == "__main__":
    main()
