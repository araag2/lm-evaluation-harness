"""What would a consensus gate over augmentations keep? (pre-registration 2026-10-27_gate_and_transfer_prereg.md)

A family is an original and its label-preserving variants: AugExp exports (`source_id`, `augmentation.mode`) or the
legacy paraphrase sets (`<id>_paraphraseN`). Per run:
- all correct / mixed / all wrong: share of families whose original and variants are all right, partly right, all
  wrong. A gate that drops families the model already solves keeps "mixed" + "all wrong";
- wrong among kept: share of kept families that are all wrong (consistent errors or label noise);
- acc if agree, acc if disagree: accuracy of the original when every variant is predicted like it, or not;
- AUROC: share of variants predicted like the original, as a score for "the original is correct".

    python scripts/consensus_gate_stats.py --run untrained=SAMPLES.jsonl --run base-ce=SAMPLES.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lm_eval.opencteval.augmentation import is_original
from lm_eval.opencteval.metrics import _base_intervention_id


METRICS = ("families", "all correct", "mixed", "all wrong", "kept", "wrong among kept",
           "acc if agree", "acc if disagree", "AUROC")


def predicted(sample: dict) -> int:
    scores = [float(r[0]) for r in sample["filtered_resps"]]
    return scores.index(max(scores))


def family_of(doc: dict) -> tuple[str, bool]:
    if doc.get("augmentation") is not None:
        return str(doc["source_id"]), is_original(doc)
    base = _base_intervention_id(str(doc["id"]))
    return base, base == str(doc["id"])


def read_families(path: Path) -> list[dict]:
    """Families with an original and at least one variant of the same gold label."""
    groups = defaultdict(lambda: {"original": None, "variants": []})
    for line in path.open():
        sample = json.loads(line)
        key, original = family_of(sample["doc"])
        row = {"gold": str(sample["target"]), "pred": predicted(sample), "correct": bool(sample["acc"])}
        if original:
            groups[key]["original"] = row
        else:
            groups[key]["variants"].append(row)
    families = []
    for group in groups.values():
        original = group["original"]
        if original is None:
            continue
        variants = [v for v in group["variants"] if v["gold"] == original["gold"]]
        if variants:
            families.append({"original": original, "variants": variants})
    return families


def auroc(scores: list[float], positive: list[bool]) -> float:
    """Probability that a correct original scores above a wrong one (ties count half)."""
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def stats(families: list[dict]) -> dict[str, float]:
    n = len(families)
    rows = [[f["original"]["correct"]] + [v["correct"] for v in f["variants"]] for f in families]
    all_correct = sum(all(r) for r in rows)
    all_wrong = sum(not any(r) for r in rows)
    kept = n - all_correct
    agreement = [sum(v["pred"] == f["original"]["pred"] for v in f["variants"]) / len(f["variants"]) for f in families]
    correct = [f["original"]["correct"] for f in families]
    agree = [c for a, c in zip(agreement, correct) if a == 1]
    disagree = [c for a, c in zip(agreement, correct) if a < 1]
    share = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "families": n,
        "all correct": all_correct / n,
        "mixed": (n - all_correct - all_wrong) / n,
        "all wrong": all_wrong / n,
        "kept": kept / n,
        "wrong among kept": all_wrong / kept if kept else float("nan"),
        "acc if agree": share(agree),
        "acc if disagree": share(disagree),
        "AUROC": auroc(agreement, correct),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=SAMPLES.jsonl")
    args = parser.parse_args()
    print("| run | " + " | ".join(METRICS) + " |")
    print("|-|" + "-|" * len(METRICS))
    for item in args.run:
        name, path = item.split("=", 1)
        s = stats(read_families(Path(path)))
        print(f"| {name} | {s['families']} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS[1:]) + " |")


if __name__ == "__main__":
    main()
