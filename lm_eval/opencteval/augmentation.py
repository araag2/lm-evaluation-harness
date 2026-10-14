"""Scores of an augmented dataset by slice: originals, variants, augmentation type, label.

A row belongs to several slices. `sliced_results` is the body of a task's
`process_results`: it emits `acc@<slice>` only for rows in that slice, so the harness's mean
aggregation gives the accuracy of each slice. `scripts/augmentation_breakdown.py` uses the same
functions on logged samples, so both report the same rows in the same slices.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence


ORIGINAL_MODE = "original"
SLICE_SEPARATOR = "@"


def mode(doc: dict) -> str:
    return (doc.get("augmentation") or {}).get("mode", ORIGINAL_MODE)


def is_original(doc: dict) -> bool:
    return mode(doc) == ORIGINAL_MODE


def slices(doc: dict, label: str, extra: Iterable[str] = ()) -> list[str]:
    """Names of the slices this row belongs to; `extra` adds task-specific ones."""
    names = ["all", f"label={label}"]
    if is_original(doc):
        names.append("original")
    else:
        metadata = doc["augmentation"]
        names += ["augmented", f"type={metadata['mode']}"]
        names += [
            f"{key}={metadata[key]}" for key in ("side", "effect") if metadata.get(key)
        ]
    return names + list(extra)


def prediction(results: Sequence, choices: Sequence[str]) -> int | None:
    """Index of the predicted choice, or None when a generated answer names none.

    `results` is one (log-likelihood, is_greedy) pair per choice, or one generated string.
    Logged samples store the log-likelihoods as text.
    """
    if len(results) == 1 and isinstance(results[0], str):
        answer = results[0].strip().lower()
        matches = [i for i, choice in enumerate(choices) if choice.lower() == answer]
        return matches[0] if matches else None
    scores = [float(result[0]) for result in results]
    return scores.index(max(scores))


def sliced_results(
    doc: dict,
    results: Sequence,
    choices: Sequence[str],
    label_key: str = "Label",
    extra: Iterable[str] = (),
) -> dict:
    label = doc[label_key]
    gold = list(choices).index(label)
    pred = prediction(results, choices)
    correct = float(pred == gold)
    pair = (gold, pred)
    scores = {
        "acc": correct,
        "f1": pair,
        "consistency": (str(doc["source_id"]), is_original(doc), gold, pred),
    }
    scores["acc_drop"] = scores["consistency"]
    for name in slices(doc, label, extra):
        if name == "all":
            continue
        scores[f"acc{SLICE_SEPARATOR}{name}"] = correct
        if name in ("original", "augmented"):
            scores[f"f1{SLICE_SEPARATOR}{name}"] = pair
    if len(results) == 1 and isinstance(results[0], str):
        scores["unparsed"] = float(pred is None)
    return scores


def macro_f1(pairs: Iterable[tuple[int, int | None]]) -> float:
    """Macro F1 over the gold classes present; an unparsed answer is a miss for its class."""
    pairs = list(pairs)
    classes = sorted({gold for gold, _ in pairs})
    if not classes:
        return float("nan")
    total = 0.0
    for cls in classes:
        hits = sum(gold == cls and pred == cls for gold, pred in pairs)
        predicted = sum(pred == cls for _, pred in pairs)
        actual = sum(gold == cls for gold, _ in pairs)
        total += 2 * hits / (predicted + actual)
    return total / len(classes)


def matched_pairs(items: Iterable[tuple]) -> list[tuple[tuple, tuple]]:
    """(original, variant) for each variant whose original is present with the same gold label.

    A variant with another label than its original (a derived label) is not expected to get the
    same prediction, so it is not paired.
    """
    items = list(items)
    originals = {item[0]: item for item in items if item[1]}
    pairs = []
    for item in items:
        original = originals.get(item[0])
        if not item[1] and original is not None and original[2] == item[2]:
            pairs.append((original, item))
    return pairs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def consistency(items: Iterable[tuple]) -> float:
    """Share of variants predicted like their original; two unparsed answers do not agree."""
    return _mean(
        [float(o[3] is not None and o[3] == v[3]) for o, v in matched_pairs(items)]
    )


def acc_drop(items: Iterable[tuple]) -> float:
    """Accuracy on the variants' own originals minus accuracy on the variants (positive = worse)."""
    return _mean(
        [float(o[3] == o[2]) - float(v[3] == v[2]) for o, v in matched_pairs(items)]
    )


def flips(items: Iterable[tuple]) -> dict[str, int]:
    """How many variants went from a correct original to wrong, and from wrong to correct."""
    counts = defaultdict(int)
    for original, variant in matched_pairs(items):
        before, after = original[3] == original[2], variant[3] == variant[2]
        if before and not after:
            counts["correct_to_wrong"] += 1
        elif after and not before:
            counts["wrong_to_correct"] += 1
    return {key: counts[key] for key in ("correct_to_wrong", "wrong_to_correct")}
