"""Why do criteria views hurt on TREC CT? Diagnostics from logged TREC_CT_2023 samples (plan 2026-10-29).

Compares a run trained with criteria views (`--treated`, ect-ce) with its control (`--control`, aug-ce):
- transitions: per gold grade, how the predicted grade moves from control to treated;
- prior shift: the control's scores with one constant added to the "possibly" score, chosen so that it predicts
  "possibly" as often as the treated run. If this reproduces the treated run's predictions and macro-F1, the views
  acted as a class-prior shift; if not, the change depends on the pair's content;
- doses: on gold "definitely" pairs, P(treated says possibly) - P(control says possibly) by quintile of (a) the
  patient's overlap with the trial's exclusion criteria (largest share of an exclusion criterion's content words that
  the patient note also contains) and (b) the number of exclusion criteria. Lexical over-application predicts a rise
  with (a); a per-criterion false-alarm rate predicts a rise with (b).

    python scripts/trec_view_diagnostics.py --control aug-ce=SAMPLES.jsonl --treated ect-ce=SAMPLES.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from lm_eval.opencteval.augmentation import macro_f1


LABELS = ["not relevant", "possibly relevant", "definitely relevant"]
SHORT = ["not", "possibly", "definitely"]
POSSIBLY, DEFINITELY = 1, 2
QUANTILES = 5
EXCLUSION = re.compile(r"^\s*exclusion\s+criteri", re.IGNORECASE | re.MULTILINE)
CRITERION_SPLIT = re.compile(r"\n\s*(?:[-*•]|\d+[.)])\s+")
WORD = re.compile(r"[a-z][a-z0-9-]{3,}")
STOPWORDS = frozenset(
    ["with", "have", "been", "within", "prior", "than", "that", "this", "from", "history", "patients", "patient", "years", "other", "must", "will", "known", "current", "including", "disease", "treatment", "study", "subjects", "subject", "within", "months", "weeks", "days", "any"]
)


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOPWORDS}


def exclusion_overlap(patient: str, eligibility: str) -> float:
    """Largest share of one exclusion criterion's content words that also occur in the patient note (0 if none)."""
    found = EXCLUSION.search(eligibility.replace("\r\n", "\n").replace("\r", " "))
    if found is None:
        return 0.0
    section = eligibility.replace("\r\n", "\n").replace("\r", " ")[found.end():]
    note = content_words(patient)
    shares = [
        len(words & note) / len(words)
        for words in (content_words(c) for c in CRITERION_SPLIT.split(section))
        if len(words) >= 2
    ]
    return max(shares, default=0.0)


def exclusion_count(eligibility: str) -> int:
    """Number of list items after the exclusion heading (0 if the trial has none)."""
    text = eligibility.replace("\r\n", "\n").replace("\r", " ")
    found = EXCLUSION.search(text)
    return len(CRITERION_SPLIT.split(text[found.end():])) if found else 0


def read(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        sample = json.loads(line)
        doc = sample["doc"]
        scores = [float(r[0]) for r in sample["filtered_resps"]]
        rows[str(doc["id"])] = {
            "gold": LABELS.index(doc["Label"]),
            "scores": scores,
            "pred": scores.index(max(scores)),
            "patient": doc["Patient"],
            "eligibility": doc.get("Eligibility") or "",
        }
    return rows


def predict(scores: list[float], bias: float) -> int:
    shifted = [s + (bias if i == POSSIBLY else 0.0) for i, s in enumerate(scores)]
    return shifted.index(max(shifted))


def matched_bias(control: list[dict], target_rate: float) -> float:
    """Bisection on the "possibly" offset that makes the control predict "possibly" at `target_rate`."""
    low, high = -20.0, 20.0
    for _ in range(60):
        middle = (low + high) / 2
        rate = sum(predict(r["scores"], middle) == POSSIBLY for r in control) / len(control)
        low, high = (middle, high) if rate < target_rate else (low, middle)
    return (low + high) / 2


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "-|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", required=True, help="NAME=SAMPLES.jsonl")
    parser.add_argument("--treated", required=True, help="NAME=SAMPLES.jsonl")
    args = parser.parse_args()
    (cname, cpath), (tname, tpath) = (item.split("=", 1) for item in (args.control, args.treated))
    control, treated = read(Path(cpath)), read(Path(tpath))
    keys = sorted(set(control) & set(treated))
    c, t = [control[k] for k in keys], [treated[k] for k in keys]
    print(f"{len(keys)} pairs\n")

    print(f"## Transitions {cname} -> {tname} (rows: gold grade; share of that grade's pairs)\n")
    rows = []
    for gold in range(3):
        idx = [i for i, r in enumerate(c) if r["gold"] == gold]
        moves = {}
        for a in range(3):
            for b in range(3):
                if a != b:
                    moves[f"{SHORT[a]}→{SHORT[b]}"] = sum(c[i]["pred"] == a and t[i]["pred"] == b for i in idx)
        top = sorted(moves.items(), key=lambda kv: -kv[1])[:3]
        changed = sum(c[i]["pred"] != t[i]["pred"] for i in idx)
        rows.append([SHORT[gold], len(idx), f"{changed / len(idx):.3f}"] + [f"{k} {v / len(idx):.3f}" for k, v in top])
    print(table(["gold", "n", "changed", "top move", "2nd", "3rd"], rows))

    rate = sum(r["pred"] == POSSIBLY for r in t) / len(t)
    bias = matched_bias(c, rate)
    shifted = [predict(r["scores"], bias) for r in c]
    agree_shift = sum(s == r["pred"] for s, r in zip(shifted, t, strict=True)) / len(t)
    agree_control = sum(r["pred"] == s["pred"] for r, s in zip(c, t, strict=True)) / len(t)
    demoted = [i for i in range(len(t)) if c[i]["pred"] == DEFINITELY and t[i]["pred"] == POSSIBLY]
    reproduced = sum(shifted[i] == POSSIBLY for i in demoted)
    print(f"\n## Prior shift: {cname} + {bias:+.3f} on 'possibly' predicts it at {rate:.3f}, as {tname} does\n")
    print(table(
        ["", "macro-F1", "agrees with " + tname],
        [
            [cname, f"{macro_f1([(r['gold'], r['pred']) for r in c]):.3f}", f"{agree_control:.3f}"],
            [f"{cname} + shift", f"{macro_f1([(r['gold'], s) for r, s in zip(c, shifted, strict=True)]):.3f}", f"{agree_shift:.3f}"],
            [tname, f"{macro_f1([(r['gold'], r['pred']) for r in t]):.3f}", "1.000"],
        ],
    ))
    print(f"\n{tname}'s definitely→possibly moves reproduced by the shift: {reproduced}/{len(demoted)}")

    gold_def = [i for i, r in enumerate(c) if r["gold"] == DEFINITELY]
    doses = {
        "exclusion overlap": {i: exclusion_overlap(c[i]["patient"], c[i]["eligibility"]) for i in gold_def},
        "exclusion criteria": {i: exclusion_count(c[i]["eligibility"]) for i in gold_def},
    }
    for name, value in doses.items():
        ranked = sorted(gold_def, key=lambda i: value[i])
        print(f"\n## Dose: {name} (gold definitely, {len(gold_def)} pairs, quintiles)\n")
        rows = []
        for q in range(QUANTILES):
            part = ranked[q * len(ranked) // QUANTILES:(q + 1) * len(ranked) // QUANTILES]
            pc = sum(c[i]["pred"] == POSSIBLY for i in part) / len(part)
            pt = sum(t[i]["pred"] == POSSIBLY for i in part) / len(part)
            ps = sum(shifted[i] == POSSIBLY for i in part) / len(part)
            rows.append([q + 1, f"{value[part[0]]:.2f}–{value[part[-1]]:.2f}", len(part),
                         f"{pc:.3f}", f"{pt:.3f}", f"{pt - pc:+.3f}", f"{ps - pc:+.3f}"])
        print(table(["quintile", name, "n", f"P(possibly) {cname}", tname, "diff", "diff if prior shift"], rows))

if __name__ == "__main__":
    main()
