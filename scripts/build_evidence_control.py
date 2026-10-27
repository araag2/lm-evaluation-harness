"""Build the matched control for the NLI4CT evidence ablation: remove as many lines as the evidence has, at random
from the non-evidence lines, keeping the evidence.

For each of the 205 statements with an `evidence_removed` view (data/nli4ct_evidence_ablation/test.jsonl), the
`random_removed` view drops, in each section, a seeded random choice of non-evidence content lines, as many as
that section's evidence content lines; headers and evidence stay, so the label is the gold label. A model that
follows the evidence keeps its prediction; a model that learned "lines removed -> Contradiction" flips here as
often as under `evidence_removed`. Sections without enough non-evidence lines remove all but one of them
(`matched: false`). Score with scripts/evidence_control_scores.py.

    python scripts/build_evidence_control.py   # writes data/nli4ct_evidence_ablation/control.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from lm_eval.opencteval.evidence import is_header

from build_evidence_ablation import FIELDS, NLI4CT_DATA


VIEW = "random_removed"


def random_removed(lines: list[str], evidence: list[int], rng: random.Random) -> tuple[list[str], bool]:
    """`lines` without k random non-evidence content lines, k = the evidence content lines in this section."""
    chosen = set(evidence)
    k = sum(1 for i in chosen if i < len(lines) and not is_header(lines[i]))
    others = [i for i, line in enumerate(lines) if i not in chosen and not is_header(line)]
    matched = len(others) > k  # keep at least one non-evidence line, else this equals evidence_only
    drop = set(rng.sample(others, min(k, max(len(others) - 1, 0))))
    return [line for i, line in enumerate(lines) if i not in drop], matched


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ablation", type=Path, default=Path("data/nli4ct_evidence_ablation/test.jsonl"))
    parser.add_argument("--statements", type=Path, default=NLI4CT_DATA / "_2023/source/test.jsonl")
    parser.add_argument("--corpus", type=Path, default=NLI4CT_DATA / "corpus/NLI4CT_corpus.json")
    parser.add_argument("--output", type=Path, default=Path("data/nli4ct_evidence_ablation/control.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ablated = {
        row["source_id"]
        for row in map(json.loads, args.ablation.open())
        if row["view"] == "evidence_removed"
    }
    corpus = json.loads(args.corpus.read_text())
    statements = [s for s in map(json.loads, args.statements.open()) if s["id"] in ablated]
    rows, counts = [], Counter()
    for statement in statements:
        rng = random.Random(f"{args.seed}:{statement['id']}")
        trials = [statement["Primary_id"]]
        indices = [statement["Primary_evidence_index"]]
        if statement["Type"] == "Comparison":
            trials.append(statement["Secondary_id"])
            indices.append(statement["Secondary_evidence_index"])
        views = [
            random_removed(list(corpus[t][statement["Section_id"]]), idx, rng)
            for t, idx in zip(trials, indices, strict=True)
        ]
        matched = all(m for _, m in views)
        counts[(statement["Label"], matched)] += 1
        rows.append({
            "id": f"{statement['id']}:{VIEW}",
            "source_id": statement["id"],
            "view": VIEW,
            "matched": matched,
            **{key: statement.get(key) for key in FIELDS},
            "Label": statement["Label"],
            "Gold_Label": statement["Label"],
            "Primary_Section": views[0][0],
            "Secondary_Section": views[1][0] if len(views) > 1 else None,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} statements -> {args.output}")
    for (label, matched), n in sorted(counts.items()):
        print(f"  {label:13} matched={matched} {n}")


if __name__ == "__main__":
    main()
