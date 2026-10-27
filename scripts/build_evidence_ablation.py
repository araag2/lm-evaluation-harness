"""Build the NLI4CT evidence-ablation test set: each SemEval-2024 base statement with its full CTR section(s),
and, where the gold evidence is partial, with only the evidence and with the evidence removed.

The 500 SemEval-2024 base statements are the 2023 test statements, which carry gold evidence indices.
Labels follow `lm_eval.opencteval.evidence`: `evidence_only` keeps the gold label, `evidence_removed` is
Contradiction (unsupported). Output rows have the 2024 test schema plus `source_id` and `view`.

    python scripts/build_evidence_ablation.py   # writes data/nli4ct_evidence_ablation/test.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from lm_eval.opencteval.evidence import FULL, evidence_views, view_label


NLI4CT_DATA = (
    Path(__file__).resolve().parents[2] / "OpenCTEval-Data-Pipeline/data/SemEval_NLI4CT"
)
FIELDS = ("Type", "Section_id", "Primary_id", "Secondary_id", "Statement")


def rows_of(statement: dict, corpus: dict) -> list[dict]:
    section = statement["Section_id"]
    trials = [statement["Primary_id"]]
    indices = [statement["Primary_evidence_index"]]
    if statement["Type"] == "Comparison":
        trials.append(statement["Secondary_id"])
        indices.append(statement["Secondary_evidence_index"])
    sections = [corpus[trial][section] for trial in trials]
    views = {FULL: [list(lines) for lines in sections]}
    views.update(evidence_views(sections, indices) or {})
    base = {key: statement.get(key) for key in FIELDS}
    return [
        {
            "id": f"{statement['id']}:{view}",
            "source_id": statement["id"],
            "view": view,
            **base,
            "Label": view_label(view, statement["Label"]),
            "Gold_Label": statement["Label"],
            "Primary_Section": lines[0],
            "Secondary_Section": lines[1] if len(lines) > 1 else None,
        }
        for view, lines in views.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--statements", type=Path, default=NLI4CT_DATA / "_2023/source/test.jsonl"
    )
    parser.add_argument(
        "--corpus", type=Path, default=NLI4CT_DATA / "corpus/NLI4CT_corpus.json"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/nli4ct_evidence_ablation/test.jsonl")
    )
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text())
    statements = [json.loads(line) for line in args.statements.open()]
    rows = [row for statement in statements for row in rows_of(statement, corpus)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter((row["view"], row["Gold_Label"]) for row in rows)
    print(f"{len(statements)} statements -> {len(rows)} rows in {args.output}")
    for key, n in sorted(counts.items()):
        print(f"  {key[0]:16} {key[1]:13} {n}")


if __name__ == "__main__":
    main()
