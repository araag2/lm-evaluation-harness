"""Build the NLI4CT AugExp v3 test set: the v2 test part plus contrary and numeric minimal-pair variants of its statements.

The held-out recipe-family protocol (OpenCTEval `_Reports/NLI4CT_Training/HYPOTHESES.md`, protocol v2) needs
test rows for every family. v2 has no contrary (F4) or numeric minimal pairs (F3). OpenCTEval generated both for
all 2024 train statements (`nli4ct_contrary`, `nli4ct_numeric`, audited), so this adds the ones whose source is
in the v2 test part. `augmentation` keeps only what scoring reads (`mode`, `effect`).

    python scripts/build_augexp_v3.py   # writes data/nli4ct_augexp_v3/test.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


AUG = Path(__file__).resolve().parents[2] / "OpenCTEval-Data-Pipeline/outputs/augmentation/NLI4CT_2024"


def slim(row: dict, columns: list[str]) -> dict:
    out = {c: row.get(c) for c in columns}
    meta = row.get("augmentation") or {}
    out["augmentation"] = {"mode": meta.get("mode", "original"), "effect": meta.get("effect", "original")}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/nli4ct_augexp_v3/test.jsonl"))
    args = parser.parse_args()

    v2 = load_dataset("araag2/SemEval_NLI4CT_Augmented", "2024_augmented_v2", split="test")
    columns = [c for c in v2.column_names if c != "augmentation"]
    rows = [slim(r, columns) for r in v2]
    test_ids = {r["source_id"] for r in rows}
    for experiment in ("nli4ct_contrary", "nli4ct_numeric"):
        for line in (AUG / experiment / "train.jsonl").open():
            r = json.loads(line)
            if r["source_id"] in test_ids and r["augmentation"]["mode"] != "original":
                rows.append(slim(r, columns))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(len(rows), "rows;", dict(Counter(r["augmentation"]["mode"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
