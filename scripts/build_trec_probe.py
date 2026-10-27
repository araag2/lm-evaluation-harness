"""Build the TREC CT 2023 necessity probe from a criteria-augmentation export (no Hub upload).

The export is `trec_ct_train_criteria.yaml` run on the 300 seeded 2023 test pairs (experiment probe_2023_300_v8, variants 8). For
each eligible ("definitely relevant") pair with a view, the probe keeps:
- `original`: the pair as published;
- `flip_criterion` views: one criterion moved to the opposite list, so that the judge finds the patient now fails it.
  TREC grades such a patient "excluded", i.e. `possibly relevant` (`label_evidence.graded_label`). Views the judge
  could not settle (null label) are left out;
- `subset_criteria` views: half of each criteria list kept; still `definitely relevant` (derived by rule). They also
  edit the criteria, so they are the control: a model that reacts to any criteria edit moves on them too.
Scored by scripts/trec_probe_scores.py.

    python scripts/build_trec_probe.py   # writes data/trec_ct_probe/test.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


PIPELINE = Path(__file__).resolve().parents[2] / "OpenCTEval-Data-Pipeline"
EXPORT = PIPELINE / "outputs/augmentation/trec_ct/probe_2023_300_v8/2023_processed/test.jsonl"
ELIGIBLE = "definitely relevant"
VIEWS = {"original", "flip_criterion", "subset_criteria"}


def probe_rows(rows: list[dict]) -> list[dict]:
    originals = {r["source_id"]: r for r in rows if r["augmentation"]["mode"] == "original"}
    out = []
    for row in rows:
        view = row["augmentation"]["mode"]
        if view not in VIEWS - {"original"} or originals[row["source_id"]]["Label"] != ELIGIBLE:
            continue
        graded = (row["augmentation"].get("label_evidence") or {}).get("graded_label")
        label = graded if graded is not None else row["Label"]
        if label is None:
            continue
        out.append({**row, "view": view, "Label": label})
    families = {r["source_id"] for r in out}
    out += [{**originals[s], "view": "original"} for s in sorted(families)]
    for row in out:
        row.pop("augmentation", None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--output", type=Path, default=Path("data/trec_ct_probe/test.jsonl"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.export.open()]
    probe = probe_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for row in probe:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter((r["view"], r["Label"]) for r in probe)
    print(f"{len(probe)} rows, {len({r['source_id'] for r in probe})} eligible pairs, "
          f"{len({r['query_id'] for r in probe})} patients -> {args.output}")
    for (view, label), n in sorted(counts.items()):
        print(f"  {view:16} {label:20} {n}")


if __name__ == "__main__":
    main()
