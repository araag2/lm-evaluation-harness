"""Detailed scores of augmentation tasks, from the samples a run logged.

Reads every `samples_<task>_<time>.jsonl` written by a task whose `process_results` is
`lm_eval.opencteval.augmentation.sliced_results` (they carry the slices of each row and its
gold and predicted label), keeps the latest run per task and model, and writes
`<output_dir>/<output_name>.md` and `.csv`.

    python scripts/augmentation_breakdown.py --input_folders outputs/AugExp --output_dir outputs/AugExp_Tables
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from lm_eval.opencteval.augmentation import (
    SLICE_SEPARATOR,
    acc_drop,
    consistency,
    flips,
    macro_f1,
    matched_pairs,
)


SAMPLES_NAME = re.compile(r"samples_(?P<task>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}T[\d.-]+)\.jsonl")
SLICE_PREFIX = f"acc{SLICE_SEPARATOR}"
SMALL_CELL = 10  # a cell with fewer rows is marked: its accuracy is not reliable
SLICE_ORDER = ("all", "original", "augmented", "effect", "side", "type", "label", "truncated")
UNPARSED = "unparsed"


@dataclass(frozen=True)
class Row:
    """One scored row: (source_id, is_original, gold, pred) and the slices it belongs to."""

    item: tuple
    slices: frozenset[str]

    @property
    def gold(self) -> int:
        return self.item[2]

    @property
    def pred(self) -> int | None:
        return self.item[3]


@dataclass(frozen=True)
class Run:
    task: str
    model: str
    path: Path
    rows: list[Row]
    labels: dict[int, str]

    def in_slice(self, name: str) -> list[Row]:
        return [row for row in self.rows if name in row.slices]

    def slice_names(self) -> list[str]:
        names = {name for row in self.rows for name in row.slices}
        return sorted(names, key=lambda n: (SLICE_ORDER.index(n.split("=")[0]), n))

    def label(self, index: int | None) -> str:
        return UNPARSED if index is None else self.labels.get(index, str(index))


def read_run(path: Path, task: str, stamp: str) -> Run | None:
    rows, labels = [], {}
    with path.open(encoding="utf8") as handle:
        for line in handle:
            sample = json.loads(line)
            if "consistency" not in sample:
                return None  # not a sliced augmentation task
            names = {key[len(SLICE_PREFIX) :] for key in sample if key.startswith(SLICE_PREFIX)}
            row = Row(tuple(sample["consistency"]), frozenset(names | {"all"}))
            labels[row.gold] = str(sample["target"])
            rows.append(row)
    results = path.with_name(f"results_{stamp}.json")
    model = path.parent.name
    if results.exists():
        model = json.loads(results.read_text(encoding="utf8")).get("model_name") or model
    return Run(task, str(model).split("/")[-1], path, rows, labels) if rows else None


def collect(folders: list[str]) -> list[Run]:
    """The latest run of each task and model found under the folders."""
    latest: dict[tuple[str, str], tuple[str, Run]] = {}
    for folder in folders:
        for path in sorted(Path(folder).rglob("samples_*.jsonl")):
            match = SAMPLES_NAME.fullmatch(path.name)
            run = read_run(path, match["task"], match["stamp"]) if match else None
            if run is None:
                continue
            key = (run.task, run.model)
            if key not in latest or match["stamp"] > latest[key][0]:
                latest[key] = (match["stamp"], run)
    return [run for _, run in sorted(latest.values(), key=lambda p: (p[1].task, p[1].model))]


def accuracy(rows: list[Row]) -> float:
    return sum(row.pred == row.gold for row in rows) / len(rows) if rows else float("nan")


def slice_scores(run: Run, name: str) -> dict:
    """Scores of one slice; its variants are compared with their own originals, wherever those are."""
    rows = run.in_slice(name)
    variants = [row.item for row in rows if not row.item[1]]
    paired = [row.item for row in run.rows if row.item[1]] + variants
    pairs = matched_pairs(paired)
    predicted = Counter(run.label(row.pred) for row in rows)
    scores = {
        "slice": name,
        "n": len(rows),
        "acc": accuracy(rows),
        "f1": macro_f1((row.gold, row.pred) for row in rows),
        "paired_variants": len(pairs),
        "acc_matched_originals": (
            sum(o[3] == o[2] for o, _ in pairs) / len(pairs) if pairs else float("nan")
        ),
        "acc_drop": acc_drop(paired),
        "consistency": consistency(paired),
        **flips(paired),
    }
    for label in [*run.labels.values(), UNPARSED]:
        scores[f"predicted={label}"] = predicted[label] / len(rows)
    return scores


def percent(value: float) -> str:
    return "–" if math.isnan(value) else f"{value * 100:.1f}"


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return "\n".join(lines + ["| " + " | ".join(cells) + " |" for cells in body]) + "\n"


def slice_table(run: Run, scores: list[dict]) -> str:
    labels = [label for _, label in sorted(run.labels.items())]
    if any(row.pred is None for row in run.rows):
        labels.append(UNPARSED)
    header = ["Slice", "n", "Acc", "Macro-F1", *[f"Pred. {label} %" for label in labels]]
    header += ["Paired", "Acc of their originals", "Drop", "Consistency", "Right→wrong", "Wrong→right"]
    body = []
    for s in scores:
        one_class = s["slice"].startswith("label=")  # macro F1 over one gold class says nothing
        f1 = "–" if one_class else percent(s["f1"])
        cells = [s["slice"], str(s["n"]), percent(s["acc"]), f1]
        cells += [percent(s[f"predicted={label}"]) for label in labels]
        if s["paired_variants"]:
            cells += [str(s["paired_variants"]), percent(s["acc_matched_originals"])]
            cells += [percent(s["acc_drop"]), percent(s["consistency"])]
            cells += [str(s["correct_to_wrong"]), str(s["wrong_to_correct"])]
        else:
            cells += ["–"] * 6
        body.append(cells)
    return table(header, body)


def type_by_label(run: Run) -> str:
    labels = sorted(run.labels.items())
    kinds = ["original"] + [n for n in run.slice_names() if n.startswith("type=")]
    body = []
    for kind in kinds:
        cells = [kind]
        for index, _ in labels:
            rows = [row for row in run.in_slice(kind) if row.gold == index]
            mark = "*" if 0 < len(rows) < SMALL_CELL else ""
            cells.append(f"{percent(accuracy(rows))} ({len(rows)}){mark}" if rows else "–")
        body.append(cells)
    return table(["Type \\ gold label", *[label for _, label in labels]], body)


def confusion(run: Run, name: str) -> str:
    rows = run.in_slice(name)
    columns = [*sorted(run.labels)] + ([None] if any(row.pred is None for row in rows) else [])
    counts = Counter((row.gold, row.pred) for row in rows)
    body = [
        [run.label(gold), *[str(counts[(gold, pred)]) for pred in columns]]
        for gold in sorted(run.labels)
    ]
    return table(["Gold \\ predicted", *[run.label(c) for c in columns]], body)


def model_comparison(runs: list[Run]) -> str:
    names = sorted({n for run in runs for n in run.slice_names()},
                   key=lambda n: (SLICE_ORDER.index(n.split("=")[0]), n))
    body = []
    for name in names:
        values = [accuracy(run.in_slice(name)) for run in runs]
        best = max((v for v in values if not math.isnan(v)), default=None)
        size = max(len(run.in_slice(name)) for run in runs)
        cells = [f"**{percent(v)}**" if v == best and len(runs) > 1 else percent(v) for v in values]
        body.append([name, str(size), *cells])
    return table(["Slice", "n", *[run.model for run in runs]], body)


def render(runs: list[Run], scores: dict[Path, list[dict]]) -> str:
    out = [
        "# Augmentation breakdown\n",
        (
            "All scores are percentages. `Paired` is the number of variants in the slice whose "
            "original was scored and has the same gold label; `Drop` is the accuracy of those "
            "originals minus the accuracy of the variants (positive = the variants are harder); "
            "`Consistency` is the share of them predicted like their original. "
            f"`*` marks a cell with fewer than {SMALL_CELL} rows.\n"
        ),
    ]
    by_task = defaultdict(list)
    for run in runs:
        by_task[run.task].append(run)
    for task, task_runs in by_task.items():
        out.append(f"## {task}\n")
        if len(task_runs) > 1:
            out += ["### Accuracy by slice and model\n", model_comparison(task_runs)]
        for run in task_runs:
            out += [f"### {run.model}\n", f"Samples: `{run.path}`\n", slice_table(run, scores[run.path])]
            out += ["Accuracy by type and gold label, with the number of rows:\n", type_by_label(run)]
            for name in ("original", "augmented"):
                if run.in_slice(name):
                    out += [f"Confusion matrix, {name} rows:\n", confusion(run, name)]
    return "\n".join(out)


def write_csv(path: Path, runs: list[Run], scores: dict[Path, list[dict]]) -> None:
    records = [
        {"task": run.task, "model": run.model, **s} for run in runs for s in scores[run.path]
    ]
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input_folders", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="augmentation_breakdown")
    args = parser.parse_args()

    runs = collect(args.input_folders)
    if not runs:
        raise SystemExit("No samples of a sliced augmentation task found in the input folders")
    scores = {run.path: [slice_scores(run, name) for name in run.slice_names()] for run in runs}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.output_name}.md").write_text(render(runs, scores), encoding="utf8")
    write_csv(output / f"{args.output_name}.csv", runs, scores)
    print(f"{len(runs)} run(s) -> {output / args.output_name}.md, .csv")


if __name__ == "__main__":
    main()
