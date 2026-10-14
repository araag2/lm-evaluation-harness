r"""Collect lm-eval result files into readable tables.

Reads every `results_*.json` (single-turn runs) and `Summary_*.json` (reasoning modes) under the
input folders, keeps one run per task, mode and model, and writes:

  <name>.csv   one row per task, mode, model, metric and slice (the data behind every table)
  <name>.md    overview per mode, details per dataset, scores by slice
  <name>_sources.md   the file of each score, and the runs not shown
  <name>.tex   the same tables for LaTeX (needs booktabs and adjustbox)
  charts/, heatmaps/

    python scripts/extract_run-results.py --input_folders outputs/exp1 outputs/exp2 \
        --output_dir outputs/tables --output_name my_benchmark
"""

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")

# Task names end in their inference mode; the longest suffix is tried first.
MODE_SUFFIXES = ("Para-0-shot", "CoT_SC", "0-shot", "CoT", "SC")
MODE_ORDER = ["0-shot", "Para-0-shot", "CoT", "SC", "CoT_SC", "multi-turn_CoT",
              "multi-turn_CoT-SC", "multi-turn_CoT-MBR", "cross-consistency", "self-refine_CoT"]
MODEL_ORDER = ["Qwen", "deepseek", "DeepSeek", "Mistral", "Llama", "Gemma", "gemma", "Fleming",
               "Panacea", "II-Medical", "MedReason"]
# The metric shown for a task in the overview: the first of these it has, else its first metric.
PRIMARY_METRICS = ("acc", "f1", "span_f1", "rouge_l", "partial_match", "nDCG@10")
NOT_METRICS = {"name", "alias", "sample_len", "sample_count", "samples", "docs",
               "reasoning_outputs", "reasoning_chains"}
UNSCALED_WORDS = ("error", "loss", "perplexity")  # not on a 0-1 scale, and lower is better
SLICE_SEPARATOR = "@"
SLICE_PREFIXES = ("original", "augmented", "type=", "side=", "effect=", "label=", "truncated=")
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T[\d.-]*\d)")
LIMITED_MARK = "†"
MISSING = "–"
LARGE_VALUE = 1e4  # an error metric this large is written with three significant digits


@dataclass(frozen=True)
class Score:
    metric: str
    slice: str
    value: float
    stderr: float | None
    higher_is_better: bool


@dataclass(frozen=True)
class Run:
    """The scores of one task for one model in one file."""

    dataset: str
    mode: str
    model: str
    n: int
    limited: bool
    date: str
    path: str
    scores: tuple[Score, ...]
    family: str = ""
    children: tuple[str, ...] = field(default=(), compare=False)  # datasets of a group's subtasks

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset, self.mode, self.model)

    @property
    def subtask(self) -> str:
        if self.dataset.startswith(self.family + "_"):
            return self.dataset[len(self.family) + 1 :]
        return "all" if self.children else ""


# ── collect ──────────────────────────────────────────────────────────────────

def split_task(task: str) -> tuple[str, str]:
    """`TREC_CT_2021_CoT` -> (`TREC_CT_2021`, `CoT`)."""
    for suffix in MODE_SUFFIXES:
        if task.endswith("_" + suffix):
            return task[: -len(suffix) - 1], suffix
    return task, "single-turn"


def split_metric(name: str) -> tuple[str, str]:
    """`acc@type=permute` -> (`acc`, `type=permute`); `P@10` stays a metric."""
    metric, separator, rest = name.partition(SLICE_SEPARATOR)
    if separator and rest.startswith(SLICE_PREFIXES):
        return metric, rest
    return name, ""


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def lower_is_better(metric: str) -> bool:
    return any(word in metric.lower() for word in UNSCALED_WORDS)


def read_scores(result: dict, directions: dict) -> tuple[Score, ...]:
    """The numeric metrics of one task; `acc,none` and `acc` are both the metric `acc`."""
    scores, seen = [], set()
    for key, value in result.items():
        name, _, filter_name = key.partition(",")
        if name in NOT_METRICS or name.endswith("_stderr") or not is_number(value) or name in seen:
            continue
        seen.add(name)
        stderr = result.get(f"{name}_stderr,{filter_name}" if filter_name else f"{name}_stderr")
        metric, slice_name = split_metric(name)
        higher = directions.get(name)
        if higher is None:
            higher = not lower_is_better(metric)
        scores.append(Score(metric, slice_name, float(value), stderr if is_number(stderr) else None, higher))
    return tuple(scores)


def file_date(path: str) -> str:
    match = STAMP.search(Path(path).name)
    return match.group(1) if match else ""


def model_name(*candidates) -> str:
    """The first usable name, without its organisation: `unsloth/Qwen3-8B` -> `Qwen3-8B`."""
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("pretrained")
        if isinstance(candidate, str):
            match = re.search(r"pretrained=([^,]+)", candidate)
            candidate = match.group(1) if match else candidate
            if candidate and candidate != "N/A" and "=" not in candidate:
                return candidate.rstrip("/").split("/")[-1]
    return "unknown model"


def read_results_file(path: str) -> list[Run]:
    data = json.loads(Path(path).read_text(encoding="utf8"))
    config = data.get("config", {})
    model = model_name(data.get("model_name"), config.get("model_args"))
    groups = data.get("group_subtasks", {})
    group_of = {child: split_task(group)[0] for group, children in groups.items() for child in children}
    runs = []
    for task, result in data.get("results", {}).items():
        dataset, mode = split_task(task)
        counts = data.get("n-samples", {}).get(task, {})
        n = counts.get("effective", result.get("sample_len", 0))
        limited = config.get("limit") is not None or n < counts.get("original", n)
        runs.append(Run(
            dataset, mode, model, int(n or 0), limited, file_date(path), path,
            read_scores(result, data.get("higher_is_better", {}).get(task, {})),
            family=group_of.get(task, ""),
            children=tuple(split_task(child)[0] for child in groups.get(task, ())),
        ))
    return runs


def read_summary_file(path: str) -> list[Run]:
    """A reasoning-mode summary: flat metrics, or one block per voting method or per task."""
    data = json.loads(Path(path).read_text(encoding="utf8"))
    mode = data.get("mode", "unknown mode")
    family, _, _ = data.get("reasoning_task", "").partition(":")
    dataset = split_task(data.get("reasoning_task", "").replace(":", "_"))[0]
    model = model_name(data.get("reasoning_model"), data.get("config", {}).get("model_args"))
    n = len(data.get("samples") or ())
    results = data.get("results", {})
    blocks = {key: value for key, value in results.items() if isinstance(value, dict)}

    def run(name: str, run_mode: str, result: dict) -> Run:
        return Run(name, run_mode, model, n, False, file_date(path), path,
                   read_scores(result, {}), family=family)

    if not blocks:
        return [run(dataset, mode, results)]
    voting = {"majority", "rrf", "condorcet", "logits", "borda"} & set(blocks)
    if voting:
        return [run(dataset, f"{mode} ({method})", block) for method, block in blocks.items()]
    return [run(split_task(task)[0], mode, block) for task, block in blocks.items()]


def collect(folders: list[str], file_filter: str | None = None) -> list[Run]:
    runs, failed = [], 0
    for folder in folders:
        if not os.path.exists(folder):
            print(f"[WARNING] Folder does not exist: {folder}")
            continue
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                if not name.endswith(".json") or (file_filter and file_filter not in name):
                    continue
                reader = (read_results_file if name.startswith("results_")
                          else read_summary_file if name.startswith("Summary_") else None)
                if reader is None:
                    continue
                try:
                    runs.extend(reader(os.path.join(root, name)))
                except (json.JSONDecodeError, OSError, AttributeError, TypeError) as error:
                    failed += 1
                    print(f"[WARNING] Could not read {os.path.join(root, name)}: {error}")
    print(f"[INFO] {len(runs)} task results read, {failed} files failed")
    return [run for run in runs if run.scores]


def with_families(runs: list[Run]) -> list[Run]:
    """Give each run its family: the group that lists it as a subtask, else the longest known
    family its name starts with (`TrialBench_x` -> `TrialBench`), else the family its file
    names, else itself.
    """
    family_of = {child: run.dataset for run in runs for child in run.children}
    parents = set(family_of.values())
    known = sorted(parents | {run.family for run in runs if run.family}, key=len, reverse=True)
    out = []
    for run in runs:
        family = family_of.get(run.dataset) or next(
            (f for f in known if run.dataset == f or run.dataset.startswith(f + "_")),
            run.family or run.dataset)
        out.append(replace(run, family=family,
                           children=run.children or (("-",) if run.dataset in parents else ())))
    return out


# ── select ───────────────────────────────────────────────────────────────────

def mark_partial_summaries(runs: list[Run]) -> list[Run]:
    """Mark a summary run as limited when another model has more samples for the same task
    and mode: a summary file does not record a sample limit.
    """
    largest = defaultdict(int)
    for run in runs:
        largest[run.dataset, run.mode] = max(largest[run.dataset, run.mode], run.n)
    return [replace(run, limited=run.limited or run.n < largest[run.dataset, run.mode])
            if Path(run.path).name.startswith("Summary_") else run for run in runs]


def select(runs: list[Run], all_runs: bool = False) -> tuple[list[Run], list[tuple[Run, str]]]:
    """One run per task, mode and model: the one on most samples, then the latest.

    Returns the runs to show and the others with the reason. With `all_runs` nothing is
    dropped; the others are shown as separate models named with their date.
    """
    best: dict[tuple, Run] = {}
    for run in runs:
        current = best.get(run.key)
        if current is None or (run.n, run.date) > (current.n, current.date):
            best[run.key] = run
    shown, hidden = [], []
    for run in runs:
        chosen = best[run.key]
        if run is chosen:
            shown.append(run)
        elif all_runs:
            shown.append(replace(run, model=f"{run.model} [{run.date}]"))
        else:
            hidden.append((run, "fewer samples" if run.n < chosen.n else "older run"))
    return shown, hidden


# ── tables ───────────────────────────────────────────────────────────────────

def order_of(name: str, order: list[str]) -> tuple[int, str]:
    position = next((i for i, key in enumerate(order) if key in name), len(order))
    return (position, name)


def slice_order(name: str) -> tuple[int, str]:
    return (next(i for i, prefix in enumerate(SLICE_PREFIXES) if name.startswith(prefix)), name)


def mode_order(mode: str) -> tuple[int, str]:
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER), mode)


@dataclass(frozen=True)
class Row:
    """One line of a table: a label per leading column and the run of each model."""

    labels: tuple[str, ...]
    metric: str
    slice: str
    runs: dict[str, Run]

    def score(self, model: str) -> Score | None:
        run = self.runs.get(model)
        return next((s for s in run.scores if (s.metric, s.slice) == (self.metric, self.slice)),
                    None) if run else None

    def best(self, models: list[str]) -> float | None:
        scores = [s for s in map(self.score, models) if s]
        if len(scores) < 2:
            return None
        values = [s.value for s in scores]
        if len(set(values)) == 1:
            return None
        return max(values) if scores[0].higher_is_better else min(values)

    def n(self) -> int:
        return max((run.n for run in self.runs.values()), default=0)


def number(score: Score) -> str:
    if lower_is_better(score.metric):
        return f"{score.value:.3g}" if abs(score.value) >= LARGE_VALUE else f"{score.value:.3f}"
    return f"{score.value * 100:.2f}"


def cells(row: Row, models: list[str], bold: str, limited_mark: str) -> list[str]:
    best, out = row.best(models), []
    for model in models:
        score = row.score(model)
        if score is None:
            out.append(MISSING)
            continue
        text = number(score)
        if best is not None and score.value == best:
            text = bold.format(text)
        out.append(text + (limited_mark if row.runs[model].limited else ""))
    return out


def primary_metric(runs: list[Run]) -> str:
    metrics = [s.metric for run in runs for s in run.scores if not s.slice]
    return next((m for m in PRIMARY_METRICS if m in metrics), metrics[0])


class Report:
    """The selected runs, arranged for the writers."""

    def __init__(self, runs: list[Run]):
        self.runs = runs
        self.models = sorted({run.model for run in runs}, key=lambda m: order_of(m, MODEL_ORDER))
        self.modes = sorted({run.mode for run in runs}, key=mode_order)
        self.families = sorted({run.family for run in runs})

    def of(self, family: str | None = None, mode: str | None = None) -> dict[str, dict[str, Run]]:
        """Runs of a family and a mode, as dataset -> model -> run."""
        table = defaultdict(dict)
        for run in self.runs:
            if family in (None, run.family) and mode in (None, run.mode):
                table[run.dataset][run.model] = run
        return dict(sorted(table.items(), key=lambda item: self._dataset_order(item[1])))

    @staticmethod
    def _dataset_order(by_model: dict[str, Run]) -> tuple[int, str]:
        run = next(iter(by_model.values()))
        return (0 if run.subtask in ("", "all") else 1, run.dataset)

    def models_of(self, rows: list[Row]) -> list[str]:
        return [model for model in self.models if any(model in row.runs for row in rows)]

    def overview(self, mode: str) -> list[Row]:
        """The primary metric of each task; a group stands for its subtasks."""
        rows = []
        for family in self.families:
            datasets = self.of(family, mode)
            grouped = [d for d, by_model in datasets.items()
                       if next(iter(by_model.values())).subtask == "all"]
            for dataset in grouped or datasets:
                by_model = datasets[dataset]
                metric = primary_metric(list(by_model.values()))
                rows.append(Row((dataset, metric), metric, "", by_model))
        return rows

    def details(self, family: str, mode: str) -> list[Row]:
        """Every metric of every task of a family; empty when the overview already shows it all."""
        rows = []
        for by_model in self.of(family, mode).values():
            metrics = dict.fromkeys(s.metric for run in by_model.values() for s in run.scores if not s.slice)
            subtask = next(iter(by_model.values())).subtask
            rows += [Row((subtask or MISSING, metric), metric, "", by_model) for metric in metrics]
        return rows if len(rows) > 1 else []

    def sliced(self, family: str, mode: str) -> dict[str, list[Row]]:
        """One row per slice and metric for each task that reports slices, by dataset."""
        tables = {}
        for dataset, by_model in self.of(family, mode).items():
            pairs = dict.fromkeys((s.metric, s.slice) for run in by_model.values()
                                  for s in run.scores if s.slice)
            ordered = sorted(pairs, key=lambda pair: (slice_order(pair[1]), pair[0]))
            if ordered:
                tables[dataset] = [Row((name, metric), metric, name, by_model) for metric, name in ordered]
        return tables

    def mean_ranks(self, rows: list[Row], models: list[str]) -> tuple[int, list[str]]:
        """Mean rank of each model over the rows where every listed model has a score, and
        the number of those rows.
        """
        ranks = defaultdict(list)
        for row in rows:
            scores = {model: row.score(model) for model in models}
            if not all(scores.values()):
                continue
            higher = next(iter(scores.values())).higher_is_better
            ordered = sorted({s.value for s in scores.values()}, reverse=higher)
            for model, score in scores.items():
                ranks[model].append(ordered.index(score.value) + 1)
        covered = len(next(iter(ranks.values()), ()))
        return covered, [f"{sum(ranks[m]) / len(ranks[m]):.2f}" if ranks[m] else MISSING for m in models]


# ── writers ──────────────────────────────────────────────────────────────────

def tidy_frame(runs: list[Run]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Family": run.family, "Dataset": run.dataset, "Subtask": run.subtask, "Mode": run.mode,
         "Model": run.model, "Metric": score.metric, "Slice": score.slice, "Value": score.value,
         "Stderr": score.stderr, "N": run.n, "Limited": run.limited,
         "Higher_is_better": score.higher_is_better, "Date": run.date, "Path": run.path}
        for run in runs for score in run.scores
    ])


def used_labels(heads: list[str], rows: list[Row]) -> tuple[list[str], list[tuple[str, ...]]]:
    """The label columns without those that are empty in every row."""
    keep = [i for i in range(len(heads)) if any(row.labels[i] != MISSING for row in rows)]
    return [heads[i] for i in keep], [tuple(row.labels[i] for i in keep) for row in rows]


def markdown_table(header: list[str], body: list[list[str]], numeric_from: int) -> str:
    align = [":--" if i < numeric_from else "--:" for i in range(len(header))]
    lines = [header, align, *body]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines) + "\n\n"


def markdown_rows(report: Report, rows: list[Row], heads: list[str], with_n: bool = True) -> str:
    models = report.models_of(rows)
    heads, labels = used_labels(heads, rows)
    body = [[*label, *([str(row.n())] if with_n else []), *cells(row, models, "**{}**", LIMITED_MARK)]
            for label, row in zip(labels, rows, strict=True)]
    header = [*heads, *(["n"] if with_n else []), *models]
    return markdown_table(header, body, len(heads))


def save_markdown(report: Report, hidden: list[tuple[Run, str]], path: Path) -> None:
    sources = path.with_name(f"{path.stem}_sources.md")
    out = [
        "# Evaluation results\n\n",
        (
            f"{len(report.runs)} task results from {len(report.models)} models. Scores are "
            "percentages, except error metrics, which keep their unit. The best score of a row is "
            f"in bold. `{LIMITED_MARK}` marks a run on a limited number of samples; `n` is the "
            "largest number of samples in the row. The file of each score, and "
            f"{len(hidden)} other runs of the same task, mode and model that are not shown, are "
            f"listed in `{sources.name}`.\n\n"
        ),
        (
            "## Overview\n\nOne metric per task. A group of subtasks is shown as its aggregate; "
            "the subtasks are under Details.\n\n"
        ),
    ]
    for mode in report.modes:
        rows = report.overview(mode)
        models = report.models_of(rows)
        out.append(f"### {mode}\n\n")
        table = markdown_rows(report, rows, ["Task", "Metric"])
        if len(models) > 1:
            covered, ranks = report.mean_ranks(rows, models)
            line = ["**Mean rank**", f"{covered} tasks all models have", "", *ranks]
            table = table.rstrip("\n") + "\n| " + " | ".join(line) + " |\n\n"
        out.append(table)

    out.append("## Details\n\n")
    for family in report.families:
        section = []
        for mode in report.modes:
            rows = report.details(family, mode)
            if rows:
                section += [f"#### {mode}\n\n", markdown_rows(report, rows, ["Subtask", "Metric"])]
            for dataset, slice_rows in report.sliced(family, mode).items():
                section += [f"#### {dataset}, {mode}: by slice\n\n",
                            markdown_rows(report, slice_rows, ["Slice", "Metric"], with_n=False)]
        if section:
            out += [f"### {family}\n\n", *section]

    path.write_text("".join(out), encoding="utf8")
    out = ["# Sources\n\n"]
    body = [[run.dataset, run.mode, run.model, str(run.n), "yes" if run.limited else "", run.date,
             f"`{run.path}`"] for run in sorted(report.runs, key=lambda r: (r.dataset, mode_order(r.mode), r.model))]
    out.append(markdown_table(["Task", "Mode", "Model", "n", "Limited", "Date", "File"], body, 7))
    if hidden:
        out.append("## Runs not shown\n\n")
        body = [[run.dataset, run.mode, run.model, str(run.n), reason, run.date, f"`{run.path}`"]
                for run, reason in sorted(hidden, key=lambda p: (p[0].dataset, p[0].mode, p[0].model, p[0].date))]
        out.append(markdown_table(["Task", "Mode", "Model", "n", "Why", "Date", "File"], body, 7))
    sources.write_text("".join(out), encoding="utf8")
    print(f"    - Saved Markdown to: {path}, {sources.name}")


LATEX_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


HEADER_ANGLE = 70  # model names are written at this angle so a wide table keeps narrow columns


def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIALS.get(char, char) for char in str(text))


def latex_table(report: Report, heads: list[str], by_mode: dict[str, list[Row]], caption: str,
                label: str) -> str:
    """One table: label columns, then a group of model columns per mode."""
    by_mode = {mode: rows for mode, rows in by_mode.items() if rows}
    all_rows = [row for rows in by_mode.values() for row in rows]
    keep = [i for i in range(len(heads)) if any(row.labels[i] != MISSING for row in all_rows)]
    heads = [heads[i] for i in keep]
    models = {mode: report.models_of(rows) for mode, rows in by_mode.items()}
    keys = list(dict.fromkeys((row.labels, row.metric, row.slice) for rows in by_mode.values() for row in rows))
    lookup = {mode: {(row.labels, row.metric, row.slice): row for row in rows} for mode, rows in by_mode.items()}
    width = sum(len(m) for m in models.values())
    lines = [r"\begin{table}[p]", r"\centering", r"\small", rf"\caption{{{latex_escape(caption)}}}",
             rf"\label{{tab:{label}}}", r"\begin{adjustbox}{max width=\textwidth}",
             rf"\begin{{tabular}}{{{'l' * len(heads)}{'r' * width}}}", r"\toprule"]
    if len(by_mode) > 1:
        groups, rules, start = [""] * len(heads), [], len(heads) + 1
        for mode, names in models.items():
            groups.append(rf"\multicolumn{{{len(names)}}}{{c}}{{{latex_escape(mode)}}}")
            rules.append(rf"\cmidrule(lr){{{start}-{start + len(names) - 1}}}")
            start += len(names)
        lines += [" & ".join(groups) + r" \\", " ".join(rules)]
    header = [*heads, *[rf"\rotatebox{{{HEADER_ANGLE}}}{{{latex_escape(name)}}}"
                        for names in models.values() for name in names]]
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    for key in keys:
        line = [latex_escape(key[0][i]) for i in keep]
        for mode, names in models.items():
            row = lookup[mode].get(key)
            line += cells(row, names, r"\textbf{{{}}}", r"$^\dagger$") if row else [MISSING] * len(names)
        lines.append(" & ".join(line).replace(MISSING, "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(lines)


def save_latex(report: Report, path: Path) -> None:
    tables = [
        (
            "% Generated by scripts/extract_run-results.py. Needs \\usepackage{booktabs,adjustbox}.\n"
            "% Scores are percentages (error metrics keep their unit); bold = best of the row within a\n"
            "% mode; a dagger marks a run on a limited number of samples.\n"
        ),
        latex_table(report, ["Task", "Metric"], {mode: report.overview(mode) for mode in report.modes},
                    "Primary metric of each task.", "overview"),
    ]
    for family in report.families:
        details = {mode: report.details(family, mode) for mode in report.modes}
        if any(details.values()):
            tables.append(latex_table(report, ["Subtask", "Metric"], details,
                                      f"{family}: all metrics.", f"{family}-details"))
        sliced = defaultdict(dict)
        for mode in report.modes:
            for dataset, rows in report.sliced(family, mode).items():
                sliced[dataset][mode] = rows
        for dataset, by_mode in sliced.items():
            tables.append(latex_table(report, ["Slice", "Metric"], by_mode,
                                      f"{dataset}: scores by slice.", f"{dataset}-slices"))
    path.write_text("\n".join(tables), encoding="utf8")
    print(f"    - Saved LaTeX to:    {path}")


def wide_frame(runs: list[Run]) -> pd.DataFrame:
    """One row per task, mode and model with a column per metric, for the charts."""
    return pd.DataFrame([
        {"Dataset": run.dataset, "Mode": run.mode, "Model": run.model,
         **{score.metric: score.value for score in run.scores if not score.slice}}
        for run in runs
    ])


def save_visualizations(df: pd.DataFrame, output_dir: Path, output_name: str):
    """Generate charts and heatmaps.

    Charts   → output_dir/charts/      one bar chart per base dataset (all modes, sorted descending)
    Heatmaps → output_dir/heatmaps/    one heatmap per (base_dataset × mode) + one global heatmap
    """
    charts_dir   = output_dir / "charts"
    heatmaps_dir = output_dir / "heatmaps"
    charts_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    reserved_cols = {"Dataset", "Mode", "Model", "Model_Full", "Timestamp",
                     "Model Args", "Path", "Voting_Method", "_BaseDataset", "_Label", "_ModeRank"}
    metric_columns = [
        col for col in df.columns
        if col not in reserved_cols
        and pd.api.types.is_numeric_dtype(df[col])
        and not col.endswith("_rank")
    ]

    if not metric_columns:
        print("[INFO] No numeric metrics found for visualization")
        return
    if "Dataset" not in df.columns or "Model" not in df.columns:
        print("[INFO] Missing Dataset or Model columns for visualization")
        return

    MODE_ORDER = ["0-shot", "multi-turn_CoT", "multi-turn_CoT-SC",
                  "multi-turn_CoT-MBR", "cross-consistency", "self-refine_CoT"]

    def mode_rank(mode: str) -> int:
        try:
            return MODE_ORDER.index(mode)
        except ValueError:
            return len(MODE_ORDER)

    def base_dataset(name: str) -> str:
        return name.removesuffix("_0-shot") if isinstance(name, str) else name

    def safe(s: str) -> str:
        return s.replace('/', '_').replace(' ', '_').replace(':', '_')

    def _heatmap(pivot: pd.DataFrame, title: str, path: Path):
        fig_w = max(4, len(pivot.columns) * 1.6)
        fig_h = max(3, len(pivot) * 0.55)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                    cbar_kws={'label': 'Score (%)', 'shrink': 0.6},
                    vmin=0, vmax=100, linewidths=0.4,
                    annot_kws={"size": 8})
        plt.title(title, fontsize=11, fontweight='bold', pad=8)
        plt.xlabel('Metric', fontsize=9)
        plt.ylabel('')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8, rotation=0)
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches='tight')
        plt.close()

    df = df.copy()
    df["_BaseDataset"] = df["Dataset"].apply(base_dataset)
    has_mode = "Mode" in df.columns

    if has_mode:
        df["_Label"]    = df["Model"] + " — " + df["Mode"].fillna("")
        df["_ModeRank"] = df["Mode"].apply(mode_rank)
    else:
        df["_Label"]    = df["Model"]
        df["_ModeRank"] = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Charts: one figure per base dataset, bars sorted descending per metric
    # ─────────────────────────────────────────────────────────────────────────
    for base_ds in sorted(df["_BaseDataset"].unique()):
        group_df = df[df["_BaseDataset"] == base_ds].copy()
        group_df = group_df.sort_values(["_ModeRank", "Model"]).reset_index(drop=True)

        avail = [m for m in metric_columns
                 if m in group_df.columns and not group_df[m].isna().all()]
        if not avail:
            continue

        try:
            n = len(avail)
            n_bars = len(group_df)
            bar_w  = max(0.5, min(0.8, 6 / n_bars))        # narrower bars when many entries
            fig_w  = max(10, n_bars * 1.1)
            fig_h  = 4.5 * n

            fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h))
            if n == 1:
                axes = [axes]

            for ax, metric in zip(axes, avail):
                plot_df = group_df[["_Label", metric]].dropna().copy()
                plot_df[metric] = plot_df[metric] * 100
                plot_df = plot_df.sort_values(metric, ascending=False).reset_index(drop=True)

                cmap   = plt.cm.RdYlGn
                norm   = plt.Normalize(vmin=plot_df[metric].min(), vmax=plot_df[metric].max())
                colors = [cmap(norm(v)) for v in plot_df[metric]]

                bars = ax.bar(plot_df["_Label"], plot_df[metric],
                              color=colors, width=bar_w, edgecolor='white', linewidth=0.5)

                # Value labels on top of bars
                for bar, val in zip(bars, plot_df[metric]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=7.5, fontweight='bold')

                ax.set_ylabel(f'{metric} (%)', fontsize=9)
                ax.set_title(metric, fontsize=10, fontweight='bold')
                ax.set_ylim(0, 108)
                ax.set_xticks(range(len(plot_df)))
                ax.set_xticklabels(plot_df["_Label"], rotation=40, ha='right', fontsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(axis='y', alpha=0.25, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            fig.suptitle(base_ds, fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            path = charts_dir / f"{output_name}_{safe(base_ds)}.png"
            plt.savefig(path, dpi=180, bbox_inches='tight')
            plt.close()
            print(f"    - Saved chart:    {path}")
        except Exception as e:
            plt.close('all')
            print(f"    - Failed chart for '{base_ds}': {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmaps: one per (base_dataset × mode), rows = Model, cols = metrics
    # ─────────────────────────────────────────────────────────────────────────
    group_cols = ["_BaseDataset", "Mode"] if has_mode else ["_BaseDataset"]
    for key, grp in df.groupby(group_cols, sort=False):
        base_ds, mode = (key[0], key[1]) if has_mode else (key, "")
        avail = [m for m in metric_columns
                 if m in grp.columns and not grp[m].isna().all()]
        if not avail:
            continue
        try:
            pivot = grp.set_index("Model")[avail].copy()
            pivot = pivot.sort_index()
            title = f"{base_ds} — {mode}" if mode else base_ds
            path  = heatmaps_dir / f"{output_name}_heatmap_{safe(base_ds)}_{safe(mode)}.png"
            _heatmap(pivot, title, path)
            print(f"    - Saved heatmap:  {path}")
        except Exception as e:
            plt.close('all')
            print(f"    - Failed heatmap for '{base_ds} / {mode}': {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Global heatmap: rows = "Model — Mode", cols = "Dataset / metric"
    # ─────────────────────────────────────────────────────────────────────────
    try:
        global_rows = []
        for _, row in df.sort_values(["_BaseDataset", "_ModeRank", "Model"]).iterrows():
            avail = [m for m in metric_columns if pd.notna(row.get(m))]
            for m in avail:
                global_rows.append({
                    "row_label": row["_Label"],
                    "col_label": f"{row['_BaseDataset']} / {m}",
                    "value":     row[m],
                })

        if global_rows:
            gdf   = pd.DataFrame(global_rows)
            pivot = gdf.pivot_table(index="row_label", columns="col_label",
                                    values="value", aggfunc="first")
            # Preserve row order: mode rank then model
            ordered_labels = (
                df.sort_values(["_ModeRank", "Model"])["_Label"].drop_duplicates().tolist()
            )
            pivot = pivot.reindex([l for l in ordered_labels if l in pivot.index])

            path = heatmaps_dir / f"{output_name}_heatmap_GLOBAL.png"
            _heatmap(pivot, "All Datasets × All Modes", path)
            print(f"    - Saved global heatmap: {path}")
    except Exception as e:
        plt.close('all')
        print(f"    - Failed global heatmap: {e}")

    df.drop(columns=["_BaseDataset", "_Label", "_ModeRank"], errors="ignore", inplace=True)


def save_outputs(runs: list[Run], hidden: list[tuple[Run, str]], output_dir: str, output_name: str,
                 output_csv: bool = True, output_md: bool = True, output_latex: bool = True,
                 output_charts: bool = True) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = Report(runs)
    print("\n[INFO] Generating outputs...\n")
    if output_csv:
        tidy_frame(runs).to_csv(output / f"{output_name}.csv", index=False)
        print(f"    - Saved CSV to:      {output / f'{output_name}.csv'}")
    if output_md:
        save_markdown(report, hidden, output / f"{output_name}.md")
    if output_latex:
        save_latex(report, output / f"{output_name}.tex")
    if output_charts:
        save_visualizations(wide_frame(runs), output, output_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input_folders", nargs="+", required=True,
                        help="Folders searched recursively for results_*.json and Summary_*.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True, help="Base name of the output files")
    parser.add_argument("--file_filter", default=None,
                        help="Only read JSON files whose name contains this text")
    parser.add_argument("--all-runs", action="store_true",
                        help="Show every run instead of one per task, mode and model")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--no-latex", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    runs = mark_partial_summaries(with_families(collect(args.input_folders, args.file_filter)))
    if not runs:
        raise SystemExit("[ERROR] No results found: check the input folders and the file filter")
    shown, hidden = select(runs, args.all_runs)
    print(f"[INFO] {len(shown)} results shown, {len(hidden)} older or smaller runs not shown")
    save_outputs(shown, hidden, args.output_dir, args.output_name, not args.no_csv,
                 not args.no_markdown, not args.no_latex, not args.no_charts)
    print(f"\n[SUCCESS] Results written to {args.output_dir}/\n")


if __name__ == "__main__":
    main()
