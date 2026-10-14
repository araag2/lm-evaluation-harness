"""The result-collection script: which runs are shown, how they are named and written."""

import importlib.util
import json
import re
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_run-results.py"
spec = importlib.util.spec_from_file_location("extract_run_results", SCRIPT)
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def results_file(folder, stamp, results, model="org/Model-A", limit=None, groups=None, original=100):
    path = folder / f"results_{stamp}.json"
    path.write_text(json.dumps({
        "results": results,
        "group_subtasks": groups or {},
        "n-samples": {task: {"original": original, "effective": r.get("sample_len", original)}
                      for task, r in results.items()},
        "config": {"limit": limit, "model_args": {"pretrained": model}},
        "model_name": model,
        "higher_is_better": {},
    }))
    return path


def summary_file(folder, stamp, results, task="TREC_CT:balanced_2023_CoT", n=4):
    path = folder / f"Summary_{stamp}.json"
    path.write_text(json.dumps({
        "mode": "multi-turn_CoT", "reasoning_task": task, "results": results,
        "reasoning_model": "pretrained=org/Model-A,max_length=4096",
        "samples": {str(i): {} for i in range(n)},
    }))
    return path


def load(folder, all_runs=False):
    runs = extract.mark_partial_summaries(extract.with_families(extract.collect([str(folder)])))
    return extract.select(runs, all_runs)


@pytest.mark.parametrize("task, expected", [
    ("MedNLI_Para-0-shot", ("MedNLI", "Para-0-shot")),
    ("TREC_CT_balanced_2021_CoT_SC", ("TREC_CT_balanced_2021", "CoT_SC")),
    ("TREC_CT_2021_CoT", ("TREC_CT_2021", "CoT")),
    ("MedQA_0-shot", ("MedQA", "0-shot")),
    ("plain_task", ("plain_task", "single-turn")),
])
def test_mode_comes_from_the_task_name(task, expected):
    assert extract.split_task(task) == expected


def test_slices_are_split_from_the_metric_but_ranking_cutoffs_are_not():
    assert extract.split_metric("acc@type=permute-reword_all") == ("acc", "type=permute-reword_all")
    assert extract.split_metric("f1@original") == ("f1", "original")
    assert extract.split_metric("P@10") == ("P@10", "")
    assert extract.split_metric("nDCG@5") == ("nDCG@5", "")


def test_model_name_never_comes_from_a_placeholder():
    assert extract.model_name("N/A", {"pretrained": "unsloth/Qwen3-8B"}) == "Qwen3-8B"
    assert extract.model_name(None, "pretrained=org/Name,max_length=1") == "Name"
    assert extract.model_name(None, None) == "unknown model"


def test_only_numeric_metrics_are_read(tmp_path):
    results_file(tmp_path, "2026-01-01T00-00-00.1", {"MedQA_0-shot": {
        "name": "MedQA_0-shot", "alias": "MedQA_0-shot", "sample_len": 100,
        "sample_count": {"acc,none": 100}, "acc,none": 0.5, "acc_stderr,none": 0.05,
        "f1,none": 0.4, "f1_stderr,none": "N/A"}})
    (run,), _ = load(tmp_path)
    assert [(s.metric, s.value, s.stderr) for s in run.scores] == [("acc", 0.5, 0.05), ("f1", 0.4, None)]
    assert (run.dataset, run.mode, run.model, run.n, run.limited) == ("MedQA", "0-shot", "Model-A", 100, False)


def test_one_run_per_task_mode_and_model(tmp_path):
    """A smoke test and an older rerun are not shown; the reason is recorded."""
    task = "MedQA_0-shot"
    results_file(tmp_path, "2026-01-01T00-00-00.1", {task: {"sample_len": 100, "acc,none": 0.5}})
    results_file(tmp_path, "2026-01-02T00-00-00.1", {task: {"sample_len": 100, "acc,none": 0.6}})
    results_file(tmp_path, "2026-01-03T00-00-00.1", {task: {"sample_len": 3, "acc,none": 1.0}}, limit=3)
    shown, hidden = load(tmp_path)
    assert [run.scores[0].value for run in shown] == [0.6]
    assert sorted(reason for _, reason in hidden) == ["fewer samples", "older run"]
    everything, none_hidden = load(tmp_path, all_runs=True)
    assert len(everything) == 3 and not none_hidden
    assert len({run.model for run in everything}) == 3


def test_a_limited_run_is_kept_and_marked_when_it_is_the_only_one(tmp_path):
    results_file(tmp_path, "2026-01-01T00-00-00.1",
                 {"MedQA_0-shot": {"sample_len": 50, "acc,none": 0.5}}, limit=50)
    (run,), _ = load(tmp_path)
    assert run.limited
    row = extract.Report([run]).overview("0-shot")[0]
    assert extract.cells(row, ["Model-A"], "**{}**", "†") == ["50.00†"]


def test_group_and_subtasks_form_a_family_across_file_kinds(tmp_path):
    results_file(tmp_path, "2026-01-01T00-00-00.1", {
        "TREC_CT_balanced_2023_0-shot": {"sample_len": 4, "acc,none": 0.5},
        "TREC_CT_balanced_0-shot": {"sample_len": 4, "acc,none": 0.5},
    }, groups={"TREC_CT_balanced_0-shot": ["TREC_CT_balanced_2023_0-shot"]}, original=4)
    summary_file(tmp_path, "2026-01-02T00-00", {"name": 0.0, "sample_len": 0.0, "acc": 0.7})
    shown, _ = load(tmp_path)
    assert {(run.family, run.subtask, run.mode) for run in shown} == {
        ("TREC_CT_balanced", "all", "0-shot"), ("TREC_CT_balanced", "2023", "0-shot"),
        ("TREC_CT_balanced", "2023", "multi-turn_CoT")}
    report = extract.Report(shown)
    assert [row.labels[0] for row in report.overview("0-shot")] == ["TREC_CT_balanced"]


def test_best_is_bold_by_direction_and_not_when_all_are_equal(tmp_path):
    for model, acc, mse in (("org/A", 0.5, 2.0), ("org/B", 0.7, 3.0)):
        folder = tmp_path / model.split("/")[1]
        folder.mkdir()
        results_file(folder, "2026-01-01T00-00-00.1", {"T_0-shot": {
            "sample_len": 100, "acc,none": acc, "mean_squared_error,none": mse, "f1,none": 0.2}},
            model=model)
    shown, _ = load(tmp_path)
    rows = {row.metric: row for row in extract.Report(shown).details("T", "0-shot")}
    cells = lambda metric: extract.cells(rows[metric], ["A", "B"], "**{}**", "†")
    assert cells("acc") == ["50.00", "**70.00**"]
    assert cells("mean_squared_error") == ["**2.000**", "3.000"]  # lower is better, not a percentage
    assert cells("f1") == ["20.00", "20.00"]


def test_written_files(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    results_file(source, "2026-01-01T00-00-00.1", {"Aug_Exp_0-shot": {
        "sample_len": 100, "acc,none": 0.5, "acc@type=permute_x,none": 0.25, "acc@original,none": 0.75}})
    results_file(source, "2026-01-01T00-00-00.2", {"Aug_Exp_CoT": {"sample_len": 100, "acc,none": 0.6}})
    shown, hidden = load(source)
    extract.save_outputs(shown, hidden, str(tmp_path / "out"), "r", output_charts=False)
    markdown = (tmp_path / "out" / "r.md").read_text()
    assert "| type=permute_x |" in markdown and "25.00" in markdown
    assert "Aug_Exp, 0-shot: by slice" in markdown
    assert (tmp_path / "out" / "r_sources.md").exists()

    latex = (tmp_path / "out" / "r.tex").read_text()
    text = "\n".join(line for line in latex.split("\n") if not line.startswith(r"\label"))
    assert r"Aug\_Exp" in text and "Aug_Exp" not in text  # escaped everywhere but in labels
    assert latex.count(r"\begin{table}") == latex.count(r"\end{table}") > 0
    for table in re.findall(r"\\begin\{tabular\}\{(\w+)\}(.*?)\\end\{tabular\}", latex, re.DOTALL):
        columns, body = table
        lines = [line for line in body.split("\n") if line.endswith(r"\\") and "multicolumn" not in line]
        assert lines and all(line.count(" & ") == len(columns) - 1 for line in lines)

    import pandas as pd
    frame = pd.read_csv(tmp_path / "out" / "r.csv")
    sliced = frame[frame["Slice"] == "type=permute_x"].iloc[0]
    assert (sliced["Metric"], sliced["Value"], sliced["Mode"], sliced["N"]) == ("acc", 0.25, "0-shot", 100)
