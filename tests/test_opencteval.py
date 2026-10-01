"""Offline regression tests for OpenCTEval extensions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lm_eval.api import metrics
from lm_eval.api.registry import get_metric, get_metric_aggregation, is_higher_better
from lm_eval.filters.selection import TakeLastFilter, TakeLastKFilter
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.opencteval import metrics as custom_metrics
from lm_eval.reasoning_modes.reasoning_utils import inject_reasoning_into_dataset
from lm_eval.reasoning_modes.voting.strategies.simple import majority_aggregate_votes
from tests.opencteval_metric_directions import OPENCTEVAL_METRIC_DIRECTIONS
from tests.test_metrics import MockConfigurableTask


@pytest.mark.parametrize(
    "statement",
    [
        "from lm_eval.tasks import TaskManager; TaskManager()",
        "from lm_eval.opencteval.metrics import consistency_agg; assert consistency_agg([]) == 0",
        "from lm_eval.api.task import CrossConsistencyCoT",
        (
            "import multiprocessing as mp; before = mp.get_start_method(allow_none=True); "
            "import lm_eval.__main__; assert mp.get_start_method(allow_none=True) == before"
        ),
    ],
)
def test_fresh_process_imports(statement):
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned statements
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "metric,function,aggregation",
    [
        (
            "augmentation_consistency",
            "augmentation_consistency_fn",
            "augmentation_consistency_agg",
        ),
        ("consistency", "consistency_fn", "consistency_agg"),
        ("MAP", "MAP_fn", "MAP_score"),
        ("span_f1", "span_f1_fn", "span_f1_agg"),
    ],
)
def test_legacy_imports_and_registry(metric, function, aggregation):
    assert getattr(metrics, function) is getattr(custom_metrics, function)
    assert get_metric(metric) is getattr(custom_metrics, function)
    assert get_metric_aggregation(metric) is getattr(custom_metrics, aggregation)


def test_classification_and_exact_match_contracts():
    assert metrics.f1_score([(0, 0), (1, 2), (2, 2)]) == pytest.approx(5 / 9)
    assert metrics.f1_score(
        [("included", "included"), ("excluded", "included")]
    ) == pytest.approx(2 / 3)
    assert (
        metrics.exact_match_fn(predictions=[" Yes! "], references=["yes"])[
            "exact_match"
        ]
        == 1
    )
    assert (
        metrics.exact_match_fn(
            predictions=[" Yes! "],
            references=["yes"],
            ignore_case=False,
            ignore_punctuation=False,
        )["exact_match"]
        == 0
    )


def test_span_and_numeric_scoring_contracts():
    assert custom_metrics.span_f1_agg(
        [
            ("aspirin\nplacebo", " Aspirin \nibuprofen"),
            ("saline", "saline"),
        ]
    ) == pytest.approx(2 / 3)
    assert (
        custom_metrics.mean_absolute_error_fn(references=[10, 4], predictions=[13, 2])
        == 2.5
    )
    assert (
        custom_metrics.mean_squared_error_fn(references=[10, 4], predictions=[13, 2])
        == 6.5
    )


def test_document_and_ranking_payloads_reach_aggregations():
    task = MockConfigurableTask()
    names = [
        "acc",
        "consistency",
        "augmentation_consistency",
        "Precision",
        "Recall",
        "MAP",
        "nDCG@10",
    ]
    task._metric_fn_list = dict.fromkeys(names)
    task._metric_fn_kwargs = {name: {} for name in names}
    doc = {"query_id": "q1"}
    result = task.process_results(doc, [(-2.0, False), (-0.1, False), (-1.0, False)])
    assert result["acc"] == 1
    assert result["consistency"] == (doc, 1, 1)
    assert result["augmentation_consistency"] == (doc, 1, 1)
    assert result["Precision"] == result["Recall"] == (1, 1)
    for name in ["MAP", "nDCG@10"]:
        assert result[name][:3] == (doc, 1, 1)
        np.testing.assert_allclose(
            result[name][3], np.exp([-2, -0.1, -1]) / np.exp([-2, -0.1, -1]).sum()
        )


def test_generated_accuracy_normalization():
    task = MockConfigurableTask()
    task.OUTPUT_TYPE = "generate_until"
    task.doc_to_target = lambda doc: " Yes! "
    task._metric_fn_list = {"acc": metrics.acc_fn}
    task._metric_fn_kwargs = {"acc": {"ignore_case": True, "ignore_punctuation": True}}
    assert task.process_results({}, ["yes"])["acc"] == 1


def test_last_response_filters():
    responses = [["first", "second", "third"], ["a", "b", "c"]]
    assert list(TakeLastFilter().apply(responses, [])) == ["third", "c"]
    assert list(TakeLastKFilter(k=2).apply(responses, [])) == [
        ["second", "third"],
        ["b", "c"],
    ]


def test_reasoning_injection_aligns_documents_and_prompts():
    first = (
        "The first clinical trial supports the intervention after careful comparison."
    )
    second = "The second clinical trial contradicts the intervention after careful comparison."
    samples = [
        {
            "doc_id": 1,
            "resps": [[second + "\nAnswer: B"]],
            "arguments": [["prompt 2", {}]],
        },
        {
            "doc_id": 0,
            "resps": [[first + "\nAnswer: A"]],
            "arguments": [["prompt 1", {}]],
        },
    ]
    result = inject_reasoning_into_dataset([{"id": "one"}, {"id": "two"}], samples)
    assert list(result["id"]) == ["one", "two"]
    assert list(result["Reasoning_Chain"]) == [first, second]
    assert list(result["Reasoning_Chain_Prompt"]) == ["prompt 1", "prompt 2"]


def test_majority_vote_scores_using_task_contract():
    task = MockConfigurableTask()
    data = {
        0: {
            "doc": {},
            "preds": [1, 0, 1],
            "pred_probs": [
                [-2.0, -0.1, -3.0],
                [-0.1, -2.0, -3.0],
                [-1.0, -0.2, -3.0],
            ],
        }
    }
    task._metric_fn_list = {"acc": None}
    assert majority_aggregate_votes(data, ["A", "B", "C"], task)[0]["acc"] == 1
    assert data[0]["majority"] == "B"


@pytest.mark.parametrize("filename", [None, "custom.json"])
def test_flat_result_paths_and_saved_sample_schema(tmp_path, filename):
    destination = tmp_path / filename if filename else tmp_path
    tracker = EvaluationTracker(output_path=str(destination))
    tracker.general_config_tracker.model_name_sanitized = "example_model"
    tracker.save_results_aggregated({"results": {"example": {"acc,none": 1.0}}})
    sample = {
        "doc_id": 0,
        "doc": {"id": "one"},
        "target": "A",
        "arguments": [("prompt", {"until": []})],
        "resps": [["A"]],
        "filtered_resps": ["A"],
    }
    tracker.save_results_samples("example", [sample])
    results = list(tmp_path.glob("results_*.json"))
    samples = list(tmp_path.glob("samples_example_*.jsonl"))
    assert len(results) == len(samples) == 1
    assert not (tmp_path / "example_model").exists()
    saved = json.loads(samples[0].read_text())
    assert saved["arguments"]["gen_args_0"]["arg_0"] == "prompt"
    assert saved["resps"] == [["A"]]


@pytest.mark.parametrize("name,expected", sorted(OPENCTEVAL_METRIC_DIRECTIONS.items()))
def test_custom_metric_directions(name, expected):
    assert is_higher_better(name) is expected
    assert get_metric(name) is not None
    assert get_metric_aggregation(name) is not None


@pytest.mark.parametrize("name", ["acc_bytes", "likelihood"])
def test_upstream_metric_registrations(name):
    assert is_higher_better(name) is True
    assert get_metric(name)([1, 2]) == [1, 2]
    assert get_metric_aggregation(name) is metrics.mean
