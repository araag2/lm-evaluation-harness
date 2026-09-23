import pytest

from lm_eval.api.metrics import (
    _base_intervention_id,
    acc_original_agg,
    acc_paraphrase_agg,
    augmentation_consistency_agg,
    consistency_agg,
    faithfulness_agg,
    filter_by_id,
)


@pytest.mark.parametrize("suffix", [
    "_paraphrase", "_paraphrase1", "_paraphrase_1",
    "_paraphrase_lexical_1", "_paraphrase_lexical-syntactic_2",
    "_paraphrase_tonal_shift_lay_3", "_paraphrase_syntactic",
    "_contradiction", "_contradiction2", "_contradiction_lexical_1",
])
def test_intervention_suffixes(suffix):
    assert _base_intervention_id("sample_42" + suffix) == "sample_42"


@pytest.mark.parametrize("original", ["sample_42", "sample_paraphrased_1", "sample_contradictions"])
def test_original_ids_are_unchanged(original):
    assert _base_intervention_id(original) == original


@pytest.mark.parametrize("id_field", ["id", "query_id"])
def test_consistency_pairs_current_pipeline_ids(id_field):
    items = [
        ({id_field: "sample_paraphrase_lexical-syntactic_1"}, 1, 1),
        ({id_field: "sample"}, 1, 1),
        ({id_field: "sample_paraphrase_tonal_shift_lay_2"}, 1, 0),
        ({id_field: "unpaired_paraphrase_lexical_1"}, 1, 1),
    ]
    assert consistency_agg(items) == 0.5
    assert acc_original_agg(items[:3]) == 1.0
    assert acc_paraphrase_agg(items[:3]) == 0.5


def test_query_id_and_numeric_original_id():
    items = [({"query_id": 42}, 0, 0), ({"query_id": "42_paraphrase_lexical_1"}, 0, 0)]
    assert consistency_agg(items) == 1.0


def test_empty_metrics_and_legacy_contradictions():
    assert filter_by_id([], lambda _: True) == []
    assert acc_original_agg([]) == 0.0
    assert acc_paraphrase_agg([]) == 0.0
    assert consistency_agg([]) == 0.0
    assert faithfulness_agg([
        ({"id": "a"}, 1, 1), ({"id": "a_contradiction2"}, 0, 0)
    ]) == 1.0


def augmented_doc(mode, option_ids, effect="preserving", source_id="q"):
    return {"source_id": source_id, "augmentation": {
        "mode": mode, "option_ids": option_ids,
        "effect": "original" if mode == "original" else effect,
        "consistency_eligible": effect == "preserving",
    }}


def test_augmentation_consistency_tracks_identity_not_position():
    original = (augmented_doc("original", ["a", "b", "c"]), 0, 0)
    same_answer = (augmented_doc("shuffle_options", ["c", "a", "b"]), 1, 1)
    same_letter_wrong_answer = (augmented_doc("shuffle_options", ["b", "c", "a"]), 2, 0)
    assert augmentation_consistency_agg([original, same_answer]) == 1.0
    assert augmentation_consistency_agg([original, same_letter_wrong_answer]) == 0.0
    assert augmentation_consistency_agg([same_answer, original, same_letter_wrong_answer]) == 0.5


def test_augmentation_consistency_excludes_task_and_answer_changes():
    original = (augmented_doc("original", ["a", "b", "c"]), 0, 0)
    preserving = (augmented_doc("paraphrase_both", ["a", "b", "c"]), 0, 0)
    negative = (augmented_doc("negate_question", ["a", "b"], "task_changed"), 1, 1)
    nota = (augmented_doc("none_of_above", ["b", "none_of_above"], "answer_set_changed"), 1, 1)
    orphan = (augmented_doc("shuffle_options", ["a", "b"], source_id="missing"), 0, 1)
    assert augmentation_consistency_agg([original, preserving, negative, nota, orphan]) == 1.0
    assert augmentation_consistency_agg([original, negative, nota]) == 0.0
    assert augmentation_consistency_agg([]) == 0.0


def test_augmentation_metric_is_emitted_by_task():
    from tests.test_metrics import MockConfigurableTask

    task = MockConfigurableTask()
    task._metric_fn_list = {"augmentation_consistency": None}
    task._metric_fn_kwargs = {"augmentation_consistency": {}}
    doc = augmented_doc("original", ["a", "b", "c"])
    result = task.process_results(doc, [(-2.0, False), (-0.1, False), (-1.0, False)])
    assert result["augmentation_consistency"] == (doc, 1, 1)
