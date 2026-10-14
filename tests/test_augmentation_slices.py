import math

import pytest

from lm_eval.opencteval.augmentation import (
    acc_drop,
    consistency,
    flips,
    macro_f1,
    matched_pairs,
    prediction,
    sliced_results,
    slices,
)


CHOICES = ["not relevant", "possibly relevant", "definitely relevant"]


def doc(mode="original", label="possibly relevant", source_id="s1", **metadata):
    return {"source_id": source_id, "Label": label, "augmentation": {"mode": mode, **metadata}}


def test_slices_of_an_original_and_of_a_variant():
    assert slices(doc(), "possibly relevant") == ["all", "label=possibly relevant", "original"]
    variant = doc("permute-reword_all", side="patient", effect="preserving")
    assert slices(variant, "possibly relevant", extra=["truncated=no"]) == [
        "all", "label=possibly relevant", "augmented", "type=permute-reword_all",
        "side=patient", "effect=preserving", "truncated=no",
    ]


def test_original_stored_with_null_variant_fields():
    """A published table gives every row the same struct, so originals have nulls."""
    original = doc(side=None, effect=None)
    assert slices(original, "possibly relevant")[-1] == "original"


@pytest.mark.parametrize("results, expected", [
    ([(-2.0, False), (-0.1, True), (-1.0, False)], 1),
    ([["-2.0", "False"], ["-3.0", "False"], ["-1.0", "False"]], 2),  # as logged in samples
    (["Definitely relevant"], 2),
    (["[invalid]"], None),
])
def test_prediction(results, expected):
    assert prediction(results, CHOICES) == expected


def test_sliced_results_emits_a_key_only_for_the_rows_slices():
    variant = doc("permute-reword_all", side="patient", effect="preserving")
    scores = sliced_results(variant, [(-2.0, False), (-0.1, True), (-1.0, False)], CHOICES)
    assert scores["acc"] == scores["acc@augmented"] == scores["acc@type=permute-reword_all"] == 1.0
    assert "acc@original" not in scores and "unparsed" not in scores
    assert scores["f1@augmented"] == (1, 1)
    assert scores["consistency"] == ("s1", False, 1, 1)


def test_unparsed_generation_is_wrong_and_counted():
    scores = sliced_results(doc(), ["[invalid]"], CHOICES)
    assert scores["acc"] == 0.0 and scores["unparsed"] == 1.0


def test_pairing_consistency_and_drop():
    items = [
        ("s1", True, 1, 1),   # original, correct
        ("s1", False, 1, 1),  # same prediction, correct
        ("s1", False, 1, 0),  # changed prediction, now wrong
        ("s2", True, 2, 0),   # original, wrong
        ("s2", False, 2, 2),  # changed prediction, now correct
        ("s2", False, 0, 0),  # derived label differs from its original: not paired
        ("s3", False, 1, 1),  # its original was not scored: not paired
    ]
    assert len(matched_pairs(items)) == 3
    assert consistency(items) == pytest.approx(1 / 3)
    assert acc_drop(items) == pytest.approx(2 / 3 - 2 / 3)
    assert flips(items) == {"correct_to_wrong": 1, "wrong_to_correct": 1}


def test_two_unparsed_answers_are_not_consistent():
    assert consistency([("s1", True, 1, None), ("s1", False, 1, None)]) == 0.0


def test_no_pairs_gives_nan_not_zero():
    assert math.isnan(consistency([("s1", True, 1, 1)]))
    assert math.isnan(acc_drop([]))


def test_macro_f1_over_gold_classes():
    assert macro_f1([(0, 0), (1, 1), (2, 2)]) == 1.0
    # class 0: one hit, two predicted, one gold -> 2/3; class 1: missed -> 0
    assert macro_f1([(0, 0), (1, 0)]) == pytest.approx(1 / 3)
    assert macro_f1([(1, None)]) == 0.0
