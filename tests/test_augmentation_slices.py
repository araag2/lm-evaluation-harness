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


def test_faithfulness_counts_only_label_altering_variants():
    from lm_eval.opencteval.augmentation import faithfulness

    items = [
        ("s1", True, 1, 1),      # original: Entailment
        ("s1", False, 0, 0),     # altered to Contradiction, prediction moved: faithful
        ("s1", False, 0, 1),     # altered, prediction stayed on the original gold: not faithful
        ("s1", False, 0, None),  # altered, unparsed: not faithful
        ("s1", False, 1, 0),     # preserving: not counted
        ("s2", False, 0, 0),     # original missing: not counted
    ]
    assert faithfulness(items) == pytest.approx(1 / 3)


def test_positive_f1_and_semeval_score():
    from lm_eval.opencteval.augmentation import positive_f1, semeval_score

    assert positive_f1([(1, 1), (1, 0), (0, 1), (0, 0)]) == pytest.approx(0.5)
    assert math.isnan(positive_f1([(0, 0)]))
    items = [
        ("a", True, 1, 1), ("a", False, 1, 1), ("a", False, 0, 0),  # F1 pair, consistent, faithful
        ("b", True, 0, 0), ("b", False, 0, 1),                      # inconsistent
    ]
    assert semeval_score(items) == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_semeval_2024_rows_map_onto_their_base_statement():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "lm_eval/tasks/SemEval_NLI4CT/0-shot/utils.py"
    spec = importlib.util.spec_from_file_location("semeval_0shot_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = {"id": "b1", "Label": "Entailment", "Intervention": "Base_Statement", "Causal_type": None}
    altered = {"id": "v1", "Label": "Contradiction", "Intervention": "Contradiction",
               "Causal_type": ["Altering", "b1"]}
    entail_wins = [(-2.0, False), (-0.5, True)]
    base_scores = module.process_results_2024(base, entail_wins)
    variant_scores = module.process_results_2024(altered, entail_wins)
    assert base_scores["consistency"] == ("b1", True, 1, 1)
    assert base_scores["f1_control"] == (1, 1)
    assert variant_scores["consistency"] == ("b1", False, 0, 1)
    assert "f1_control" not in variant_scores and "f1@augmented" not in variant_scores
    assert variant_scores["acc@type=Contradiction"] == 0.0
    assert variant_scores["acc@effect=altering"] == 0.0


def test_necessity_counts_only_originals_predicted_right():
    from lm_eval.opencteval.augmentation import necessity

    items = [
        ("a", True, 1, 1), ("a", False, 0, 0),  # right original, prediction moved: necessary
        ("b", True, 1, 1), ("b", False, 0, 1),  # right original, prediction stayed: not
        ("c", True, 1, 0), ("c", False, 0, 0),  # wrong original: not counted
    ]
    assert necessity(items) == pytest.approx(0.5)


def test_evidence_views_keep_headers_and_need_partial_evidence():
    from lm_eval.opencteval.evidence import evidence_views, is_header, view_label

    section = ["Inclusion Criteria:", "  Age 18", "  ECOG 0-1", "Exclusion Criteria:", "  Pregnancy"]
    assert is_header("INTERVENTION 1: ") and not is_header("  Age 18") and not is_header("ANAEMIA 1/2")
    views = evidence_views([section], [[2]])
    assert views["evidence_only"] == [["Inclusion Criteria:", "  ECOG 0-1", "Exclusion Criteria:"]]
    assert views["evidence_removed"] == [
        ["Inclusion Criteria:", "  Age 18", "Exclusion Criteria:", "  Pregnancy"]
    ]
    assert evidence_views([section], [[0, 1, 2, 3, 4]]) is None  # whole section: nothing left
    assert evidence_views([section], [[0]]) is None  # a header only: no evidence content
    # comparison: the primary evidence is whole, the secondary's is partial
    assert evidence_views([section, section], [[1, 2, 4], [1]]) is not None
    assert view_label("evidence_removed", "Entailment") == "Contradiction"
    assert view_label("evidence_only", "Entailment") == "Entailment"


def test_evidence_ablation_rows_score_sufficiency_and_necessity():
    import importlib.util
    from pathlib import Path

    from lm_eval.opencteval.augmentation import consistency, necessity

    path = Path(__file__).resolve().parents[1] / "lm_eval/tasks/SemEval_NLI4CT/0-shot/utils.py"
    spec = importlib.util.spec_from_file_location("semeval_0shot_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def row(view, label):
        return {"source_id": "s", "view": view, "Label": label, "Gold_Label": "Entailment"}

    entail, contra = [(-2.0, False), (-0.5, True)], [(-0.5, True), (-2.0, False)]
    full = module.process_results_evidence(row("full", "Entailment"), entail)
    only = module.process_results_evidence(row("evidence_only", "Entailment"), entail)
    removed = module.process_results_evidence(row("evidence_removed", "Contradiction"), contra)
    assert "sufficiency" not in removed and "necessity" not in only
    assert "consistency" not in full and "acc_drop" not in full
    assert consistency([full["sufficiency"], only["sufficiency"]]) == 1.0
    assert necessity([full["necessity"], removed["necessity"]]) == 1.0
    assert removed["acc@type=evidence_removed"] == 1.0 and removed["acc@effect=altering"] == 1.0
