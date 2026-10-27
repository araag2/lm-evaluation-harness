from lm_eval.opencteval.augmentation import (  # noqa: F401
    acc_drop,
    consistency,
    faithfulness,
    necessity,
    positive_f1,
    semeval_score,
    sliced_results,
)
from lm_eval.opencteval.evidence import EVIDENCE_ONLY, EVIDENCE_REMOVED, FULL
from lm_eval.tasks.SemEval_NLI4CT.utils import *  # noqa: F403
from lm_eval.tasks.SemEval_NLI4CT.utils import pos_answers


def as_augmented(doc: dict) -> dict:
    """A 2024 test row in the augmentation schema: a contrast row is a variant of its base statement.

    `Causal_type` is [Preserving|Altering, id of the base statement]; base statements have none.
    """
    causal = doc.get("Causal_type")
    if not causal:
        return {**doc, "source_id": doc["id"], "augmentation": None}
    kind, base_id = causal[0], causal[1]
    return {
        **doc,
        "source_id": base_id,
        "augmentation": {"mode": doc["Intervention"], "effect": kind.lower()},
    }


def process_results_2024(doc, results):
    """Scores of one 2024 test row.

    Accuracy and F1 over every row as before, plus SemEval-2024's official F1 (base statements only),
    Faithfulness, Consistency and their average, and accuracy per intervention.
    """
    scores = sliced_results(as_augmented(doc), results, pos_answers)
    for key in ("f1@original", "f1@augmented"):  # f1_control is the official form
        scores.pop(key, None)
    item = scores["consistency"]
    scores["faithfulness"] = scores["semeval_score"] = item
    if item[1]:
        scores["f1_control"] = (item[2], item[3])
    return scores


def as_evidence_view(doc: dict) -> dict:
    """An evidence-ablation row in the augmentation schema: each view is a variant of the full one.

    Removing the evidence of an Entailment statement alters its label (to Contradiction); the other
    views preserve it.
    """
    if doc["view"] == FULL:
        return {**doc, "augmentation": None}
    effect = "altering" if doc["Label"] != doc["Gold_Label"] else "preserving"
    return {**doc, "augmentation": {"mode": doc["view"], "effect": effect}}


def process_results_evidence(doc, results):
    """Scores of one evidence-ablation row: accuracy per view, sufficiency and necessity.

    sufficiency: evidence-only predictions that equal the full-section prediction. faithfulness and
    necessity: Entailment statements that lose Entailment once the evidence is removed (necessity
    counts only those predicted right with the full section).
    """
    scores = sliced_results(as_evidence_view(doc), results, pos_answers)
    item = scores.pop("consistency")
    for key in ("acc_drop", "f1@original", "f1@augmented"):
        scores.pop(key, None)
    if doc["view"] != EVIDENCE_REMOVED:
        scores["sufficiency"] = item
    if doc["view"] != EVIDENCE_ONLY:
        scores["faithfulness"] = scores["necessity"] = item
    return scores
