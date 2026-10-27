# Task YAMLs load their functions from this module, so it re-exports the base
# task utils and the augmentation aggregations.
from lm_eval.opencteval.augmentation import (  # noqa: F401
    acc_drop,
    consistency,
    faithfulness,
    macro_f1,
    sliced_results,
)
from lm_eval.tasks.SemEval_NLI4CT.utils import *  # noqa: F403
from lm_eval.tasks.SemEval_NLI4CT.utils import is_comparison, pos_answers


def process_results(doc, results):
    kind = "comparison" if is_comparison(doc) else "single"
    scores = sliced_results(doc, results, pos_answers, extra=[f"trials={kind}"])
    scores["faithfulness"] = scores["consistency"]
    return scores
