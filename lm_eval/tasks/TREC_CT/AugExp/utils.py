# Task YAMLs load their functions from this module, so it re-exports the base
# task utils and the augmentation aggregations.
from lm_eval.opencteval.augmentation import (  # noqa: F401
    acc_drop,
    consistency,
    macro_f1,
    sliced_results,
)
from lm_eval.tasks.TREC_CT.utils import *  # noqa: F403
from lm_eval.tasks.TREC_CT.utils import is_truncated, pos_answers


def process_results(doc, results):
    truncated = "yes" if is_truncated(doc) else "no"
    return sliced_results(doc, results, pos_answers, extra=[f"truncated={truncated}"])
