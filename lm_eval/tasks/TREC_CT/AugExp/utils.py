from lm_eval.opencteval.augmentation import acc_drop, consistency, macro_f1, sliced_results
from lm_eval.tasks.TREC_CT.utils import *


def process_results(doc, results):
    truncated = "yes" if is_truncated(doc) else "no"
    return sliced_results(doc, results, pos_answers, extra=[f"truncated={truncated}"])
