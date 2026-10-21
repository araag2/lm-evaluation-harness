from lm_eval.opencteval.augmentation import acc_drop, consistency, macro_f1, sliced_results
from lm_eval.tasks.SemEval_NLI4CT.utils import *


def process_results(doc, results):
    kind = "comparison" if is_comparison(doc) else "single"
    return sliced_results(doc, results, pos_answers, extra=[f"trials={kind}"])
