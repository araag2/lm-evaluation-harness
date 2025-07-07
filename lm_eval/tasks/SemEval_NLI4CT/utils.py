import random
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

pos_answers = ['Entailment', 'Contradiction']

@register_filter("random_label_if_invalid")
class RandomLabelIfInvalid(Filter):
    """Returns a random label if no valid responses are provided."""
    def apply(self, resps, docs, **kwargs):
        print(f'{resps=}, {list(resps)=}, {docs=}')
        for resp in resps:
            if not resp or resp not in pos_answers:
                resp = random.choice(pos_answers)
        print(f'{resps=}, {list(resps)=}, {docs=}')
        return resps


#def random_label_if_invalid(resps) -> str:
#    out = [resp if resp and resp in pos_answers else random.choice(pos_answers) for resp in resps]
#    return out

def label_to_index(doc) -> int:
    assert isinstance(pos_answers.index(doc["Label"]), int)
    return pos_answers.index(doc["Label"])