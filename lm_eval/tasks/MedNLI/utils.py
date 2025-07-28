poss_answers = ['entailment', 'neutral', 'contradiction']

def label_to_index(doc) -> int:
    return poss_answers.index(doc["Label"])