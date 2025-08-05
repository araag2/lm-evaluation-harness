baseline_prompt="You are a medical expert tasked with determining the relationship between a medical premise and a hypothesis.\n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nPlease provide your judgement (entailment, neutral or contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nAnswer: "

poss_answers = ['entailment', 'neutral', 'contradiction']

def label_to_index(doc) -> int:
    return poss_answers.index(doc["Label"])

def doc_to_text(doc):
    res = baseline_prompt
    for key in ["Premise", "Hypothesis"]:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res