baseline_prompt="You are a medical expert tasked with determining the relationship between a medical premise and a hypothesis.\n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nPlease provide your judgement (entailment, neutral or contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nAnswer: "

reasoning_prompt="You are a medical expert tasked with determining the logical relationship between a medical premise and a hypothesis. The possible relationships are:\n- Entailment: The premise directly supports the hypothesis as true.\n- Neutral: The premise does not provide enough information to confirm or deny the hypothesis.\n- Contradiction: The premise directly conflicts with the hypothesis. \n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nPlease provide your judgement (Entailment, Neutral or Contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nLet's think step by step: "

poss_answers = ['entailment', 'neutral', 'contradiction']

def label_to_index(doc) -> int:
    return poss_answers.index(doc["Label"])

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_prompt
    for key in ["Premise", "Hypothesis"]:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)