baseline_prompt="You are a medical expert tasked with determining the relationship between a medical premise and a hypothesis.\n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nPlease provide your judgement (entailment, neutral or contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nAnswer: "

reasoning_prompt="You are a medical expert tasked with determining the logical relationship between a medical premise and a hypothesis. The possible relationships are:\n- Entailment: The premise directly supports the hypothesis as true.\n- Neutral: The premise does not provide enough information to confirm or deny the hypothesis.\n- Contradiction: The premise directly conflicts with the hypothesis. \n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nPlease provide your judgement (Entailment, Neutral or Contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [Entailment / Neutral / Contradiction] <END>"

answer_selection_prompt="You are a medical expert tasked with determining the logical relationship between a medical premise and a hypothesis. The possible relationships are:\n- Entailment: The premise directly supports the hypothesis as true.\n- Neutral: The premise does not provide enough information to confirm or deny the hypothesis.\n- Contradiction: The premise directly conflicts with the hypothesis. \n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Premise, the Hypothesis and with special atention to the presented Reasoning Chain, provide your judgement (entailment, neutral or contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nFinal Answer: "

verify_reasoning_prompt="You are a medical expert tasked with verifying a reasoning chain that determines the logical relationship between a medical premise and a hypothesis. The possible relationships are:\n- Entailment: The premise directly supports the hypothesis as true.\n- Neutral: The premise does not provide enough information to confirm or deny the hypothesis.\n- Contradiction: The premise directly conflicts with the hypothesis. \n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [Entailment / Neutral / Contradiction] <END>"

answer_selection_after_verification_prompt="You are a medical expert tasked with determining the logical relationship between a medical premise and a hypothesis. The possible relationships are:\n- Entailment: The premise directly supports the hypothesis as true.\n- Neutral: The premise does not provide enough information to confirm or deny the hypothesis.\n- Contradiction: The premise directly conflicts with the hypothesis. \n\nMedical Premise: {{Premise}}\n\nHypothesis: {{Hypothesis}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Premise, the Hypothesis, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your judgement (entailment, neutral or contradiction), corresponding to the correct option that associates the premise and hypothesis. Be as accurate as possible.\nFinal Answer: "

pos_answers = ['entailment', 'neutral', 'contradiction']

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Premise", "Hypothesis"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    relevant_keys.append("Reasoning_Chain")
    return doc_to_text(doc, answer_selection_prompt)