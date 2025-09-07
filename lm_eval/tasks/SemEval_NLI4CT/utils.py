from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, **kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, **kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

# ====================
# Single Prompts
# ====================

single_baseline_prompt = "You are a medical expert tasked with determining semantic entailment relationships between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate) and a section (the premise you will base your judgement in) from a Clinical Trial Report (CTR). The task is to classify the relationship as one of the following: Entailment (supported by the CTR) or Contradiction (in conflict with the CTR).\n\nSection Name:\n\n{{Section_id}}\n\nCTR Section: \n\n{{Primary_Section}}\n\nHypothesis: {{Statement}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the CTR Section and the Hypothesis.\nAnswer: "

single_reasoning_prompt = "You are a medical expert tasked with determining semantic entailment relationships between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate) and a section (the premise you will base your judgement in) from a Clinical Trial Report (CTR). The task is to classify the relationship as one of the following: Entailment (supported by the CTR) or Contradiction (in conflict with the CTR).\n\nSection Name:\n\n{{Section_id}}\n\nCTR Section: \n\n{{Primary_Section}}\n\nHypothesis: {{Statement}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that describes the logical relationship betweeb the CTR and the Hypotheesis. Let's think step by step, and at the very end write your answer in the form: \nAnswer: [Entailment / Contradiction] <END>"

single_answer_selection_prompt = "You are a medical expert tasked with determining semantic entailment relationships between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate), a section (the premise you will base your judgement in) from a Clinical Trial Report (CTR) and a reasoning chain that explains the rationale behind the relationship between the CTR section and the hypothesis. The task is to classify the relationship as one of the following: Entailment (supported by the CTR) or Contradiction (in conflict with the CTR).\n\nSection Name:\n\n{{Section_id}}\n\nCTR Section: \n\n{{Primary_Section}}\n\nHypothesis: {{Statement}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the CTR Section, the Hypothesis and with special atention to the presented Reasoning Chain, provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the CTR Section and the Hypothesis.\nFinal Answer: "

single_verify_reasoning_prompt = "You are a medical expert tasked with verifying a reasoning chain that determines the semantic entailment relationship between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate), a section (the premise you will base your judgement in) from a Clinical Trial Report (CTR) and a reasoning chain that explains the rationale behind the relationship between the CTR section and the hypothesis. The task is to classify the relationship as one of the following: Entailment (supported by the CTR) or Contradiction (in conflict with the CTR).\n\nSection Name:\n\n{{Section_id}}\n\nCTR Section: \n\n{{Primary_Section}}\n\nHypothesis: {{Statement}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [Entailment / Contradiction] <END>"

single_answer_selection_after_verification_prompt = "You are a medical expert tasked with determining semantic entailment relationships between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate), a section (the premise you will base your judgement in) from a Clinical Trial Report (CTR), an initial reasoning chain that explains the rationale behind the relationship between the CTR section and the hypothesis, and a verification of that reasoning chain. The task is to classify the relationship as one of the following: Entailment (supported by the CTR) or Contradiction (in conflict with the CTR).\n\nSection Name:\n\n{{Section_id}}\n\nCTR Section: \n\n{{Primary_Section}}\n\nHypothesis: {{Statement}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the CTR Section, the Hypothesis, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the CTR Section and the Hypothesis.\nFinal Answer: "

# ====================
# Comparison Prompts
# ====================

replace_pairs = [("a section", "two sections"), ("a section (the premise you will base your judgement in)", "two sections (the premises you will base your judgement in)"), ("{{Primary_Section}}", "{{Primary_Section}}\n\nSecondary CTR Section: \n\n{{Secondary_Section}}")]

def replace_prompt_parts(prompt: str) -> str:
    for old, new in replace_pairs:
        prompt = prompt.replace(old, new)
    return prompt

comparison_baseline_prompt = replace_prompt_parts(single_baseline_prompt)

comparison_reasoning_prompt = replace_prompt_parts(single_reasoning_prompt)

comparison_answer_selection_prompt = replace_prompt_parts(single_answer_selection_prompt)

comparison_verify_reasoning_prompt = replace_prompt_parts(single_verify_reasoning_prompt)

comparison_answer_selection_after_verification_prompt = replace_prompt_parts(single_answer_selection_after_verification_prompt)

# ====================

pos_answers = ["Contradiction", "Entailment"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

def format_sections(sections):
   return '\n'.join(sections) if sections else ""

relevant_keys = ["Section_id", "Primary_Section", "Statement", "Reasoning_Chain", "Verified_Reasoning_Chain"]

def doc_to_text_process(doc, prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", format_sections(doc[key]) if key in ["Primary_Section", "Secondary_Section"] else doc[key])
    return res

def doc_to_text(doc):
    return doc_to_text_process(doc, single_baseline_prompt if "Secondary_Section" not in doc else comparison_baseline_prompt)

def doc_to_text_reasoning(doc):
    return doc_to_text_process(doc, single_reasoning_prompt if "Secondary_Section" not in doc else comparison_reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text_process(doc, single_answer_selection_prompt if "Secondary_Section" not in doc else comparison_answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text_process(doc, single_verify_reasoning_prompt if "Secondary_Section" not in doc else comparison_verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text_process(doc, single_answer_selection_after_verification_prompt if "Secondary_Section" not in doc else comparison_answer_selection_after_verification_prompt)