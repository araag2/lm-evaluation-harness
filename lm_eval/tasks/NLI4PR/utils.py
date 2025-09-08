# =============================
# Medical Language Prompts
# =============================

medical_baseline_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's medical profile (written in medical language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Medical-Language}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the Clinical Trial and the Patient Description.\nAnswer: "

medical_reasoning_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's medical profile (written in medical language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Medical-Language}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the Clinical Trial and the Patient Description. Let's think step by step, and at the very end write your answer in the form: \nAnswer: [Entailment / Contradiction] <END>"

answer_selection_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's medical profile (written in medical language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Medical-Language}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Clinical Trial Report Context, the Patient Description and with special atention to the presented Reasoning Chain, provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the Clinical Trial and the Patient Description.\nFinal Answer: "

verify_reasoning_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's medical profile (written in medical language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Medical-Language}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [Entailment / Contradiction] <END>"

answer_selection_after_verification_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's medical profile (written in medical language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Medical-Language}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Clinical Trial Report Context, the Patient Description, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the Clinical Trial and the Patient Description.\nFinal Answer: "

# =============================
# Patient Language Prompts
# =============================

replace_pairs = [("medical language", "their own everyday language"), ("medical profile (written in medical language)", "self-described medical profile (written in their own everyday language)"), ("{{Description_Medical-Language}}", "{{Description_Patient-Language}}")]

def replace_prompt_parts(prompt: str) -> str:
    for old, new in replace_pairs:
        prompt = prompt.replace(old, new)
    return prompt

patient_baseline_prompt = replace_prompt_parts(medical_baseline_prompt)

patient_reasoning_prompt = replace_prompt_parts(medical_reasoning_prompt)

patient_answer_selection_prompt = replace_prompt_parts(answer_selection_prompt)

patient_verify_reasoning_prompt = replace_prompt_parts(verify_reasoning_prompt)

patient_answer_selection_after_verification_prompt = replace_prompt_parts(answer_selection_after_verification_prompt)

# ====================

pos_answers = ["Contradiction", "Entailment"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["CTR_Context", "Description_Medical-Language", "Description_Patient-Language", "Reasoning_Chain", "Verified_Reasoning_Chain"]

def doc_to_text_process(doc, prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text(doc):
    return doc_to_text_process(doc, patient_baseline_prompt if "Description_Patient-Language" in doc else medical_baseline_prompt)

def doc_to_text_reasoning(doc):
    return doc_to_text_process(doc, patient_reasoning_prompt if "Description_Patient-Language" in doc else medical_reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text_process(doc, patient_answer_selection_prompt if "Description_Patient-Language" in doc else answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text_process(doc, patient_verify_reasoning_prompt if "Description_Patient-Language" in doc else verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text_process(doc, patient_answer_selection_after_verification_prompt if "Description_Patient-Language" in doc else answer_selection_after_verification_prompt)