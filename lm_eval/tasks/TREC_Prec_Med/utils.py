from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

baseline_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a specific Patient Description. \n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n- Disease: {{Disease}}\n- Gene: {{Gene}}\n- Demographic: {{Demographic}}\n- Other: {{Other}}\n\nBased on the above information, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Focus on the patient's disease, gene, demographic, and other factors in relation to the Eligibility Criteria, and other sections of the Clinical Trial Report.\n\nPlease provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

reasoning_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a specific Patient Description. The possible relevance categories are:\n- Not Relevant: The clinical trial does not pertain to the patient's condition, gene, demographic, or other factors.\n- Possibly Relevant: The clinical trial may be relevant to the patient's condition, gene, demographic, or other factors, but there is some uncertainty.\n- Definitely Relevant: The clinical trial is clearly relevant to the patient's condition, gene, demographic, or other factors.\n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n- Disease: {{Disease}}\n- Gene: {{Gene}}\n- Demographic: {{Demographic}}\n- Other: {{Other}}\n\nBased on the above information, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Focus on the patient's disease, gene, demographic, and other factors in relation to the Eligibility Criteria, and other sections of the Clinical Trial Report.\n\nPlease provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [not relevant / possibly relevant / definitely relevant] <END>"

answer_selection_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a specific Patient Description. The possible relevance categories are:\n- Not Relevant: The clinical trial does not pertain to the patient's condition, gene, demographic, or other factors.\n- Possibly Relevant: The clinical trial may be relevant to the patient's condition, gene, demographic, or other factors, but there is some uncertainty.\n- Definitely Relevant: The clinical trial is clearly relevant to the patient's condition, gene, demographic, or other factors.\n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n- Disease: {{Disease}}\n- Gene: {{Gene}}\n- Demographic: {{Demographic}}\n- Other: {{Other}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Clinical Trial information and the Patient Description, and with special atention to the presented Reasoning Chain, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Focus on the patient's disease, gene, demographic, and other factors in relation to the Eligibility Criteria, and other sections of the Clinical Trial Report.\n\nPlease provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nFinal Answer: "

verify_reasoning_prompt = "You are a medical expert tasked with verifying a reasoning chain that determines wether a Clinical Trial is relevant to a specific Patient Description. The possible relevance categories are:\n- Not Relevant: The clinical trial does not pertain to the patient's condition, gene, demographic, or other factors.\n- Possibly Relevant: The clinical trial may be relevant to the patient's condition, gene, demographic, or other factors, but there is some uncertainty.\n- Definitely Relevant: The clinical trial is clearly relevant to the patient's condition, gene, demographic, or other factors.\n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n- Disease: {{Disease}}\n- Gene: {{Gene}}\n- Demographic: {{Demographic}}\n- Other: {{Other}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease find mistakes and critique the logical sense and the conclusion of the Reasoning Chain. Let's think step by step, and after explaining the mistakes you find, provide your final answer in the form: \nVerified Answer: [not relevant / possibly relevant / definitely relevant] <END>"

answer_selection_after_verification_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a specific Patient Description. The possible relevance categories are:\n- Not Relevant: The clinical trial does not pertain to the patient's condition, gene, demographic, or other factors.\n- Possibly Relevant: The clinical trial may be relevant to the patient's condition, gene, demographic, or other factors, but there is some uncertainty.\n- Definitely Relevant: The clinical trial is clearly relevant to the patient's condition, gene, demographic, or other factors.\n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n- Disease: {{Disease}}\n- Gene: {{Gene}}\n- Demographic: {{Demographic}}\n- Other: {{Other}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nFlaws in the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Clinical Trial information and the Patient Description, the initial Reasoning Chain and the possible flaws of the reasoning chain, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Focus on the patient's disease, gene, demographic, and other factors in relation to the Eligibility Criteria, and other sections of the Clinical Trial Report.\n\nPlease provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nFinal Answer: "

pos_answers = ["not relevant", "possibly relevant", "definitely relevant"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Title", "Summary", "Detailed_description", "Eligibility", "Disease", "Gene", "Demographic", "Reasoning_Chain", "Verified_Reasoning_Chain"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", doc[key][:1000] + doc[key][-1000:]if len(doc[key]) > 2000 else doc[key])
    res = res.replace("{{Other}}", doc["Other"][:1000])  if "Other" in doc else res.replace("\n- Other: {{Other}}", "")
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text(doc, answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text(doc, verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text(doc, answer_selection_after_verification_prompt)