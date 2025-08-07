from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

baseline_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a specific Patient Description. \n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n- Disease: {{Disease}}\n- Gene: {{Gene}}\n- Demographic: {{Demographic}}\n- Other: {{Other}}\n\nBased on the above information, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Focus on the patient's disease, gene, demographic, and other factors in relation to the Eligibility Criteria, and other sections of the Clinical Trial Report.\n\nPlease provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

def doc_to_text(doc):
    res = baseline_prompt
    for key in ["Title", "Summary", "Detailed_description", "Eligibility", "Disease", "Gene", "Demographic"]:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    res = res.replace("{{Other}}", doc["Other"])  if "Other" in doc else res.replace("\n- Other: {{Other}}", "")
    return res