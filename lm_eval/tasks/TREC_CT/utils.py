from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

baseline_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a Patient Description. \n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n{{Patient}}\n\nBased on the above information, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Please provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

def doc_to_text(doc):
    res = baseline_prompt
    for key in ["Title", "Summary", "Detailed_description", "Eligibility", "Patient"]:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res