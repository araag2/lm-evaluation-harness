from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

baseline_prompt = "You are a medical expert tasked with determining wether a Clinical Trial is relevant to a Patient Description. \n\nClinical Trial Title:\n\n{{Title}}\n\nClinical Trial Summary:\n\n{{Summary}}\n\nClinical Trial Detailed Description:\n\n{{Detailed_description}}\n\nClinical Trial Eligibility Criteria:\n\n{{Eligibility}}\n\nPatient Description: \n\n{{Patient}}\n\nBased on the above information, provide your judgement wether the Clinical Trial is relevant to the Patient Description. Please provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

reasoning_prompt = baseline_prompt[:-10] + "\nLet's think step by step:"

pos_answers = ["not relevant", "possibly relevant", "definitely relevant"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Title", "Summary", "Detailed_description", "Eligibility", "Patient"]

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_prompt
    for key in relevant_keys:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)