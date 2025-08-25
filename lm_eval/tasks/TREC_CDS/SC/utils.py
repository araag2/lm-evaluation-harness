baseline_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nBased on the above information, provide your judgement whether the Article is relevant to answer the Clinical Question, within the context of the Patient Description. Provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

reasoning_prompt = baseline_prompt[:-10] + "\nLet's think step by step:"

pos_answers = ["not relevant", "possibly relevant", "definitely relevant"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Title", "Abstract", "Body", "Question", "Patient_Summary"]

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_prompt
    for key in relevant_keys:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)