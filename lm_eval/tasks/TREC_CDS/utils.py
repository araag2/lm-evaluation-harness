baseline_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nBased on the above information, provide your judgement whether the Article is relevant to answer the Clinical Question, within the context of the Patient Description. Provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

def doc_to_text(doc):
    res = baseline_prompt
    for key in ["Title", "Abstract", "Body", "Question", "Patient_Summary"]:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res