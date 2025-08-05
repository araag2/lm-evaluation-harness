medical_baseline_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's medical profile (written in medical language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Medical-Language}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the Clinical Trial and the Patient Description.\nAnswer: "

patient_baseline_prompt = "You are an medical expert AI assistant specialized in natural language inference for clinical trial recruitment. The task is to determine if a patient's self-described medical profile (written in their own everyday language) entails or contradicts the eligibility criteria of a given Clinical Trial Report.\n\nClinical Trial Report Context: {{CTR_Context}}\n\nPatient Description: {{Description_Patient-Language}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the Clinical Trial and the Patient Description.\nAnswer: "

def doc_to_text(doc, res, opts):
    for key in opts:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def medical_lang_doc_to_text(doc):
   return doc_to_text(doc, medical_baseline_prompt, ["CTR_Context", "Description_Medical-Language"])

def patient_lang_doc_to_text(doc):
    return doc_to_text(doc, patient_baseline_prompt, ["CTR_Context", "Description_Patient-Language"])