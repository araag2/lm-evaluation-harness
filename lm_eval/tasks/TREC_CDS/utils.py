baseline_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nBased on the above information, provide your judgement whether the Article is relevant to answer the Clinical Question, within the context of the Patient Description. Provide your judgement in short form, using 'not relevant', 'possibly relevant' or 'definitely relevant'.\nAnswer: "

reasoning_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question. The possible relevance judgements are:\n- Not Relevant: The article does not provide information that helps answer the clinical question in the context of the patient description.\n- Possibly Relevant: The article provides some information that might help answer the clinical question, but it is not definitive or directly applicable.\n- Definitely Relevant: The article provides clear and direct information that helps answer the clinical question in the context of the patient description.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [not relevant / possibly relevant / definitely relevant] <END>"

answer_selection_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question. The possible relevance judgements are:\n- Not Relevant: The article does not provide information that helps answer the clinical question in the context of the patient description.\n- Possibly Relevant: The article provides some information that might help answer the clinical question, but it is not definitive or directly applicable.\n- Definitely Relevant: The article provides clear and direct information that helps answer the clinical question in the context of the patient description.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Article, the Clinical Question, the Patient Description and with special attention to the presented Reasoning Chain, provide your judgement (not relevant, possibly relevant or definitely relevant), corresponding to the correct option. Be as accurate as possible.\nFinal Answer: "

verify_reasoning_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question. The possible relevance judgements are:\n- Not Relevant: The article does not provide information that helps answer the clinical question in the context of the patient description.\n- Possibly Relevant: The article provides some information that might help answer the clinical question, but it is not definitive or directly applicable.\n- Definitely Relevant: The article provides clear and direct information that helps answer the clinical question in the context of the patient description.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease find mistakes and critique the logical sense and the conclusion of the Reasoning Chain. Let's think step by step, and after explaining the mistakes you find, provide your final answer in the form: \nVerified Answer: [not relevant / possibly relevant / definitely relevant] <END>"

answer_selection_after_verification_prompt = "You are a medical expert tasked with performing clinical decision support to a doctor, by reading a Patient Description and a article of biomedical literature, determining wether that article is relevant to the Patient Description and the Question. The possible relevance judgements are:\n- Not Relevant: The article does not provide information that helps answer the clinical question in the context of the patient description.\n- Possibly Relevant: The article provides some information that might help answer the clinical question, but it is not definitive or directly applicable.\n- Definitely Relevant: The article provides clear and direct information that helps answer the clinical question in the context of the patient description.\n\nArticle Title:\n\n{{Title}}\n\nArticle Abstract:\n\n{{Abstract}}\n\nArticle Body:\n\n{{Body}}\n\nClinical Question:\n\n{{Question}}\n\nPatient Description:\n\n{{Patient_Summary}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nFlaws in the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Article, the Clinical Question, the Patient Description, the initial Reasoning Chain and the possible flaws of the reasoning chain, provide your judgement (not relevant, possibly relevant or definitely relevant), corresponding to the correct option. Be as accurate as possible.\nFinal Answer: "

pos_answers = ["not relevant", "possibly relevant", "definitely relevant"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Title", "Abstract", "Body", "Question", "Patient_Summary", "Reasoning_Chain", "Verified_Reasoning_Chain"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", doc[key][:1000] + doc[key][-1000:]if len(doc[key]) > 2000 else doc[key])
        else:
            res = res.replace(f"{{{{{key}}}}}", "")
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text(doc, answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text(doc, verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text(doc, answer_selection_after_verification_prompt)

def process_docs(dataset):
    dataset_relevant = dataset.filter(lambda doc: doc["Label"] == "definitely relevant" or doc["Label"] == "possibly relevant")
    relevant_size = len(dataset_relevant)
    dataset_irrelevant = dataset.filter(lambda doc: doc["Label"] == "not relevant").shuffle(seed=42).select(range(relevant_size))
    dataset_balanced = dataset_relevant.concatenate(dataset_irrelevant).shuffle(seed=42)
    return dataset_balanced