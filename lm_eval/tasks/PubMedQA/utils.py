baseline_prompt = "You are a medical expert specialized in biomedical question answering focused on research-level queries from PubMed articles. You task is given the context of a PubMed Article and a Question, answer the question based on the article.\n\nPubMed Article Information: \n\n{{Context}}\n\nQuestion: {{Question}}\n\nPlease provide your judgement in a single word (Yes, No or Maybe), answering the question based on the Pubmed Article Information.\nAnswer: "

reasoning_prompt = "You are a medical expert specialized in biomedical question answering focused on research-level queries from PubMed articles. You task is given the context of a PubMed Article and a Question, answer the question based on the article.\n\nPubMed Article Information: \n\n{{Context}}\n\nQuestion: {{Question}}\n\nPlease provide your judgement in a single word (Yes, No or Maybe), answering the question based on the Pubmed Article Information. Let's think step by step, and at the very end write your answer in the form: \nAnswer: [Yes / No / Maybe] <END>"

answer_selection_prompt = "You are a medical expert specialized in biomedical question answering focused on research-level queries from PubMed articles. You task is given the context of a PubMed Article, a Question and a Reasoning Chain, answer the question based on the article and the reasoning chain.\n\nPubMed Article Information: \n\n{{Context}}\n\nQuestion: {{Question}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the PubMed Article Information, the Question and with special atention to the presented Reasoning Chain, provide your judgement in a single word (Yes, No or Maybe), answering the question based on the Pubmed Article Information.\nFinal Answer: "

verify_reasoning_prompt = "You are a medical expert specialized in biomedical question answering focused on research-level queries from PubMed articles. You task is given the context of a PubMed Article, a Question and a Reasoning Chain, verify the reasoning chain and answer the question based on the article and the reasoning chain.\n\nPubMed Article Information: \n\n{{Context}}\n\nQuestion: {{Question}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [Yes / No / Maybe] <END>"

answer_selection_after_verification_prompt = "You are a medical expert specialized in biomedical question answering focused on research-level queries from PubMed articles. You task is given the context of a PubMed Article, a Question, a Reasoning Chain and a Verification of the Reasoning Chain, answer the question based on the article, the reasoning chain and its verification.\n\nPubMed Article Information: \n\n{{Context}}\n\nQuestion: {{Question}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the PubMed Article Information, the Question, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your judgement in a single word (Yes, No or Maybe), answering the question based on the Pubmed Article Information.\nFinal Answer: "

pos_answers = ["No", "Maybe", "Yes"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

def format_context(contexts):
    return "\n".join([f"- <{label}> : {context}" for label, context in contexts])

relevant_keys = ["Context", "Question"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
      res = res.replace(f"{{{{{key}}}}}", doc[key] if key != "Context" else format_context(doc[key]))
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    relevant_keys.append("Reasoning_Chain")
    return doc_to_text(doc, answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    relevant_keys.append("Reasoning_Chain")
    return doc_to_text(doc, verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    relevant_keys += ["Reasoning_Chain", "Verified_Reasoning_Chain"]
    return doc_to_text(doc, answer_selection_after_verification_prompt)