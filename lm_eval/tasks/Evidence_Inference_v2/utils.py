baseline_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nPlease output 'significantly increase', 'no significant difference' or 'significantly decrease' corresponding to the correct option. Be as accurate as possible.\nAnswer: "

reasoning_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nPlease output the correct answer to the question (significantly increase, no significant difference or significantly decrease), corresponding to the correct option.  Be as accurate as possible.\nLet's think step by step, and at the very end write your answer in the form:\nAnswer: [significantly increased / no significant difference / significantly decreased] <END>"

answer_selection_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nReasoning Chain: {{Reasoning_Chain}}\nGiven the Article, the Question and with special atention to the presented Reasoning Chain, please output 'significantly increase', 'no significant difference' or 'significantly decrease' corresponding to the correct option. Be as accurate as possible.\nFinal Answer: "

verify_reasoning_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nReasoning Chain: {{Reasoning_Chain}}\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form:\nVerified Answer: [significantly increased / no significant difference / significantly decreased] <END>"

answer_selection_after_verification_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nReasoning Chain: {{Reasoning_Chain}}\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\nGiven the Article, the Question, the initial Reasoning Chain and the Verification of the Reasoning Chain, please output 'significantly increase', 'no significant difference' or 'significantly decrease' corresponding to the correct option. Be as accurate as possible.\nFinal Answer: "

pos_answers = ["significantly decreased", "no significant difference", "significantly increased"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Article_Content", "Question"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
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