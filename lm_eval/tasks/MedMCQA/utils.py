baseline_prompt="You are a medical expert tasked with answering multiple-choice questions by using your medical knowledge and reasoning skills. Carefully read the Question: and the provided Options:. \n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nPlease provide your answer to the Question using a single-letter corresponding to the correct Option (A, B, C, or D). Be as accurate as possible.\nAnswer: "

reasoning_prompt="You are a medical expert tasked with answering multiple-choice questions by using your medical knowledge and reasoning skills. Carefully read the Question: and the provided Options:. \n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nPlease provide your answer to the Question using a single-letter corresponding to the correct Option (A, B, C, or D). Be as accurate as possible.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [A / B / C / D] <END>"

answer_selection_prompt="You are a medical expert tasked with answering multiple-choice questions by using your medical knowledge and reasoning skills. Carefully read the Question:, the provided Options: and the Reasoning Chain: that leads to the answer. \n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Question, the Options and with special atention to the presented Reasoning Chain, provide your answer to the Question using a single-letter corresponding to the correct Option (A, B, C, or D). Be as accurate as possible.\nFinal Answer: "

verify_reasoning_prompt="You are a medical expert tasked with verifying a reasoning chain that leads to the answer of a multiple-choice question. Carefully read the Question:, the provided Options: and the Reasoning Chain: that needs to be verified. \n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [A / B / C / D] <END>"

answer_selection_after_verification_prompt="You are a medical expert tasked with answering multiple-choice questions by using your medical knowledge and reasoning skills. Carefully read the Question:, the provided Options:, the initial Reasoning Chain: and the Verification of the Reasoning Chain:. \n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Question, the Options, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your answer to the Question using a single-letter corresponding to the correct Option (A, B, C, or D). Be as accurate as possible.\nFinal Answer: "

pos_answers = ['A', 'B', 'C', 'D']

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Question", "Option_A", "Option_B", "Option_C", "Option_D", "Reasoning_Chain", "Verified_Reasoning_Chain"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
      if key in doc:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text(doc, answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text(doc, verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text(doc, answer_selection_after_verification_prompt)