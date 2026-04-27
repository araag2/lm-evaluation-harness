baseline_prompt = """
You are a medical expert tasked with answering questions by using your medical knowledge and reasoning skills. Carefully read the **Context:** and the **Question:**, analyzing the information provided in the Context to determine the correct answer to the Question. 

Determine the correct answer based on your medical expertise. Be as accurate as possible.

**Context:** {{Context}}

**Question:** {{Question}}

Please provide your judgement in a single word (Yes or No), corresponding to the correct option that answers the Question based on the Context.
Answer: """



reasoning_prompt = """You are a medical expert tasked with answering questions by using your medical knowledge and reasoning skills. Carefully read the **Context:** and the **Question:**, analyzing the information provided in the Context to determine the correct answer to the Question. 

Determine the correct answer based on your medical expertise. Be as accurate as possible.

**Context:** {{Context}}

**Question:** {{Question}}

Let's think step by step, and at the very end write your answer in the form: \nAnswer: [Yes / No / Maybe] <END>"""

answer_selection_prompt = """You are a medical expert tasked with answering questions by using your medical knowledge and reasoning skills. Carefully read the **Context:**, the **Question:**, and the **Reasoning Chain:**, analyzing the information provided in the Context to determine the correct answer to the Question. 

**Context:** {{Context}}

**Question:** {{Question}}

**Reasoning Chain:** {{Reasoning_Chain}}

Given the Context, the Question and with special atention to the presented Reasoning Chain, provide your judgement in a single word (Yes or No), answering the question based on the Context.

Final Answer: """

verify_reasoning_prompt = """You are a medical expert tasked with answering questions by using your medical knowledge and reasoning skills. Carefully read the **Context:**, the **Question:**, and the **Reasoning Chain:**, analyzing the information provided in the Context to determine the validity of the reasoning chain in relation to the Question and the Context.

**Context:** {{Context}}

**Question:** {{Question}}

**Reasoning Chain:** {{Reasoning_Chain}}

Please verify if the Reasoning Chain makes logical sense, and supports the correct conclusion.

Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [Yes / No] <END>"""

answer_selection_after_verification_prompt = """You are a medical expert tasked with answering questions by using your medical knowledge and reasoning skills. Carefully read the **Context:**, the **Question:**, the **Reasoning Chain:**, and the **Verification of the Reasoning Chain:**, analyzing the information provided to determine the correct answer to the Question.

**Context:** {{Context}}

**Question:** {{Question}}

**Reasoning Chain:** {{Reasoning_Chain}}

**Verification of the Reasoning Chain:** {{Verified_Reasoning_Chain}}

Given the Context, the Question, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your judgement in a single word (Yes or No), answering the question based on the Context.

Answer: """

pos_answers = ["No", "Yes"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Context", "Question", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

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