baseline_prompt="You are a medical expert tasked with answering multiple-choice questions by using your medical knowledge and reasoning skills.\n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nPlease provide your answer in the format of a single letter (A, B, C, or D) corresponding to the correct option.\nAnswer: "

reasoning_prompt = baseline_prompt[:-10] + "\nLet's think step by step:"

pos_answers = ['A', 'B', 'C', 'D']

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Question", "Option_A", "Option_B", "Option_C", "Option_D"]

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_prompt
    for key in relevant_keys:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)