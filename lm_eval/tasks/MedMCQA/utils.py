baseline_prompt="You are a medical expert tasked with answering multiple-choice questions by using your medical knowledge and reasoning skills. Carefully read the Question: and the provided Options:. \n\nDetermine the correct answer based on your medical expertise. Be as accurate as possible.\n\nQuestion: {{Question}}\n\nOptions:\nA) {{Option_A}}\nB) {{Option_B}}\nC) {{Option_C}}\nD) {{Option_D}}\n\nPlease provide your answer in the format of a single letter (A, B, C, or D) corresponding to the correct option.\nAnswer: "

def doc_to_text(doc):
    res = baseline_prompt
    for key in ["Question", "Option_A", "Option_B", "Option_C", "Option_D"]:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res