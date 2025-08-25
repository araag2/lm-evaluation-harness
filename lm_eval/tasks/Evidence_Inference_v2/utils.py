baseline_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nPlease output 'significantly increase', 'no significant difference' or 'significantly decrease' corresponding to the correct option. Be as accurate as possible.\nAnswer: "

reasoning_prompt ="Article: {{Article_Content}}\nQuestion: {{Question}}\nPlease output 'significantly increase', 'no significant difference' or 'significantly decrease' corresponding to the correct option. Be as accurate as possible.\nLet's think step by step: "

pos_answers = ["significantly decreased", "no significant difference", "significantly increased"]

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_prompt
    for key in ['Article_Content', 'Question']:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)