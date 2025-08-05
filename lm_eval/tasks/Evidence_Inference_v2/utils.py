baseline_prompt = "Article: {{Article_Content}}\nQuestion: {{Question}}\nPlease output 'significantly increase', 'no significant difference' or 'significantly decrease' corresponding to the correct option. Be as accurate as possible.\nAnswer: "
  
def doc_to_text(doc):
    res = baseline_prompt
    for key in ['Article_Content', 'Question']:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res