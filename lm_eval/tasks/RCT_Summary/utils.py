baseline_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract of this medical article, generate the Conclusion Section. Only output the newly generated section.\nAnswer: "

reasoning_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract of this medical article, generate the Conclusion Section. Only output the newly generated section. Be as accurate as possible.\nLet's think step by step: "

relevant_keys = ["Title", "Abstract"]

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_prompt
    for key in relevant_keys:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)