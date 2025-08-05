baseline_prompt = "You are a medical expert specialized in biomedical question answering focused on research-level queries from PubMed articles. You task is given the context of a PubMed Article and a Question, answer the question based on the article.\n\nPubMed Article Information: \n\n{{Context}}\n\nQuestion: {{Question}}\n\nPlease provide your judgement in a single word (Yes, No or Maybe), answering the question based on the Pubmed Article Information.\nAnswer: "

def format_context(contexts):
   return "\n".join([f"- <{label}> : {context}" for label, context in contexts])

def doc_to_text(doc):
    res = baseline_prompt.replace("{{Context}}", format_context(doc["Context"]))
    return res.replace("{{Question}}", doc["Question"])