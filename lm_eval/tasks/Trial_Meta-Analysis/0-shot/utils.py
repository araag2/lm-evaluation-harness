baseline_type_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to determine whether the outcome of a given trial is binary or continuous based on the provided article context.\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nPlease output 'binary' or 'continuous' corresponding to the correct option.\nAnswer: "

def type_doc_to_text(doc):
    res = baseline_type_prompt
    for key in ["Context", "outcome"]:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res  