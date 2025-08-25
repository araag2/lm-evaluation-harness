baseline_type_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to determine whether the outcome of a given trial is binary or continuous based on the provided article context.\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nPlease output 'binary' or 'continuous' corresponding to the correct option.\nAnswer: "

reasoning_prompt = baseline_type_prompt[:-10] + "\nLet's think step by step:"

pos_answers = ["binary", "continuous"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

relevant_keys = ["Context", "outcome"]

def doc_to_text(doc, reasoning = False):
    res = reasoning_prompt if reasoning else baseline_type_prompt
    for key in relevant_keys:
      res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning=True)