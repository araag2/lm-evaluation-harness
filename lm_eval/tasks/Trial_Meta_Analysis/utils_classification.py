# utils_classification.py
"""
Prompt and doc_to_text functions for outcome type classification (binary/continuous).
"""

baseline_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to determine whether the outcome of a given trial is binary or continuous based on the provided article context.\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nPlease output 'binary' or 'continuous' corresponding to the correct option.\nAnswer: "

reasoning_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to determine whether the outcome of a given trial is binary or continuous based on the provided article context. The possible outcome types are:\n- Binary: The outcome has two distinct categories or states (e.g., success/failure, yes/no).\n- Continuous: The outcome can take any value within a range and is not limited to specific categories (e.g., weight, blood pressure).\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [binary / continuous] <END>"

answer_selection_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to determine whether the outcome of a given trial is binary or continuous based on the provided article context. The possible outcome types are:\n- Binary: The outcome has two distinct categories or states (e.g., success/failure, yes/no).\n- Continuous: The outcome can take any value within a range and is not limited to specific categories (e.g., weight, blood pressure).\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Article, the Outcome and with special attention to the presented Reasoning Chain, provide your judgement (binary or continuous), corresponding to the correct option. Be as accurate as possible.\nFinal Answer: "

verify_reasoning_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to verify a reasoning chain that determines whether the outcome of a given trial is binary or continuous based on the provided article context. The possible outcome types are:\n- Binary: The outcome has two distinct categories or states (e.g., success/failure, yes/no).\n- Continuous: The outcome can take any value within a range and is not limited to specific categories (e.g., weight, blood pressure).\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease find mistakes and critique the logical sense and the conclusion of the Reasoning Chain. Let's think step by step, and after explaining the mistakes you find, provide your final answer in the form: \nVerified Answer: [binary / continuous] <END>"

answer_selection_after_verification_prompt = "You are an expert medical assistant specialized in judging the type of outcome from a randomized controlled trial. Your task is to determine whether the outcome of a given trial is binary or continuous based on the provided article context. The possible outcome types are:\n- Binary: The outcome has two distinct categories or states (e.g., success/failure, yes/no).\n- Continuous: The outcome can take any value within a range and is not limited to specific categories (e.g., weight, blood pressure).\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nReasoning Chain: {{Reasoning_Chain}}\n\nFlaws in the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Article, the Outcome, the initial Reasoning Chain and the possible flaws of the reasoning chain, provide your judgement (binary or continuous), corresponding to the correct option. Be as accurate as possible.\nFinal Answer: "

pos_answers = ["binary", "continuous"]

def label_to_index(doc) -> int:
    return pos_answers.index(doc["outcome_type"])

self_refine_feedback_prompt = "You are an expert medical assistant reviewing a reasoning chain that classifies the outcome type (binary or continuous) of a randomized controlled trial.\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nReasoning Chain: {{Reasoning_Chain}}\n\nCritique this reasoning chain. Identify any logical gaps, unsupported conclusions, unclear steps, or factual inaccuracies.\n\nProvide at least 2-3 specific points of improvement.\nFeedback: "

self_refine_refine_prompt = "You are an expert medical assistant tasked with improving a reasoning chain that classifies the outcome type (binary or continuous) of a randomized controlled trial. The possible outcome types are:\n- Binary: The outcome has two distinct categories or states (e.g., success/failure, yes/no).\n- Continuous: The outcome can take any value within a range and is not limited to specific categories (e.g., weight, blood pressure).\n\nArticle:\n\n{{Context}}\n\nQuestion: Based on the article, is the outcome of {{outcome}} from a randomized controlled trial a binary or continuous type?\n\nReasoning Chain: {{Reasoning_Chain}}\n\nFeedback: {{Feedback}}\n\nGiven the feedback above, revise the reasoning chain to:\n1 - Fill logical gaps with evidence from the article.\n2 - Clarify unclear reasoning steps.\n3 - Ensure all conclusions are directly supported.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [binary / continuous] <END>"

relevant_keys = ["Context", "outcome", "intervention", "comparator", "outcome", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            value = doc[key]
            value = value[:2000] + value[-2000:] if len(value) > 4000 else value
            res = res.replace(f"{{{{{key}}}}}", value)
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text(doc, answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text(doc, verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text(doc, answer_selection_after_verification_prompt)

def doc_to_text_self_refine_feedback(doc):
    return doc_to_text(doc, self_refine_feedback_prompt)

def doc_to_text_self_refine_refine(doc):
    return doc_to_text(doc, self_refine_refine_prompt)