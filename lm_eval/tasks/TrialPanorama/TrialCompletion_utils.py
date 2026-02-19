TC_baseline_prompt="You are a medical expert tasked with predicting the outcome of a clinical trial. Carefully review the provided information and determine whether the trial is likely to complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nPlease provide your answer in the format **Answer:** followed by 'complete' if the trial is likely to complete successfully based on the provided information, or 'terminate' if the trial is likely to be terminated.\nIf you believe the trial is likely to be terminated, please also provide a 'terminate_type' indicating the primary reason for termination from the following options: 'safety issues', 'enrollment issues', 'lack of efficacy', 'regulatory/approval' or 'feasibility'.\n**Answer:** "

TC_reasoning_prompt="You are a medical expert tasked with predicting the outcome of a clinical trial. Carefully review the provided information and determine whether the trial is likely to complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nPlease provide your answer in the format **Answer:** followed by 'complete' if the trial is likely to complete successfully based on the provided information, or 'terminate' if the trial is likely to be terminated.\nIf you believe the trial is likely to be terminated, please also provide a 'terminate_type' indicating the primary reason for termination from the following options: 'safety issues', 'enrollment issues', 'lack of efficacy', 'regulatory/approval' or 'feasibility'.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [complete / terminate] <END>"

TC_answer_selection_prompt="You are a medical expert tasked with predicting the outcome of a clinical trial. Carefully review the provided information and determine whether the trial is likely to complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease provide your answer in the format **Answer:** followed by 'complete' if the trial is likely to complete successfully, or 'terminate' if the trial is likely to be terminated. Pay special attention to the presented Reasoning Chain.\nFinal Answer: "

TC_verify_reasoning_prompt="You are a medical expert tasked with predicting the outcome of a clinical trial. Carefully review the provided information and determine whether the trial is likely to complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense and supports the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [complete / terminate] <END>"

TC_answer_selection_after_verification_prompt="You are a medical expert tasked with predicting the outcome of a clinical trial. Carefully review the provided information and determine whether the trial is likely to complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nPlease provide your answer in the format **Answer:** followed by 'complete' if the trial is likely to complete successfully, or 'terminate' if the trial is likely to be terminated. Consider both the initial Reasoning Chain and its Verification.\nFinal Answer: "

pos_answers = ['complete', 'terminate']

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

TC_self_refine_feedback_prompt="You are a medical expert reviewing a reasoning chain that predicts whether a clinical trial will complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nCritique this reasoning chain. Identify any logical gaps, unsupported conclusions, unclear steps, or factual inaccuracies.\n\nProvide at least 2-3 specific points of improvement.\nFeedback: "

TC_self_refine_refine_prompt="You are a medical expert tasked with improving a reasoning chain that predicts whether a clinical trial will complete successfully or be terminated.\n\n**Study Title:** {{Study_Title}}\n\n**Study Design:**\n{{Study_Design}}\n\n**Arms:**\n{{Arms}}\n\n{{Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nFeedback: {{Feedback}}\n\nGiven the feedback above, revise the reasoning chain to:\n1 - Fill logical gaps with evidence from the study information (Study Design, Arms, and Criteria).\n2 - Clarify unclear reasoning steps.\n3 - Ensure all conclusions are directly supported.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [complete / terminate] <END>"

relevant_keys = ["Study_Title", "Study_Design", "Arms", "Criteria", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

def TC_doc_to_text(doc, prompt=TC_baseline_prompt):
    res = prompt
    for key in relevant_keys:
      if key in doc:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def TC_doc_to_text_reasoning(doc):
    return TC_doc_to_text(doc, TC_reasoning_prompt)

def TC_doc_to_text_answer_selection(doc):
    return TC_doc_to_text(doc, TC_answer_selection_prompt)

def TC_doc_to_text_verify_reasoning(doc):
    return TC_doc_to_text(doc, TC_verify_reasoning_prompt)

def TC_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return TC_doc_to_text(doc, TC_answer_selection_after_verification_prompt)

def TC_doc_to_text_self_refine_feedback(doc):
    return TC_doc_to_text(doc, TC_self_refine_feedback_prompt)

def TC_doc_to_text_self_refine_refine(doc):
    return TC_doc_to_text(doc, TC_self_refine_refine_prompt)