SS_baseline_prompt="You are a medical expert tasked with evaluating clinical trials for an eligibility study. Carefully review the provided information and determine whether the Candidate Study should be included or excluded based on the predefined criteria.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\n Please provide your answer whether the Candidate Study should be 'included' or 'excluded' in an eligibility study based on the review's predefined criteria, namely the Background, Objective, and Selection Criteria provided.\nAnswer: "

SS_reasoning_prompt="You are a medical expert tasked with evaluating clinical trials for an eligibility study. Carefully review the provided information and determine whether the Candidate Study should be included or excluded based on the predefined criteria.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\n Please provide your answer whether the Candidate Study should be 'included' or 'excluded' in an eligibility study based on the review's predefined criteria, namely the Background, Objective, and Selection Criteria provided. Be as accurate as possible.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [A / B / C / D] <END>"

SS_answer_selection_prompt="You are a medical expert tasked with evaluating clinical trials for an eligibility study. Carefully review the provided information and determine whether the Candidate Study should be included or excluded based on the predefined criteria.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease provide your answer whether the Candidate Study should be 'included' or 'excluded' in an eligibility study based on the review's predefined criteria, namely the Background, Objective, Selection Criteria, and with special atention to the presented Reasoning Chain.\nFinal Answer: "

SS_verify_reasoning_prompt="You are a medical expert tasked with evaluating clinical trials for an eligibility study. Carefully review the provided information and determine whether the Candidate Study should be included or excluded based on the predefined criteria.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and support the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form: \nVerified Answer: [included / excluded] <END>"

SS_answer_selection_after_verification_prompt="You are a medical expert tasked with evaluating clinical trials for an eligibility study. Carefully review the provided information and determine whether the Candidate Study should be included or excluded based on the predefined criteria.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nPlease provide your answer whether the Candidate Study should be 'included' or 'excluded' in an eligibility study based on the review's predefined criteria, namely the Background, Objective, Selection Criteria, the initial Reasoning Chain and the Verification of the Reasoning Chain provided.\nFinal Answer: "

pos_answers = ['excluded', 'included']

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

SS_self_refine_feedback_prompt="You are a medical expert reviewing a reasoning chain about whether a candidate study should be included or excluded in an eligibility study.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\nReasoning Chain: {{Reasoning_Chain}}\n\nCritique this reasoning chain. Identify any logical gaps, unsupported conclusions, unclear steps, or factual inaccuracies.\n\nProvide at least 2-3 specific points of improvement.\nFeedback: "

SS_self_refine_refine_prompt="You are a medical expert tasked with improving a reasoning chain about whether a candidate study should be included or excluded in an eligibility study.\n**Background:** {{Background}}\n**Objective:** {{Objective}}\n**Selection Criteria:** {{Selection_Criteria}}\n**Candidate Study:**\n- **PMID: {{trial_pmid}}** - {{Trial_Title}} \n- **Abstract:** {{Trial_Abstract}}\nReasoning Chain: {{Reasoning_Chain}}\nFeedback: {{Feedback}}\n\nGiven the feedback above, revise the reasoning chain to:\n1 - Fill logical gaps with evidence from the study information (Background, Objective, Selection Criteria).\n2 - Clarify unclear reasoning steps.\n3 - Ensure all conclusions are directly supported.\nLet's think step by step, and at the very end write your answer in the form: \nAnswer: [included / excluded] <END>"

relevant_keys = ["Background", "Objective", "Selection_Criteria", "trial_pmid", "Trial_Title", "Trial_Abstract", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

def SS_doc_to_text(doc, prompt = SS_baseline_prompt):
    res = prompt
    for key in relevant_keys:
      if key in doc:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res

def SS_doc_to_text_reasoning(doc):
    return SS_doc_to_text(doc, SS_reasoning_prompt)

def SS_doc_to_text_answer_selection(doc):
    return SS_doc_to_text(doc, SS_answer_selection_prompt)

def SS_doc_to_text_verify_reasoning(doc):
    return SS_doc_to_text(doc, SS_verify_reasoning_prompt)

def SS_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return SS_doc_to_text(doc, SS_answer_selection_after_verification_prompt)

def SS_doc_to_text_self_refine_feedback(doc):
    return SS_doc_to_text(doc, SS_self_refine_feedback_prompt)

def SS_doc_to_text_self_refine_refine(doc):
    return SS_doc_to_text(doc, SS_self_refine_refine_prompt)