# Regression Task
regression_baseline_prompt = """You are a medical expert tasked with determining the duration of a Clinical Trial. Your task is to evaluate how many years the trial will last based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

**Trial Title:** 
{{Title}}

**Trial Summary:** 
{{Summary}}

**Trial Keywords:** {{Keywords}}

**Phase:** {{Phase}}

**Condition:** {{Condition}}

**Condition Keywords:** {{Condition_Keywords}}

**ICD Code:** {{icdcode}}

**Eligibility Criteria:** 
{{Eligibility_Criteria}}

**Interventions:** 
{{Interventions}}

**Study Design:** 
{{Study_Design}}

Based on the above information, provide your judgement on how many years the Clinical Trial will last. Please provide your answer as a numerical value representing the number of years, with up to six decimal places.
Answer: """

regression_reasoning_prompt = """You are a medical expert tasked with determining the duration of a Clinical Trial. Your task is to evaluate how many years the trial will last based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

**Trial Title:** 
{{Title}}

**Trial Summary:** 
{{Summary}}

**Trial Keywords:** {{Keywords}}

**Phase:** {{Phase}}

**Condition:** {{Condition}}

**Condition Keywords:** {{Condition_Keywords}}

**ICD Code:** {{icdcode}}

**Eligibility Criteria:** 
{{Eligibility_Criteria}}

**Interventions:** 
{{Interventions}}

**Study Design:** 
{{Study_Design}}

Based on the above information, provide your judgement on how many years the Clinical Trial will last. Please provide your answer as a numerical value representing the number of years, with up to six decimal places.
Let's think step by step, and at the very end write your answer in the form: 
Answer: [numerical value] <END>"""

answer_selection_prompt = """You are a medical expert tasked with determining the duration of a Clinical Trial. Your task is to evaluate how many years the trial will last based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

**Trial Title:** 
{{Title}}

**Trial Summary:** 
{{Summary}}

**Trial Keywords:** {{Keywords}}

**Phase:** {{Phase}}

**Condition:** {{Condition}}

**Condition Keywords:** {{Condition_Keywords}}

**ICD Code:** {{icdcode}}

**Eligibility Criteria:** 
{{Eligibility_Criteria}}

**Interventions:** 
{{Interventions}}

**Study Design:** 
{{Study_Design}}

Reasoning Chain: {{Reasoning_Chain}}

Given the trial information and the reasoning chain, provide your judgement on how many years the Clinical Trial will last. Please provide your answer as a numerical value representing the number of years, with up to six decimal places.
Final Answer: """

answer_selection_after_verification_prompt = """You are a medical expert tasked with determining the duration of a Clinical Trial. Your task is to evaluate how many years the trial will last based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

**Trial Title:** 
{{Title}}

**Trial Summary:** 
{{Summary}}

**Trial Keywords:** {{Keywords}}

**Phase:** {{Phase}}

**Condition:** {{Condition}}

**Condition Keywords:** {{Condition_Keywords}}

**ICD Code:** {{icdcode}}

**Eligibility Criteria:** 
{{Eligibility_Criteria}}

**Interventions:** 
{{Interventions}}

**Study Design:** 
{{Study_Design}}

Reasoning Chain: {{Reasoning_Chain}}

Flaws in the Reasoning Chain: {{Verified_Reasoning_Chain}}

Given the trial information, the reasoning chain, and the identified flaws, provide your judgement on how many years the Clinical Trial will last. Please provide your answer as a numerical value representing the number of years, with up to six decimal places.
Final Answer: """

self_refine_feedback_prompt = """You are a medical expert reviewing a reasoning chain about the expected duration of a Clinical Trial.

**Trial Title:** 
{{Title}}

**Trial Summary:** 
{{Summary}}

**Trial Keywords:** {{Keywords}}

**Phase:** {{Phase}}

**Condition:** {{Condition}}

**Condition Keywords:** {{Condition_Keywords}}

**ICD Code:** {{icdcode}}

**Eligibility Criteria:** 
{{Eligibility_Criteria}}

**Interventions:** 
{{Interventions}}

**Study Design:** 
{{Study_Design}}

Reasoning Chain: {{Reasoning_Chain}}

Critique this reasoning chain. Identify logical gaps, unsupported conclusions, unclear steps, or factual inaccuracies.

Provide at least 2-3 specific points of improvement.
Feedback: """

self_refine_refine_prompt = """You are a medical expert tasked with improving a reasoning chain about the expected duration of a Clinical Trial.

**Trial Title:** 
{{Title}}

**Trial Summary:** 
{{Summary}}

**Trial Keywords:** {{Keywords}}

**Phase:** {{Phase}}

**Condition:** {{Condition}}

**Condition Keywords:** {{Condition_Keywords}}

**ICD Code:** {{icdcode}}

**Eligibility Criteria:** 
{{Eligibility_Criteria}}

**Interventions:** 
{{Interventions}}

**Study Design:** 
{{Study_Design}}

Reasoning Chain: {{Reasoning_Chain}}

Feedback: {{Feedback}}

Given the feedback above, revise the reasoning chain to:
1 - Fill logical gaps with evidence from the trial information.
2 - Clarify unclear reasoning steps.
3 - Ensure all conclusions are directly supported.
Let's think step by step, and at the very end write your answer in the form:
Answer: [numerical value] <END>"""

relevant_keys = ["Title", "Summary", "Keywords", "Phase", "Condition", "Condition_Keywords", 
                 "icdcode", "Eligibility_Criteria", "Interventions", "Study_Design", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

def fill_template(prompt, doc):
    """Fill template with document values"""
    res = prompt
    
    # Fill document values
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", str(doc[key]))
    
    return res

# Regression functions
def trial_duration_doc_to_text(doc):
    return fill_template(regression_baseline_prompt, doc)

def trial_duration_doc_to_text_reasoning(doc):
    return fill_template(regression_reasoning_prompt, doc)

def trial_duration_doc_to_text_answer_selection(doc):
    return fill_template(answer_selection_prompt, doc)

def trial_duration_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return fill_template(answer_selection_after_verification_prompt, doc)

def trial_duration_doc_to_text_self_refine_feedback(doc):
    return fill_template(self_refine_feedback_prompt, doc)

def trial_duration_doc_to_text_self_refine_refine(doc):
    return fill_template(self_refine_refine_prompt, doc)
