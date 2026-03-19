# Multi-class Classification Task
multiclass_baseline_prompt = """You are a medical expert tasked with determining why a Clinical Trial has failed. Your task is to evaluate the reason for the Clinical Trial failure based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Based on the above information, please identify the primary reason for the Clinical Trial failure. Choose from the following options: Safety, Efficacy, Poor Enrollment, or Other.
Answer: """

multiclass_reasoning_prompt = """You are a medical expert tasked with determining why a Clinical Trial has failed. Your task is to evaluate the reason for the Clinical Trial failure based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Based on the above information, please identify the primary reason for the Clinical Trial failure. Choose from the following options: Safety, Efficacy, Poor Enrollment, or Other.
Let's think step by step, and at the very end write your answer in the form: 
Answer: [Safety / Efficacy / Poor Enrollment / Other] <END>"""

answer_selection_prompt = """You are a medical expert tasked with determining why a Clinical Trial has failed. Your task is to evaluate the reason for the Clinical Trial failure based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Given the trial information and the reasoning chain, identify the primary reason for the Clinical Trial failure. Choose from: Safety, Efficacy, Poor Enrollment, or Other.
Final Answer: """

answer_selection_after_verification_prompt = """You are a medical expert tasked with determining why a Clinical Trial has failed. Your task is to evaluate the reason for the Clinical Trial failure based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Given the trial information, the reasoning chain, and the identified flaws, identify the primary reason for the Clinical Trial failure. Choose from: Safety, Efficacy, Poor Enrollment, or Other.
Final Answer: """

self_refine_feedback_prompt = """You are a medical expert reviewing a reasoning chain about why a Clinical Trial has failed.

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

self_refine_refine_prompt = """You are a medical expert tasked with improving a reasoning chain about why a Clinical Trial has failed.

Possible answers:
- Safety
- Efficacy
- Poor Enrollment
- Other

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
Answer: [Safety / Efficacy / Poor Enrollment / Other] <END>"""

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

# Multi-class classification functions
def trial_failure_doc_to_text(doc):
    return fill_template(multiclass_baseline_prompt, doc)

def trial_failure_doc_to_text_reasoning(doc):
    return fill_template(multiclass_reasoning_prompt, doc)

def trial_failure_doc_to_text_answer_selection(doc):
    return fill_template(answer_selection_prompt, doc)

def trial_failure_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return fill_template(answer_selection_after_verification_prompt, doc)

def trial_failure_doc_to_text_self_refine_feedback(doc):
    return fill_template(self_refine_feedback_prompt, doc)

def trial_failure_doc_to_text_self_refine_refine(doc):
    return fill_template(self_refine_refine_prompt, doc)
