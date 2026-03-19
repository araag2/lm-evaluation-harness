# Binary Classification Tasks (Yes/No)
binary_baseline_prompt = """You are a medical expert tasked with {{task_description}}. Your task is to evaluate {{evaluation_target}} based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Based on the above information, provide your judgement {{judgement_instruction}}. Please provide your judgement in short form, using "Yes" and "No".
Answer: """

binary_reasoning_prompt = """You are a medical expert tasked with {{task_description}}. Your task is to evaluate {{evaluation_target}} based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Based on the above information, provide your judgement {{judgement_instruction}}. Please provide your judgement in short form, using "Yes" and "No".
Let's think step by step, and at the very end write your answer in the form: 
Answer: [Yes / No] <END>"""

answer_selection_prompt = """You are a medical expert tasked with {{task_description}}. Your task is to evaluate {{evaluation_target}} based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Given the trial information and the reasoning chain, provide your judgement {{judgement_instruction}}. Please provide your judgement in short form, using "Yes" and "No".
Final Answer: """

answer_selection_after_verification_prompt = """You are a medical expert tasked with {{task_description}}. Your task is to evaluate {{evaluation_target}} based on the provided information, including the **Trial Title:**, **Trial Summary:**, **Trial Keywords:**, **Phase:**, **Condition:**, **Condition Keywords:**, **ICD Code:**, **Eligibility Criteria:**, **Interventions:**, and finally the **Study Design:**.

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

Given the trial information, the reasoning chain, and the identified flaws, provide your judgement {{judgement_instruction}}. Please provide your judgement in short form, using "Yes" and "No".
Final Answer: """

self_refine_feedback_prompt = """You are a medical expert reviewing a reasoning chain for a trial-assessment task.

Task: {{task_description}}
Evaluation target: {{evaluation_target}}

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

self_refine_refine_prompt = """You are a medical expert tasked with improving a reasoning chain for a trial-assessment task.

Task: {{task_description}}
Evaluation target: {{evaluation_target}}

Possible answers:
- Yes
- No

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
Answer: [Yes / No] <END>"""

# Task-specific descriptions
TASK_DESCRIPTIONS = {
    "adverse_event_prediction": {
        "task_description": "determining whether a Clinical Trial has severe adverse event risk",
        "evaluation_target": "the risk of severe adverse events",
        "judgement_instruction": "whether the Clinical Trial has severe adverse event risk"
    },
    "mortality_rate_prediction": {
        "task_description": "determining whether a Clinical Trial has mortality rate risk",
        "evaluation_target": "the risk of mortality rate",
        "judgement_instruction": "whether the Clinical Trial has mortality rate risk"
    },
    "patient_dropout_prediction": {
        "task_description": "determining whether a Clinical Trial has patient dropout risk",
        "evaluation_target": "the risk of patient dropout",
        "judgement_instruction": "whether the Clinical Trial has patient dropout risk"
    },
    "trial_approval_prediction": {
        "task_description": "determining whether a Clinical Trial will be approved",
        "evaluation_target": "the likelihood of trial approval",
        "judgement_instruction": "whether the Clinical Trial will be approved"
    }
}

relevant_keys = ["Title", "Summary", "Keywords", "Phase", "Condition", "Condition_Keywords", "icdcode", "Eligibility_Criteria", "Interventions", "Study_Design", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

def fill_template(prompt, doc, task_name=None):
    """Fill template with document values"""
    res = prompt
    
    # Fill task-specific descriptions for binary tasks
    if task_name and task_name in TASK_DESCRIPTIONS:
        for key, value in TASK_DESCRIPTIONS[task_name].items():
            res = res.replace(f"{{{{{key}}}}}", value)
    
    # Fill document values
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", str(doc[key]))
    
    return res

# Binary classification functions
def adverse_event_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "adverse_event_prediction")

def adverse_event_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "adverse_event_prediction")

def adverse_event_doc_to_text_answer_selection(doc):
    return fill_template(answer_selection_prompt, doc, "adverse_event_prediction")

def adverse_event_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return fill_template(answer_selection_after_verification_prompt, doc, "adverse_event_prediction")

def adverse_event_doc_to_text_self_refine_feedback(doc):
    return fill_template(self_refine_feedback_prompt, doc, "adverse_event_prediction")

def adverse_event_doc_to_text_self_refine_refine(doc):
    return fill_template(self_refine_refine_prompt, doc, "adverse_event_prediction")

# Mortality rate functions

def mortality_rate_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "mortality_rate_prediction")

def mortality_rate_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "mortality_rate_prediction")

def mortality_rate_doc_to_text_answer_selection(doc):
    return fill_template(answer_selection_prompt, doc, "mortality_rate_prediction")

def mortality_rate_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return fill_template(answer_selection_after_verification_prompt, doc, "mortality_rate_prediction")

def mortality_rate_doc_to_text_self_refine_feedback(doc):
    return fill_template(self_refine_feedback_prompt, doc, "mortality_rate_prediction")

def mortality_rate_doc_to_text_self_refine_refine(doc):
    return fill_template(self_refine_refine_prompt, doc, "mortality_rate_prediction")

# Patient dropout functions

def patient_dropout_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "patient_dropout_prediction")

def patient_dropout_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "patient_dropout_prediction")

def patient_dropout_doc_to_text_answer_selection(doc):
    return fill_template(answer_selection_prompt, doc, "patient_dropout_prediction")

def patient_dropout_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return fill_template(answer_selection_after_verification_prompt, doc, "patient_dropout_prediction")

def patient_dropout_doc_to_text_self_refine_feedback(doc):
    return fill_template(self_refine_feedback_prompt, doc, "patient_dropout_prediction")

def patient_dropout_doc_to_text_self_refine_refine(doc):
    return fill_template(self_refine_refine_prompt, doc, "patient_dropout_prediction")

# Trial approval functions

def trial_approval_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "trial_approval_prediction")

def trial_approval_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "trial_approval_prediction")

def trial_approval_doc_to_text_answer_selection(doc):
    return fill_template(answer_selection_prompt, doc, "trial_approval_prediction")

def trial_approval_doc_to_text_answer_selection_after_verify_reasoning(doc):
    return fill_template(answer_selection_after_verification_prompt, doc, "trial_approval_prediction")

def trial_approval_doc_to_text_self_refine_feedback(doc):
    return fill_template(self_refine_feedback_prompt, doc, "trial_approval_prediction")

def trial_approval_doc_to_text_self_refine_refine(doc):
    return fill_template(self_refine_refine_prompt, doc, "trial_approval_prediction")
