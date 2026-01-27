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

relevant_keys = ["Title", "Summary", "Keywords", "Phase", "Condition", "Condition_Keywords", "icdcode", "Eligibility_Criteria", "Interventions", "Study_Design"]

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

def mortality_rate_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "mortality_rate_prediction")

def mortality_rate_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "mortality_rate_prediction")

def patient_dropout_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "patient_dropout_prediction")

def patient_dropout_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "patient_dropout_prediction")

def trial_approval_doc_to_text(doc):
    return fill_template(binary_baseline_prompt, doc, "trial_approval_prediction")

def trial_approval_doc_to_text_reasoning(doc):
    return fill_template(binary_reasoning_prompt, doc, "trial_approval_prediction")
