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

relevant_keys = ["Title", "Summary", "Keywords", "Phase", "Condition", "Condition_Keywords", 
                 "icdcode", "Eligibility_Criteria", "Interventions", "Study_Design"]

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
