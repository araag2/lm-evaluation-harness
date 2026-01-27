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

# Multi-class classification functions
def trial_failure_doc_to_text(doc):
    return fill_template(multiclass_baseline_prompt, doc)

def trial_failure_doc_to_text_reasoning(doc):
    return fill_template(multiclass_reasoning_prompt, doc)
