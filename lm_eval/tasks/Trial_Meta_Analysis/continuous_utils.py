# utils_continuous_extraction.py

continuous_reasoning_prompt = "You are an expert medical assistant specialized in extracting data from randomized controlled trials. Your task is to produce a 2x2 contingency table in YAML format based on the provided article context.\n\nYou will be given an article context preceded by **Article:**, which contains the Abstract and Results sections of a randomized controlled trial. Pay close attention to the details in the article to accurately classify the outcome type.\n\nYou will also be provided with an ICO (Intervention, Comparator, and Outcome) triplet, which includes the intervention, comparator, and outcome of interest. Your task is to extract the mean, standard deviation, and group sizes for both the intervention and comparator groups.\n\n**Article:** \n\n{{Context}}\n\nBased on the given trial article, produce a 2x2 contingency table in YAML format for the following ICO (Intervention, Comparator, and Outcome) triplet: \n\n**Intervention:** {{intervention}}\n**Comparator:** {{comparator}}\n**Outcome:** {{outcome}}\n\nThe YAML format should include the fields \"mean\", \"standard_deviation\", and \"group_size\" for \"intervention\" and \"comparator\", but not for \"outcome\".\n\n**Example:**\nintervention:\n    mean: NUMBER (Float) or x\n    standard_deviation: NUMBER (Float) or x\n    group_size: NUMBER (Integer) or x\ncomparator:\n    mean: NUMBER (Float) or x\n    standard_deviation: NUMBER (Float) or x\n    group_size: NUMBER (Integer) or x\n\nDon't add percentages (%), don't add decimals to group size. However, mean and standard_deviation always have decimals.\n\nOnly produce the YAML response. Do not provide any additional explanation. If any of the numerical information is unavailable, not extractable or not possible to calculate, replace it with \"x\".\n\nIf there is numerical data for pre and post-intervention, choose the post-intervention data. If there are multiple timeframes for the outcome, choose the one closest to the outcome timepoint of interest, or alternatively the last one. Be as accurate as possible, and only ouptut the YAML.\n\nYAML:\n"

continuous_answer_selection_prompt = "You are an expert medical assistant specialized in extracting data from randomized controlled trials. Your task is to produce a refined 2x2 contingency table in YAML format, given a draft extraction and the article context.\n\n**Article:** \n\n{{Context}}\n\nBased on the given trial article, produce a refined 2x2 contingency table in YAML format for the following ICO (Intervention, Comparator, and Outcome) triplet: \n\n**Intervention:** {{intervention}}\n**Comparator:** {{comparator}}\n**Outcome:** {{outcome}}\n\nDraft Extraction: {{Reasoning_Chain}}\n\nThe YAML format should include the fields \"mean\", \"standard_deviation\", and \"group_size\" for \"intervention\" and \"comparator\", but not for \"outcome\".\n\n**Example:**\nintervention:\n    mean: NUMBER (Float) or x\n    standard_deviation: NUMBER (Float) or x\n    group_size: NUMBER (Integer) or x\ncomparator:\n    mean: NUMBER (Float) or x\n    standard_deviation: NUMBER (Float) or x\n    group_size: NUMBER (Integer) or x\n\nDon't add percentages (%), don't add decimals to group size. However, mean and standard_deviation always have decimals.\n\nOnly produce the YAML response. Do not provide any additional explanation. If any of the numerical information is unavailable, not extractable or not possible to calculate, replace it with \"x\".\n\nIf there is numerical data for pre and post-intervention, choose the post-intervention data. If there are multiple timeframes for the outcome, choose the one closest to the outcome timepoint of interest, or alternatively the last one. Be as accurate as possible. Only output the YAML answer followed by <END>.\n\nYAML:\n"


pos_answers = ["binary", "continuous"]

def continuous_label_to_index(doc) -> int:
    return pos_answers.index(doc["outcome_type"])

relevant_keys = ["Context", "intervention", "comparator", "outcome", "Reasoning_Chain", "Verified_Reasoning_Chain", "Feedback"]

def continuous_doc_to_text(doc, prompt = continuous_reasoning_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            value = doc[key]
            value = value[:2000] + value[-2000:] if len(value) > 4000 else value
            res = res.replace(f"{{{{{key}}}}}", value)
    return res

def continuous_doc_to_text_reasoning(doc):
    return continuous_doc_to_text(doc, continuous_reasoning_prompt)

def continuous_doc_to_text_answer_selection(doc):
    return continuous_doc_to_text(doc, continuous_answer_selection_prompt)
