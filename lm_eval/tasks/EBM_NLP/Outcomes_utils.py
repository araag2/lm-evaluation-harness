Outcomes_baseline_prompt="""You are an expert biomedical annotator for PICO span extraction.

Task: extract outcome-related spans from the abstract.

Include spans that describe measured endpoints/results concepts, such as:
    - named outcomes/endpoints
    - symptom scores or clinical measures
    - adverse events and efficacy/tolerability outcome phrases

Output rules (strict):
    1. Output only the extracted spans, one span per line.
    2. Each span must be copied as an exact contiguous text span from the context.
    3. Preserve original case and punctuation exactly as written.
    4. Keep the order of appearance in the context.
    5. Keep repeated mentions if they occur multiple times.
    6. Do not add bullets, numbering, quotes, or explanations.
    7. If no valid span exists, output exactly: None

**Context:** {{Context}}

Return only the extracted spans as stated in the Output Rules.

Answer: """

Outcomes_reasoning_prompt="""You are an expert biomedical annotator for PICO span extraction.

Task: extract outcome-related spans from the abstract.

Include spans that describe measured endpoints/results concepts, such as:
    - named outcomes/endpoints
    - symptom scores or clinical measures
    - adverse events and efficacy/tolerability outcome phrases

Output rules (strict):
    1. Output only the extracted spans, one span per line.
    2. Each span must be copied as an exact contiguous text span from the context.
    3. Preserve original case and punctuation exactly as written.
    4. Keep the order of appearance in the context.
    5. Keep repeated mentions if they occur multiple times.
    6. Do not add bullets, numbering, quotes, or explanations.
    7. If no valid span exists, output exactly: None

**Context:** {{Context}}

Explain your resoning for extracting the spans, and then return all the extracted spans as stated in the Output Rules. 

Answer: """

Outcomes_answer_selection_prompt="""You are an expert biomedical annotator for PICO span extraction.

Task: extract outcome-related spans from the abstract.

Include spans that describe measured endpoints/results concepts, such as:
    - named outcomes/endpoints
    - symptom scores or clinical measures
    - adverse events and efficacy/tolerability outcome phrases

Output rules (strict):
    1. Output only the extracted spans, one span per line.
    2. Each span must be copied as an exact contiguous text span from the context.
    3. Preserve original case and punctuation exactly as written.
    4. Keep the order of appearance in the context.
    5. Keep repeated mentions if they occur multiple times.
    6. Do not add bullets, numbering, quotes, or explanations.
    7. If no valid span exists, output exactly: None

**Context:** {{Context}}

**Draft Extracted Spans:** {{Reasoning_Chain}}

Return only the extracted spans as stated in the Output Rules.

Answer: """

relevant_keys = ["Context", "Reasoning_Chain"]

def Outcomes_doc_to_text(doc, prompt=Outcomes_baseline_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc and doc[key]:
            value = doc[key]
            res = res.replace(f"{{{{{key}}}}}", value)
    return res

def Outcomes_doc_to_text_reasoning(doc):
    return Outcomes_doc_to_text(doc, Outcomes_reasoning_prompt)

def Outcomes_doc_to_text_answer_selection(doc):
    return Outcomes_doc_to_text(doc, Outcomes_answer_selection_prompt)

def Outcomes_process_docs(dataset):
    return dataset.filter(lambda doc: doc["Category"] == "Outcomes")