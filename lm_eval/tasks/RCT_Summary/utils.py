#baseline_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract of this medical article, generate the Conclusion Section. Only output the newly generated section.\nAnswer: "

baseline_prompt="""
You are a medical reviewer tasked with writing a conclusion based on a scientific biomedical article.

Title: {{Title}}

Abstract: {{Abstract}}

Write a single concise conclusion summarizing the main finding about the intervention’s effect on the outcome.

STRICT OUTPUT RULES:
- Output ONLY one paragraph (1–3 sentences).
- Focus ONLY on the intervention’s effect on the outcome.
- Do NOT include background, reasoning, speculation, or general statements.
- Do NOT repeat phrases from the abstract or draft conclusion.
- Do NOT include headings, labels, or extra text.

OUTPUT FORMAT (must follow exactly):
Answer: <conclusion> <END>

Return only the line starting with "Answer:" and ending with "<END>".

Answer: """

#reasoning_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract of this medical article, generate the Conclusion Section. Only output the newly generated section. Be as accurate as possible.\nAnswer: "

reasoning_prompt="""
You are a medical reviewer tasked with writing a conclusion based on a scientific biomedical article.

Title: {{Title}}

Abstract: {{Abstract}}

Draft Conclusion: {{Reasoning_Chain}}

Write a single concise conclusion summarizing the main finding about the intervention’s effect on the outcome.

STRICT OUTPUT RULES:
- Focus ONLY on the intervention’s effect on the outcome.
- Do NOT include background, reasoning, speculation, or general statements.
- Do NOT repeat phrases from the abstract or draft conclusion.
- Do NOT include headings, labels, or extra text.

OUTPUT FORMAT (must follow exactly):
Answer: <conclusion> <END>

Explain your resoning for generating a given conclusion, and then return the line starting with "Answer:" and ending with "<END>". """

#answer_selection_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nDraft Conclusion: {{Reasoning_Chain}}\n\nGiven the Title, Abstract and the Draft Conclusion above, generate a refined and accurate Conclusion Section. Only output the newly generated section.\nAnswer: "
answer_selection_prompt="""
You are a medical reviewer tasked with writing a conclusion based on a scientific biomedical article.

Title: {{Title}}

Abstract: {{Abstract}}

Draft Conclusion: {{Reasoning_Chain}}

Write a single concise conclusion summarizing the main finding about the intervention’s effect on the outcome.

STRICT OUTPUT RULES:
- Output ONLY one paragraph (1–3 sentences).
- Focus ONLY on the intervention’s effect on the outcome.
- Do NOT include background, reasoning, speculation, or general statements.
- Do NOT repeat phrases from the abstract or draft conclusion.
- Do NOT include headings, labels, or extra text.

OUTPUT FORMAT (must follow exactly):
Answer: <conclusion> <END>

Return only the line starting with "Answer:" and ending with "<END>".

Answer: """

relevant_keys = ["Title", "Abstract", "Reasoning_Chain"]


def _clean_reasoning_chain(text: str) -> str:
    if not text:
        return text

    # Keep only the first answer block when models start looping "Answer:".
    head = text.split("\nAnswer:")[0].strip()
    if head:
        text = head

    # Remove common markdown wrappers that often leak into generations.
    text = text.replace("**Conclusion**", "").strip()

    # Keep prompts compact for second-stage answer selection.
    return text[:1200].strip()

def doc_to_text(doc, prompt=baseline_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc and doc[key]:
            value = doc[key]
            if key == "Reasoning_Chain":
                value = _clean_reasoning_chain(value)
            res = res.replace(f"{{{{{key}}}}}", value)
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text(doc, answer_selection_prompt)