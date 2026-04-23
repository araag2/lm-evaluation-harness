#baseline_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract of this medical article, generate the Conclusion Section. Only output the newly generated section.\nAnswer: "

baseline_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientific biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract, summarize the main finding about the intervention’s effect on the outcome.\n\nStrict requirements:\n- Output only a single, concise paragraph (1-3 sentences), similar in style and length to the examples below.\n- Focus only on the main result regarding the intervention’s effect.\n- Do NOT repeat information verbatim from the Abstract.\n- Do NOT include background, speculation, generic statements, markdown, or section headers.\n- Do NOT output anything except the new Conclusion Section.\n- Write in clear, complete sentences.\n\nAnswer: "

#reasoning_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract of this medical article, generate the Conclusion Section. Only output the newly generated section. Be as accurate as possible.\nAnswer: "

reasoning_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientific biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nBased on the Title and Abstract, summarize the main finding about the intervention’s effect on the outcome.\n\nStrict requirements:\n- Output only a single, concise paragraph (1-3 sentences), similar in style and length to the examples below.\n- Focus only on the main result regarding the intervention’s effect.\n- Do NOT repeat information verbatim from the Abstract.\n- Do NOT include background, speculation, generic statements, markdown, or section headers.\n- Do NOT output anything except the new Conclusion Section.\n- Write in clear, complete sentences.\n\nAnswer: "

#answer_selection_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientifical biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nDraft Conclusion: {{Reasoning_Chain}}\n\nGiven the Title, Abstract and the Draft Conclusion above, generate a refined and accurate Conclusion Section. Only output the newly generated section.\nAnswer: "
answer_selection_prompt="You are a medical reviewer tasked with generating conclusion sections based on scientific biomedical articles.\n\nTitle: {{Title}}\n\nAbstract: {{Abstract}}\n\nDraft Conclusion: {{Reasoning_Chain}}\n\nGiven the Title, Abstract, and the Draft Conclusion above, summarize the main finding about the intervention’s effect on the outcome.\n\nStrict requirements:\n- Output only a single, concise paragraph (1-3 sentences), similar in style and length to the examples below.\n- Focus only on the main result regarding the intervention’s effect.\n- Do NOT repeat information verbatim from the Abstract or Draft Conclusion.\n- Do NOT include background, speculation, generic statements, markdown, or section headers.\n- Do NOT output anything except the new Conclusion Section.\n- Write in clear, complete sentences.\n\nAnswer: "

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