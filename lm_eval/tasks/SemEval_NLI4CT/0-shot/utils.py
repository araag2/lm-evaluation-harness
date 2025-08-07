from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, **kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, **kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

baseline_single_prompt = "You are a medical expert tasked with determining semantic entailment relationships between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate) and a section (the premise you will base your judgement in) from a Clinical Trial Report (CTR). The task is to classify the relationship as one of the following: Entailment (supported by the CTR) or Contradiction (in conflict with the CTR).\n\nSection Name:\n\n{{Section_id}}\n\nCTR Section: \n\n{{Primary_Section}}\n\nHypothesis: {{Statement}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the CTR Section and the Hypothesis.\nAnswer: "

baseline_comparison_prompt = "You are a medical expert tasked with determining semantic entailment relationships between a section of a Clinical Trial Report (CTR) and a clinical statement provided by a domain expert. \n\nEach input will consist of a statement (the hypothesis to evaluate) and two sections (the premises you will base your judgement in), one from the Primary Clinical Trial Report (CTR), and one from the Secondary CTR. The task is to classify the relationship as one of the following: Entailment (supported by both CTRs) or Contradiction (in conflict with atleast one of the CTRs). It is possible that a hypothesis compares the two CTRs, in which case the answer should be based on the comparison of the two sections.\n\nSection Name:\n\n{{Section_id}}\n\nPrimary CTR Section: \n\n{{Primary_Section}}\n\nSecondary CTR Section: \n\n{{Secondary_Section}}\n\nHypothesis: {{Statement}}\n\nPlease provide your judgement in a single word (Entailment or Contradiction), corresponding to the correct option that associates the CTR Sections and the Hypothesis.\nAnswer: "

type_to_prompt_and_opts = {
    "Single": (baseline_single_prompt, ["Section_id", "Primary_Section", "Statement"]),
    "Comparison": (baseline_comparison_prompt, ["Section_id", "Primary_Section", "Secondary_Section", "Statement"]),
}

def format_sections(sections):
   return '\n'.join(sections) if sections else ""

def doc_to_text(doc):
    res, opts = type_to_prompt_and_opts[doc["Type"]]
    for key in opts:
      res = res.replace(f"{{{{{key}}}}}", format_sections(doc[key]) if key in ["Primary_Section", "Secondary_Section"] else doc[key])
    return res