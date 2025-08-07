import re

baseline_prompt =  "Your task is to predict the outcome in binary format (Success or Failure), based on the provided information on the clinical trial.\n\nYou will be provided with the following:\n- Diseases: A list of diseases targeted by the clinical trial.\n- ICD Codes: A list of ICD codes associated with the diseases.\n- Drugs: A list of drugs used in the clinical trial.\n- SMILES: A list of SMILES strings representing the chemical structures of the drugs\n- CT Criteria: The inclusion and exclusion criteria for the clinical trial.\nDiseases: {{Diseases}}\n\nICD Codes: {{ICD_Codes}}\n\nDrugs: {{Drugs}}\n\nSMILES: {{SMILES}}\n\nCT Criteria: {{CT_Criteria}}\n\nPlease answer with a single outcome prediction, either 'Success' or 'Failure'.\nAnswer: "

def fix_formatting(text):
    """
    Fix excessive whitespace in the text.
    """
    text = re.sub(r'\n\n {3,}', '\n  ', text)
    text = re.sub(r'\n {3,}', ' ', text)
    return re.sub(r'[\[\]\']', '', text).strip()

def doc_to_text(doc):
    res = baseline_prompt
    for key in ['Diseases', 'ICD_Codes', 'Drugs', 'SMILES', 'CT_Criteria']:
      res = res.replace(f"{{{{{key}}}}}", fix_formatting(doc[key]))
    return res