import re

baseline_prompt =  "Your task is to predict the outcome in binary format (Success or Failure), based on the provided information on the clinical trial.\n\nYou will be provided with the following:\n- Diseases: A list of diseases targeted by the clinical trial.\n- ICD Codes: A list of ICD codes associated with the diseases.\n- Drugs: A list of drugs used in the clinical trial.\n- SMILES: A list of SMILES strings representing the chemical structures of the drugs\n- CT Criteria: The inclusion and exclusion criteria for the clinical trial.\nDiseases: {{Diseases}}\n\nICD Codes: {{ICD_Codes}}\n\nDrugs: {{Drugs}}\n\nSMILES: {{SMILES}}\n\nCT Criteria: {{CT_Criteria}}\n\nPlease answer with a single outcome prediction, either 'Success' or 'Failure'.\nAnswer: "

reasoning_prompt = "Your task is to predict the outcome in binary format (Success or Failure), based on the provided information on the clinical trial.\n\nYou will be provided with the following:\n- Diseases: A list of diseases targeted by the clinical trial.\n- ICD Codes: A list of ICD codes associated with the diseases.\n- Drugs: A list of drugs used in the clinical trial.\n- SMILES: A list of SMILES strings representing the chemical structures of the drugs\n- CT Criteria: The inclusion and exclusion criteria for the clinical trial.\nDiseases: {{Diseases}}\n\nICD Codes: {{ICD_Codes}}\n\nDrugs: {{Drugs}}\n\nSMILES: {{SMILES}}\n\nCT Criteria: {{CT_Criteria}}\nPlease provide your judgement (Success or Failure), corresponding to the outcome. Be as accurate as possible.\nLet's think step by step, and at the very end write your answer in the form:\nAnswer: [Success / Failure] <END>"

answer_selection_prompt = "Your task is to predict the outcome in binary format (Success or Failure), based on the provided information on the clinical trial.\n\nYou will be provided with the following:\n- Diseases: A list of diseases targeted by the clinical trial.\n- ICD Codes: A list of ICD codes associated with the diseases.\n- Drugs: A list of drugs used in the clinical trial.\n- SMILES: A list of SMILES strings representing the chemical structures of the drugs\n- CT Criteria: The inclusion and exclusion criteria for the clinical trial.\nDiseases: {{Diseases}}\n\nICD Codes: {{ICD_Codes}}\n\nDrugs: {{Drugs}}\n\nSMILES: {{SMILES}}\n\nCT Criteria: {{CT_Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nGiven the Diseases, ICD Codes, Drugs, SMILES, CT Criteria and with special atention to the presented Reasoning Chain, provide your judgement (Success or Failure), corresponding to the outcome. Be as accurate as possible.\nFinal Answer: "

verify_reasoning_prompt = "Your task is to verify a reasoning chain that determines the outcome in binary format (Success or Failure), based on the provided information on the clinical trial.\n\nYou will be provided with the following:\n- Diseases: A list of diseases targeted by the clinical trial.\n- ICD Codes: A list of ICD codes associated with the diseases.\n- Drugs: A list of drugs used in the clinical trial.\n- SMILES: A list of SMILES strings representing the chemical structures of the drugs\n- CT Criteria: The inclusion and exclusion criteria for the clinical trial.\nDiseases: {{Diseases}}\n\nICD Codes: {{ICD_Codes}}\n\nDrugs: {{Drugs}}\n\nSMILES: {{SMILES}}\n\nCT Criteria: {{CT_Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nPlease verify if the Reasoning Chain makes logical sense, and supports the correct conclusion. Let's think step by step, and after explaining your verification reasoning, provide your answer in the form:\nVerified Answer: [Success / Failure] <END>"

answer_selection_after_verification_prompt = "Your task is to predict the outcome in binary format (Success or Failure), based on the provided information on the clinical trial.\n\nYou will be provided with the following:\n- Diseases: A list of diseases targeted by the clinical trial.\n- ICD Codes: A list of ICD codes associated with the diseases.\n- Drugs: A list of drugs used in the clinical trial.\n- SMILES: A list of SMILES strings representing the chemical structures of the drugs\n- CT Criteria: The inclusion and exclusion criteria for the clinical trial.\nDiseases: {{Diseases}}\n\nICD Codes: {{ICD_Codes}}\n\nDrugs: {{Drugs}}\n\nSMILES: {{SMILES}}\n\nCT Criteria: {{CT_Criteria}}\n\nReasoning Chain: {{Reasoning_Chain}}\n\nVerification of the Reasoning Chain: {{Verified_Reasoning_Chain}}\n\nGiven the Diseases, ICD Codes, Drugs, SMILES, CT Criteria, the initial Reasoning Chain and the Verification of the Reasoning Chain, provide your judgement (Success or Failure), corresponding to the correct outcome. Be as accurate as possible.\nFinal Answer: "

pos_answers = ['Success', 'Failure']

def label_to_index(doc) -> int:
    return pos_answers.index(doc["Label"])

def fix_formatting(text):
    """
    Fix excessive whitespace in the text.
    """
    text = re.sub(r'\n\n {3,}', '\n  ', text)
    text = re.sub(r'\n {3,}', ' ', text)
    return re.sub(r'[\[\]\']', '', text).strip()

relevant_keys = ["Diseases", "ICD_Codes", "Drugs", "SMILES", "CT_Criteria", "Reasoning_Chain", "Verified_Reasoning_Chain"]

def doc_to_text(doc, prompt = baseline_prompt):
    res = prompt
    for key in relevant_keys:
        if key in doc:
            res = res.replace(f"{{{{{key}}}}}", fix_formatting(doc[key]))
    return res

def doc_to_text_reasoning(doc):
    return doc_to_text(doc, reasoning_prompt)

def doc_to_text_answer_selection(doc):
    return doc_to_text(doc, answer_selection_prompt)

def doc_to_text_verify_reasoning(doc):
    return doc_to_text(doc, verify_reasoning_prompt)

def doc_to_text_answer_selection_after_verify_reasoning(doc):
    return doc_to_text(doc, answer_selection_after_verification_prompt)