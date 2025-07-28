from sklearn.metrics import precision_score, recall_score

def precision_fn(refs, preds, **kwargs):
    return {"precision": precision_score(refs, preds, average="weighted", zero_division=0)}
    
def recall_fn(refs, preds, **kwargs):
    return {"recall": recall_score(refs, preds, average="weighted", zero_division=0)}

