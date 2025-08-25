def doc_to_text(doc, res, opts):
    for key in opts:
        res = res.replace(f"{{{{{key}}}}}", doc[key])
    return res