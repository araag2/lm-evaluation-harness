def label_to_index(doc) -> int:
    assert isinstance(pos_answers.index(doc["Label"]), int)
    return pos_answers.index(doc["Label"])