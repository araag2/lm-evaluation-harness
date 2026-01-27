from functools import partial

# Trial Phases for filtering
PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"]

def process_docs(dataset, phase):
    """Filter dataset by trial phase"""
    return dataset.filter(lambda x: x["Phase"] == phase)

def process_docs_exclude_phase4(dataset):
    """Filter dataset to exclude Phase 4 trials"""
    return dataset.filter(lambda x: x["Phase"] != "Phase 4")

# Create partial functions for each phase
process_phase_1 = partial(process_docs, phase="Phase 1")
process_phase_2 = partial(process_docs, phase="Phase 2")
process_phase_3 = partial(process_docs, phase="Phase 3")
process_phase_4 = partial(process_docs, phase="Phase 4")
process_not_applicable = partial(process_docs, phase="Not Applicable")
