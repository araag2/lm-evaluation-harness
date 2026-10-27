"""Evidence views of an NLI4CT statement: its CTR section(s) with only, or without, the gold evidence.

NLI4CT annotates which section lines are the evidence for each statement (`Primary_evidence_index`,
`Secondary_evidence_index`). Its labels are binary: Contradiction covers both a contradicted and an
unsupported statement. So, by the task's own definition:
- `evidence_only` (the evidence lines and the section headers) keeps the statement's label;
- `evidence_removed` (every line but the evidence, headers kept) leaves it unsupported: Contradiction.

Only statements whose evidence leaves out some content line get these views; when the evidence is the
whole section, removing it would leave an empty premise.
"""

from collections.abc import Sequence


FULL, EVIDENCE_ONLY, EVIDENCE_REMOVED = "full", "evidence_only", "evidence_removed"
UNSUPPORTED_LABEL = "Contradiction"


def is_header(line: str) -> bool:
    """A section header ("Inclusion Criteria:", "INTERVENTION 1: "): unindented and ending in a colon."""
    return not line.startswith(" ") and line.rstrip().endswith(":")


def _keep(lines: Sequence[str], indices: Sequence[int], evidence: bool) -> list[str]:
    chosen = set(indices)
    return [
        line
        for i, line in enumerate(lines)
        if is_header(line) or ((i in chosen) == evidence)
    ]


def _content(lines: Sequence[str]) -> list[str]:
    return [line for line in lines if not is_header(line)]


def evidence_views(
    sections: Sequence[Sequence[str]], indices: Sequence[Sequence[int]]
) -> dict[str, list[list[str]]] | None:
    """The `evidence_only` and `evidence_removed` versions of one statement's section(s), or None.

    `sections` and `indices` list the primary (and secondary) trial's section lines and evidence
    indices. None when the evidence has no content line or leaves none out in every section.
    """
    only = [
        _keep(lines, idx, evidence=True)
        for lines, idx in zip(sections, indices, strict=True)
    ]
    removed = [
        _keep(lines, idx, evidence=False)
        for lines, idx in zip(sections, indices, strict=True)
    ]
    if not any(_content(lines) for lines in only):
        return None
    if not any(_content(lines) for lines in removed):
        return None
    return {EVIDENCE_ONLY: only, EVIDENCE_REMOVED: removed}


def view_label(view: str, label: str) -> str:
    return UNSUPPORTED_LABEL if view == EVIDENCE_REMOVED else label
