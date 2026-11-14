"""Family-level audit of NLI4CT training arms (OpenCTEval `_Reports/NLI4CT_Training/HYPOTHESES.md`, Check A1).

Asks of each training arm (a `save_to_disk` set from `build_nli4ct_pilot.py` / `build_nli4ct_f6_test.py`) whether
a recipe's label can be read from its surface form, and whether its label rule fits NLI4CT's (Jullien et al. 2023:
Contradiction statements are "objectively wrong" rewrites of true ones; the 2024 Text_appended gold keeps the label).
Per mode: rows, Contradiction share, the implied rule and its fit (RULES), and how often the untrained model agrees
with the gold label (8 samples per row, where the ALL arm was rolled out). Per arm: probes that predict the label from
partial input with 5-fold cross-validation grouped by statement (recipe id only, edit only, statement only, and a
shuffled-label control) and the Gardner et al. (2021) z-test on statement tokens. With `--outcomes`, the arm scores
are set against the held-out family scores of the runs trained on each arm (seed means, `family_scores.py`).

    python scripts/family_audit.py --report R.md [--outcomes]
"""

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import norm, spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict


sys.path.insert(0, str(Path(__file__).resolve().parent))
import family_scores as fs


PIPELINE = Path(__file__).resolve().parents[2] / "OpenCTEval-Data-Pipeline/outputs/training/data"
PILOT = PIPELINE / "nli4ct_pilot"
ROLLOUTS = PIPELINE / "nli4ct_pilot_rollouts/untrained_ALL.jsonl"
FAMILY_OF = {m: f for source in fs.FAMILY_OF.values() for m, f in source.items()}
# The label each recipe implies, and how it fits NLI4CT's definitions ("judge": set per row by the generator's model).
RULES = {
    "original": ("gold", "fits"),
    "recast": ("keep (rewording)", "fits: Paraphrase keeps the label"),
    "substitute-syntactic": ("keep (synonyms)", "fits: Paraphrase keeps the label"),
    "append_definition": ("keep (unrelated definition appended)", "fits: Text_appended keeps the label"),
    "numeric": ("judge (number changed)", "judge"),
    "biomedical-recast": ("judge (term replaced)", "judge"),
    "negate_false-recast": ("C → E (false statement negated; support-checked)", "fits: negation of a wrong claim"),
    "contrary": ("E → C (opposite claim; judge-checked)", "fits: Contradictions are wrong rewrites"),
    "evidence_only": ("keep (irrelevant lines removed)", "fits: irrelevant content does not change the label"),
    "negate_true-recast": ("E → C (true statement negated)", "fits: Contradictions are wrong rewrites"),
    "evidence_control": ("keep (random non-evidence lines removed)", "fits: irrelevant content does not change the label"),
    "evidence_removed": ("E/C → C (supporting lines removed: 'not supported')",
                         "CONFLICTS: NLI4CT has no 'unsupported' class; Text_appended keeps E for unsupported text"),
}
ARMS = {  # arm → training set; the runs trained on it (seed means); the families it holds out
    "BASE": ("BASE", ["fam-ce-base", "fam-ce-base-s7", "fam-ce-base-s13"], []),
    "A": ("A", ["fam-ce-a", "fam-ce-a-s7"], ["F2", "F5"]),
    "B": ("B", ["fam-ce-b", "fam-ce-b-s7"], ["F1", "F3"]),
    "C": ("C", ["fam-ce-c", "fam-ce-c-s7"], ["F4", "F6"]),
    "ALL": ("ALL", ["fam-ce-all", "fam-ce-all-s7"], []),
    "k1": ("diversity/k1", ["fam-div-k1", "fam-div-k1-s7"], ["F2", "F5"]),
    "k2": ("diversity/k2", ["fam-div-k2", "fam-div-k2-s7"], ["F2", "F5"]),
    "k4": ("diversity/k4", ["fam-div-k4", "fam-div-k4-s7"], ["F2", "F5"]),
    "f6-bal": ("f6/bal", ["fam-f6-bal"], ["F2", "F5"]),
    "f6-noER": ("f6/noER", ["fam-f6-noer"], ["F2", "F5"]),
    "f6-noF6": ("f6/noF6", ["fam-f6-nof6"], ["F2", "F5"]),
    "f6-noER-s7": ("f6/noER", ["fam-f6-noer-s7"], ["F2", "F5"]),  # seed 7 as its own point (Check A1: n = 10)
    "f6-noF6-s7": ("f6/noF6", ["fam-f6-nof6-s7"], ["F2", "F5"]),
    "A-noER": ("f6/A-noER", ["fam-ce-a-noer"], ["F2", "F5"]),
    "A-fixed": ("fixed/A", [], ["F2", "F5"]),  # corrected convention (build_nli4ct_fixed.py); Check A2
    "A-noER-matched": ("fixed/A-noER-matched", [], ["F2", "F5"]),  # A2 control: A-noER cut to A-fixed's size
}
WORD = re.compile(r"[a-z0-9]+")
MIN_TOKEN_ROWS = 20
ALPHA = 0.01
FIXED = 0.95  # a mode whose rows carry one label this often has a fixed label


def mode(r: dict) -> str:
    return (r["augmentation"] or {}).get("mode", "original")


def lines(r: dict) -> list[str]:
    return list(r.get("Primary_Section") or []) + list(r.get("Secondary_Section") or [])


def edit_features(r: dict, original: dict) -> dict[str, float]:
    """What the edit did, without the statement's content: words added and removed, appended or not, section cut."""
    a, b = set(WORD.findall(original["Statement"].lower())), set(WORD.findall(r["Statement"].lower()))
    feats = {f"+{w}": 1.0 for w in b - a} | {f"-{w}": 1.0 for w in a - b}
    if r["Statement"].startswith(original["Statement"].rstrip(". ")) and b - a:
        feats["APPENDED"] = 1.0
    if r["Statement"] == original["Statement"]:
        feats["STATEMENT_SAME"] = 1.0
    n0, n1 = len(lines(original)), len(lines(r))
    feats["SECTION_" + ("same" if n1 == n0 else "cut_half+" if n1 <= n0 / 2 else "cut")] = 1.0
    return feats


def cv_accuracy(features, labels, groups, vectorizer) -> list[str]:
    x = vectorizer.fit_transform(features)
    folds = GroupKFold(n_splits=5)
    return list(cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), x, labels, groups=groups, cv=folds))


def ztest(rows: list[dict]) -> tuple[int, list[str]]:
    """Gardner et al. (2021): tokens whose Contradiction share departs from the rows' own beyond a Bonferroni bound.

    The expected share is the base rate of the rows, not 1/2, so a skewed label mix does not flag every common word.
    """
    seen, contra = Counter(), Counter()
    for r in rows:
        for w in set(WORD.findall(r["Statement"].lower())):
            seen[w] += 1
            contra[w] += r["Label"] == "Contradiction"
    p0 = statistics.fmean(r["Label"] == "Contradiction" for r in rows)
    tested = [w for w, n in seen.items() if n >= MIN_TOKEN_ROWS]
    bound = norm.isf(ALPHA / (2 * max(len(tested), 1)))
    z = {w: (contra[w] / seen[w] - p0) / math.sqrt(p0 * (1 - p0) / seen[w]) for w in tested}
    hits = sorted((w for w in tested if abs(z[w]) > bound), key=lambda w: -abs(z[w]))
    return len(hits), [f"{w} ({'C' if z[w] > 0 else 'E'} {z[w]:+.1f})" for w in hits[:8]]


def untrained_agreement() -> dict[str, float]:
    if not ROLLOUTS.exists():
        return {}
    return {r["id"]: statistics.fmean(s["answer"] == r["label"] for s in r["samples"]) for r in map(json.loads, ROLLOUTS.open())}


def audit(rows: list[dict], agree: dict[str, float]) -> tuple[list[str], dict[str, float]]:
    original = {r["source_id"]: r for r in rows if mode(r) == "original"}
    variants = [r for r in rows if mode(r) != "original" and r["source_id"] in original]
    labels = [r["Label"] for r in variants]
    groups = [r["source_id"] for r in variants]
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    recipe = cv_accuracy([{mode(r): 1.0} for r in variants], labels, groups, DictVectorizer())
    edits = [edit_features(r, original[r["source_id"]]) for r in variants]
    edit = cv_accuracy(edits, labels, groups, DictVectorizer())
    shuffled_labels = random.Random(0).sample(labels, len(labels))
    shuffled = cv_accuracy(edits, shuffled_labels, groups, DictVectorizer())
    every = [r["Label"] for r in rows]
    statement = cv_accuracy([r["Statement"] for r in rows], every, [r["source_id"] for r in rows],
                            TfidfVectorizer(ngram_range=(1, 2), min_df=2))
    acc = lambda pred, gold, idx: statistics.fmean(pred[i] == gold[i] for i in idx)
    by_mode = defaultdict(list)
    for i, r in enumerate(variants):
        by_mode[mode(r)].append(i)
    out = ["| mode | family | rows | C share | rule | fits NLI4CT | untrained agrees (rows) | edit-only probe |",
           "|-|-|-|-|-|-|-|-|"]
    orig = [r for r in rows if mode(r) == "original"]
    for m, idx in [("original", None)] + sorted(by_mode.items()):
        group = orig if idx is None else [variants[i] for i in idx]
        c = statistics.fmean(r["Label"] == "Contradiction" for r in group)
        seen = [agree[r["id"]] for r in group if r["id"] in agree]
        ua = f"{statistics.fmean(seen):.2f} ({len(seen)})" if seen else "–"
        probe = "–" if idx is None else f"{acc(edit, labels, idx):.2f}"
        rule, fit = RULES.get(m, ("?", "?"))
        out.append(f"| {m} | {FAMILY_OF.get(m, '–')} | {len(group)} | {c:.2f} | {rule} | {fit} | {ua} | {probe} |")
    n_hits, top = ztest(rows)
    n_hits_orig, _ = ztest(orig)
    fixed = {m for m, idx in by_mode.items() if max(Counter(labels[i] for i in idx).values()) >= FIXED * len(idx)}
    conflict = statistics.fmean(RULES.get(mode(r), ("", ""))[1].startswith("CONFLICTS") for r in variants)
    seen = [agree[r["id"]] for r in variants if r["id"] in agree]
    scores = {
        "variant rows": len(variants), "majority": majority,
        "recipe-id probe": acc(recipe, labels, range(len(labels))),
        "edit-only probe": acc(edit, labels, range(len(labels))),
        "edit-only, shuffled labels": acc(shuffled, shuffled_labels, range(len(labels))),
        "statement-only probe, originals": acc(statement, every, [i for i, r in enumerate(rows) if mode(r) == "original"]),
        "statement-only probe, variants": acc(statement, every, [i for i, r in enumerate(rows) if mode(r) != "original"]),
        "conflicting-rule share": conflict,
        "fixed-label share": statistics.fmean(mode(r) in fixed for r in variants), "variant C share": statistics.fmean(x == "Contradiction" for x in labels),
        "untrained disagreement (variants)": 1 - statistics.fmean(seen) if seen else float("nan"),
        "z-test tokens (arm / originals only)": f"{n_hits} / {n_hits_orig}",
    }
    out += ["", f"Significant z-test tokens: {n_hits} in the arm, {n_hits_orig} in its originals alone. Top: "
            + ", ".join(top)]
    return out, scores


def outcomes(arms: dict) -> dict[str, dict[str, float]]:
    """Held-out family score minus BASE's, per arm, family and source (seed means)."""
    official = fs.load_process_results()
    drop = frozenset(fs.unsupported_negations(fs.NEGATE_FALSE_VERDICTS))
    result = {}
    for src in fs.TASK_OF:
        base = [fs.read(r, src, official, drop) for r in ARMS["BASE"][1]]
        keys = sorted(base[0])
        base_f = {f: statistics.fmean(fs.family_scores(g, keys).get(f, float("nan")) for g in base) for f in fs.FAMILIES}
        for arm, (_, runs, held) in arms.items():
            present = [r for r in runs if list(Path(fs.ROOT + r).glob("samples_*"))] if runs else []
            if not present or not held:
                continue
            groups = [fs.read(r, src, official, drop) for r in present]
            for f in held:
                vals = [fs.family_scores(g, keys).get(f) for g in groups]
                if vals and vals[0] is not None and not math.isnan(vals[0]):
                    result.setdefault(arm, {})[f"{f} {src}"] = statistics.fmean(vals) - base_f[f]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--outcomes", action="store_true", help="also score held-out families of the trained runs")
    args = parser.parse_args()
    from datasets import load_from_disk

    agree = untrained_agreement()
    lines_out = ["# Family audit of the NLI4CT training arms (Check A1)", "",
                 ("Probes: 5-fold CV grouped by statement, logistic regression. Edit-only features: words added and "
                 "removed relative to the original statement, appended or not, section cut. 'Untrained agrees': share "
                 "of 8 untrained Qwen3-8B samples (answer prompt) giving the gold label, rows of the ALL rollout only."),
                 ""]
    scores = {}
    for arm, (path, _, _) in ARMS.items():
        if arm == "BASE":
            continue
        rows = list(load_from_disk(str(PILOT / path))["train"])
        table, scores[arm] = audit(rows, agree)
        lines_out += [f"## {arm} (`nli4ct_pilot/{path}`)", ""] + table + [""]
    keys = list(next(iter(scores.values())))
    lines_out += ["## Arm scores", "", "| score | " + " | ".join(scores) + " |", "|-|" + "-|" * len(scores)]
    for k in keys:
        cells = [v[k] if isinstance(v[k], str) else f"{v[k]:.3f}" if isinstance(v[k], float) else str(v[k])
                 for v in scores.values()]
        lines_out.append(f"| {k} | " + " | ".join(cells) + " |")
    if args.outcomes:
        observed = outcomes(ARMS)
        cols = sorted({c for v in observed.values() for c in v})
        lines_out += ["", "## Held-out families vs BASE (trained runs, seed means)", "",
                      "| arm | " + " | ".join(cols) + " |", "|-|" + "-|" * len(cols)]
        for arm, v in observed.items():
            lines_out.append(f"| {arm} | " + " | ".join(f"{v[c]:+.3f}" if c in v else "–" for c in cols) + " |")
        lines_out += ["", "## Audit scores vs held-out F2 (arms holding out F2; exploratory: results were known)", "",
                      "| audit score | Spearman vs F2 official (n) | Spearman vs F2 ours (n) |", "|-|-|-|"]
        for k in ("recipe-id probe", "edit-only probe", "conflicting-rule share", "fixed-label share", "variant C share",
                  "untrained disagreement (variants)"):
            cells = []
            for src in ("official", "ours"):
                pts = [(scores[a][k], observed[a][f"F2 {src}"]) for a in observed if f"F2 {src}" in observed[a]]
                rho = spearmanr(*zip(*pts, strict=True)).statistic if len(pts) > 2 else float("nan")
                cells.append(f"{rho:+.2f} ({len(pts)})")
            lines_out.append(f"| {k} | " + " | ".join(cells) + " |")
    args.report.write_text("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
