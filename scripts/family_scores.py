"""Held-out recipe-family scores and transfer ratios (OpenCTEval `_Reports/NLI4CT_Training/HYPOTHESES.md`, protocol v2).

Every variant of a family is scored against its original: a variant with the original's gold label must keep the
original's prediction (consistency), and a variant with another gold label must move off it (faithfulness). A
family's score is the share of its variants handled right, so families that keep labels, flip them or both are
on one scale. Test sources:
- `ours`: AugExp v3 test (our generator, statements never trained on);
- `official`: the SemEval-2024 test (the organizers' hand-written edits), for F1-F4;
- `evidence`: EvidenceAblation, for F6.

Amendment a1 (2026-11-07, before any GRPO result): in `ours`, F4 keeps only the negate_false-recast rows that Gemma's
support check judged supported (`--negate-false-verdicts`, the filter training already used,
`build_nli4ct_recipes.py --supported`). NLI4CT's Contradiction means "not supported", so negating an unsupported
statement need not make it entailed: 111 of the 238 test rows were judged unsupported. `--all-negations` restores
the unfiltered rows. negate_true-recast (test only) is also reported on its own: the R4 probe for a learned
"negation means Entailment" shortcut.

A map is one method's runs: BASE, ALL and the folds A, B, C (several runs per arm = seeds, averaged; `run@claims`
reads that run's `*_claims` tasks). The transfer
ratio of family f is T_f = (held-out - BASE) / (ALL - BASE), where held-out is the fold that leaves f out. With two
maps, the difference of their held-out scores gets a 95% CI from resampling base statements.

Budget forcing (`generate.py`, 2026-11-11): rows whose thought hit the token limit can carry a forced answer
(`"forced": true`). They count as answered by default; `--strict` scores them as unparsed (the rule before
forcing), and the "forced" column shows their share per family in the held-out arm.

    python scripts/family_scores.py --map ce:BASE=pilot-ce-base,ALL=pilot-ce-all+pilot-ce-all-s7,A=...,B=...,C=... \
        [--map vrex:...] [--output report.md]
"""

import argparse
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from semeval2024_scores import load_process_results

from lm_eval.opencteval.augmentation import sliced_results
from lm_eval.tasks.SemEval_NLI4CT.utils import pos_answers


ROOT = "outputs/NLI4CT_AugExp_p2/single-turn/0-shot/_user_home_aguimas_data_PhD_Active_Dev_OpenCTEval-Data-Pipeline_outputs_merged_nli4ct-qwen3-8b-"
FAMILY_OF = {
    "ours": {
        "recast": "F1", "substitute-syntactic": "F1", "append_definition": "F2", "numeric": "F3",
        "numeric-recast": "F3", "negate_false-recast": "F4", "negate_true-recast": "F4", "contrary": "F4",
        "biomedical-recast": "F5",
    },
    "official": {
        "Paraphrase": "F1", "Text_appended": "F2", "Numerical_paraphrase": "F3",
        "Numerical_contradiction": "F3", "Contradiction": "F4",
    },
    "evidence": {"evidence_only": "F6", "evidence_removed": "F6"},
}
TASK_OF = {
    "ours": "SemEval_NLI4CT_AugExp_v3_test",
    "official": "SemEval_NLI4CT_2024",
    "evidence": "SemEval_NLI4CT_EvidenceAblation",
}
NEGATE_FALSE = "negate_false-recast"
NEGATE_FALSE_VERDICTS = (
    Path(__file__).resolve().parents[2] / "OpenCTEval-Data-Pipeline/outputs/analysis/recipe_review/negate_false_support.jsonl"
)
PROBES = {"ours": ["negate_true-recast"]}  # modes also scored on their own (R4)
FOLDS = {"A": ["F2", "F5"], "B": ["F1", "F3"], "C": ["F4", "F6"]}
FAMILIES = ["F1", "F2", "F3", "F4", "F5", "F6"]
ITERATIONS, SEED = 1000, 0


def as_evidence_view(doc: dict) -> dict:
    """The EvidenceAblation task's own row mapping (its utils module sits in `0-shot/`, not importable by name)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "lm_eval/tasks/SemEval_NLI4CT/0-shot/utils.py"
    spec = importlib.util.spec_from_file_location("semeval_0shot_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals()["as_evidence_view"] = module.as_evidence_view
    return module.as_evidence_view(doc)


def unsupported_negations(path: Path) -> set[str]:
    """Ids of negate_false-recast rows the support check did not judge supported (amendment a1)."""
    with path.open() as lines:
        return {r["id"] for r in map(json.loads, lines) if r["verdict"] != "YES"}


def read(run: str, source: str, official, drop: frozenset = frozenset(), strict: bool = False) -> dict[str, list[tuple]]:
    """Rows grouped by base statement: (is_original, gold, pred, family or None, mode, forced).

    `drop`: row ids left out; `strict`: a budget-forced answer counts as unparsed.
    """
    run, _, variant = run.partition("@")  # "run@claims" reads the *_claims tasks (H1, statement bottleneck)
    task = TASK_OF[source] + (f"_{variant}" if variant else "")
    files = sorted(glob.glob(f"{ROOT}{run}/samples_{task}_0-shot_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no {task} samples for {run}")
    groups = defaultdict(list)
    with open(files[-1]) as lines:
        for line in lines:
            sample = json.loads(line)
            doc, resps, forced = sample["doc"], sample["filtered_resps"], sample.get("forced", False)
            if strict and forced:
                resps = [""]
            if source == "official":
                item, kind = official(doc, resps)["consistency"], doc["Intervention"]
            elif source == "evidence":
                item, kind = sliced_results(as_evidence_view(doc), resps, pos_answers)["consistency"], doc["view"]
            else:
                if doc["id"] in drop:
                    continue
                item, kind = sliced_results(doc, resps, pos_answers)["consistency"], doc["augmentation"]["mode"]
            groups[item[0]].append((item[1], item[2], item[3], FAMILY_OF[source].get(kind), kind, forced))
    return groups


def family_scores(groups: dict, keys: list[str], by_mode: bool = False) -> dict[str, float]:
    """Share of variants handled right, per family (or per recipe mode with `by_mode`)."""
    hits = defaultdict(list)
    for key in keys:
        rows = groups[key]
        original = next((r for r in rows if r[0]), None)
        if original is None:
            continue
        for is_original, gold, pred, family, mode, _ in rows:
            if is_original or family is None:
                continue
            group = mode if by_mode else family
            if gold == original[1]:
                hits[group].append(float(pred is not None and pred == original[2]))
            else:
                hits[group].append(float(pred is not None and pred != original[1]))
    return {f: statistics.fmean(v) for f, v in hits.items() if v}


def forced_share(groups: dict, keys: list[str], family: str) -> float:
    rows = [r for key in keys for r in groups[key] if not r[0] and r[3] == family]
    return sum(r[5] for r in rows) / len(rows) if rows else 0.0


def held_out(arms: dict, family: str) -> list:
    return next(arms[fold] for fold, out in FOLDS.items() if family in out)


def map_scores(arms: dict, source: str, keys: list[str]) -> dict[str, dict[str, float]]:
    """Per family: BASE, ALL, held-out (seed means) and the transfer ratio."""
    def mean(runs):
        per = [family_scores(groups, keys) for groups in runs]
        return {f: statistics.fmean(p[f] for p in per) for f in per[0]}

    base, full = mean(arms["BASE"]), mean(arms["ALL"])
    out = {}
    for family in base:
        held = mean(held_out(arms, family))[family]
        gain = full[family] - base[family]
        ratio = (held - base[family]) / gain if abs(gain) >= 0.02 else float("nan")
        forced = statistics.fmean(forced_share(g, keys, family) for g in held_out(arms, family))
        out[family] = {"BASE": base[family], "ALL": full[family], "held": held, "T": ratio, "forced": forced}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", action="append", required=True, help="NAME:ARM=run[+run],...")
    parser.add_argument("--sources", nargs="+", default=list(TASK_OF), choices=list(TASK_OF))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negate-false-verdicts", type=Path, default=NEGATE_FALSE_VERDICTS)
    parser.add_argument("--all-negations", action="store_true", help="keep unsupported negate_false rows (pre-a1)")
    parser.add_argument("--strict", action="store_true", help="budget-forced answers count as unparsed")
    args = parser.parse_args()

    official = load_process_results()
    drop = frozenset() if args.all_negations else frozenset(unsupported_negations(args.negate_false_verdicts))
    maps = {}
    for spec in args.map:
        name, arms = spec.split(":", 1)
        maps[name] = {arm: runs.split("+") for arm, runs in (a.split("=") for a in arms.split(","))}
    lines = []
    for source in args.sources:
        data = {
            name: {arm: [read(r, source, official, drop, args.strict) for r in runs] for arm, runs in arms.items()}
            for name, arms in maps.items()
        }
        keys = sorted(next(iter(next(iter(data.values())).values()))[0])
        lines += [f"\n### {source} ({len(keys)} base statements)", "", "| map | family | BASE | ALL | held-out | T | forced |", "|-|-|-|-|-|-|-|"]
        scores = {name: map_scores(arms, source, keys) for name, arms in data.items()}
        for name, fams in scores.items():
            for f in FAMILIES:
                if f in fams:
                    s = fams[f]
                    lines.append(f"| {name} | {f} | {s['BASE']:.3f} | {s['ALL']:.3f} | {s['held']:.3f} | {s['T']:.2f} | {s['forced']:.3f} |")
        if PROBES.get(source):
            lines += ["", "R4 probe, per recipe (seed means):", "", "| map | arm | mode | score |", "|-|-|-|-|"]
            for name, arms in data.items():
                for arm, runs in arms.items():
                    for mode in PROBES[source]:
                        per = [family_scores(g, keys, by_mode=True).get(mode, float("nan")) for g in runs]
                        lines.append(f"| {name} | {arm} | {mode} | {statistics.fmean(per):.3f} |")
        names = list(data)
        if len(names) >= 2:
            ref = names[0]
            rng = random.Random(SEED)
            draws = defaultdict(list)
            for _ in range(ITERATIONS):
                sample = [rng.choice(keys) for _ in keys]
                held = {
                    n: {f: statistics.fmean(family_scores(g, sample).get(f, float("nan")) for g in held_out(data[n], f))
                        for f in scores[ref]}
                    for n in names
                }
                for n in names[1:]:
                    diffs = [held[n][f] - held[ref][f] for f in scores[ref]]
                    draws[n].append(statistics.fmean(diffs))
            lines += ["", f"Held-out score minus {ref}, mean over families, 95% CI:", "", "| map | diff | CI | families won |", "|-|-|-|-|"]
            for n in names[1:]:
                d = sorted(draws[n])
                point = statistics.fmean(scores[n][f]["held"] - scores[ref][f]["held"] for f in scores[ref])
                won = sum(scores[n][f]["held"] > scores[ref][f]["held"] for f in scores[ref])
                lo, hi = d[int(0.025 * ITERATIONS)], d[int(0.975 * ITERATIONS) - 1]
                lines.append(f"| {n} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {won}/{len(scores[ref])} |")
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
