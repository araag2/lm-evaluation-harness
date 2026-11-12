"""Is the held-out F2 failure label noise, or training and prompting? (Check DQ1, `_Reports/NLI4CT_Training/HYPOTHESES.md`)

Reads only files that already exist: eval samples of every arm (as `family_scores.py` reads them), the untrained
rollouts of the pilot training set, Gemma's label audits and the pilot training data. Writes:
- `--report`: (1) F2 by arm: score, unparsed and forced shares, score where original and variant both parsed, median
  answer length, and a paired comparison of the 2,048-token arms on rows all of them parsed; (2) the direction of F2
  flips; (3) training-row suspects (no untrained sample right); (4) test items every arm gets wrong; (5) held-out F2
  differences against BASE with and without those items (bootstrap over base statements);
- `--sheet` and `--key`: a blind hand-check sheet (training suspects and controls, failed and random test items)
  and the key that unblinds it.

    python scripts/data_quality.py --report R.md --sheet S.md --key K.json
"""

import argparse
import ast
import glob
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import family_scores as fs

PIPELINE = Path(__file__).resolve().parents[2] / "OpenCTEval-Data-Pipeline"
TRAIN = PIPELINE / "outputs/training/data/nli4ct_pilot/ALL"
ROLLOUTS = PIPELINE / "outputs/training/data/nli4ct_pilot_rollouts"
AUDITS = [PIPELINE / "outputs/analysis/recipe_review/label_audit.jsonl", fs.NEGATE_FALSE_VERDICTS]
ARMS = {
    "BASE": ["fam-ce-base", "fam-ce-base-s7", "fam-ce-base-s13"],
    "CE A": ["fam-ce-a", "fam-ce-a-s7"],
    "step 0": ["ans-think2048-bf"],  # -bf: budget-forced backfill (job 626914); grpo-bc-s7 and RFT forced at eval
    "think 2,048": ["think2048s-bf"],
    "think 4,096": ["think4096s-bf"],
    "GRPO A": ["grpo-a-bf", "grpo-a-s7-bf"],
    "GRPO B+C": ["grpo-bc-bf", "grpo-bc-s7"],
    "RFT": ["rft", "rft-s7"],
}
CAPPED = ["step 0", "think 2,048", "GRPO A", "GRPO B+C", "RFT"]  # 2,048-token thinking arms
SOURCES = ["ours", "official"]
FAMILY = "F2"
LETTER = {None: "?", 0: "C", 1: "E"}  # label indices as the harness scores them (pos_answers)
F2_KINDS = {"append_definition", "Text_appended"}
SHEET_TRAIN = {"append_definition": 5, "original": 4, "recast": 3, "evidence_only": 2, "negate_false-recast": 2}
SHEET_TEST = 4  # per source, failed and random each
SEED, ITERATIONS = 0, 1000
EDIT = {
    "original": "No edit: the statement as written in the NLI4CT dataset.",
    "append_definition": "A definition of an unrelated medical term was added. The label should stay the same.",
    "Text_appended": "The SemEval organizers added an unrelated definition. The label should stay the same.",
    "recast": "The statement was reworded. The meaning, and so the label, should stay the same.",
    "substitute-syntactic": "Some words were swapped for synonyms. The label should stay the same.",
    "evidence_only": "The trial text was cut down to the lines that matter. The label should stay the same.",
    "evidence_removed": "The supporting lines were removed from the trial text, so it should be Contradiction.",
    "negate_false-recast": "A false statement was negated, so it should now be Entailment.",
    "contrary": "The statement was rewritten into the opposite claim, so it should be Contradiction.",
    "numeric": "A number was changed; a model re-checked the label.",
    "biomedical-recast": "A medical term was replaced; a model re-checked the label.",
}


def kind(doc: dict, source: str) -> str:
    return doc["Intervention"] if source == "official" else (doc.get("augmentation") or {}).get("mode", "")


def items(run: str, source: str, drop: frozenset) -> dict[tuple, dict]:
    """F2 variants of one run by (base key, position): hit (family-score rule), flip direction, gold, text length."""
    run_name, _, _ = run.partition("@")
    path = sorted(glob.glob(f"{fs.ROOT}{run_name}/samples_{fs.TASK_OF[source]}_0-shot_*.jsonl"))[-1]
    words = [len(x.get("text", "").split()) for x in map(json.loads, open(path)) if kind(x["doc"], source) in F2_KINDS]
    groups = fs.read(run, source, fs.load_process_results(), drop)
    out = {}
    for key, rows in groups.items():
        original = next((r for r in rows if r[0]), None)
        if original is None:
            continue
        for j, (is_original, gold, pred, family, mode, forced) in enumerate(rows):
            if is_original or family != FAMILY:
                continue
            hit = pred is not None and pred == original[2]
            out[(key, j)] = {"hit": hit, "pred": pred, "orig_pred": original[2], "gold": gold, "forced": forced,
                             "mode": mode}
    return out, statistics.median(words)


def load_arms(drop):
    data = {}
    for arm, runs in ARMS.items():
        present = [r for r in runs if glob.glob(f"{fs.ROOT}{r}/samples_*")]
        if present:
            data[arm] = {s: [items(r, s, drop) for r in present] for s in SOURCES}
    return data


def mean_hit(per_run, keys=None):
    return statistics.fmean(
        statistics.fmean(v["hit"] for k, v in run.items() if keys is None or k in keys) for run, _ in per_run
    )


def section_scores(data) -> list[str]:
    lines = ["## 1. F2 by arm (held out in fold A; seed means)", "",
             "| arm | source | score | unparsed | forced | score, both parsed | median words |", "|-|-|-|-|-|-|-|"]
    for arm, by_source in data.items():
        for s in SOURCES:
            runs = by_source[s]
            un = statistics.fmean(statistics.fmean(v["pred"] is None for v in r.values()) for r, _ in runs)
            fo = statistics.fmean(statistics.fmean(v["forced"] for v in r.values()) for r, _ in runs)
            both = statistics.fmean(
                statistics.fmean(v["hit"] for v in r.values() if v["pred"] is not None and v["orig_pred"] is not None) for r, _ in runs
            )
            med = statistics.fmean(m for _, m in runs)
            lines.append(f"| {arm} | {s} | {mean_hit(runs):.3f} | {un:.3f} | {fo:.3f} | {both:.3f} | {med:.0f} |")
    lines += ["", "Paired: 2,048-token arms on the F2 rows that every one of them parsed (original and variant):", "",
              "| source | rows | " + " | ".join(a for a in CAPPED if a in data) + " |",
              "|-|-|" + "-|" * sum(a in data for a in CAPPED)]
    for s in SOURCES:
        common = None
        for arm in (a for a in CAPPED if a in data):
            for run, _ in data[arm][s]:
                ok = {k for k, v in run.items() if v["pred"] is not None and v["orig_pred"] is not None}
                common = ok if common is None else common & ok
        vals = [f"{mean_hit(data[a][s], common):.3f}" for a in CAPPED if a in data]
        lines.append(f"| {s} | {len(common)} | " + " | ".join(vals) + " |")
    return lines


def section_direction(data) -> list[str]:
    lines = ["", "## 2. Direction of F2 flips (variant answer moved off the original's answer)", "",
             "| arm | source | flips | E→C | C→E | unparsed | accuracy vs gold |", "|-|-|-|-|-|-|-|"]
    for arm, by_source in data.items():
        for s in SOURCES:
            c, n, acc = Counter(), 0, []
            for run, _ in by_source[s]:
                for v in run.values():
                    acc.append(v["pred"] == v["gold"])
                    if v["hit"]:
                        continue
                    n += 1
                    c["unparsed" if v["pred"] is None else f"{LETTER[v['orig_pred']]}→{LETTER[v['pred']]}"] += 1
            share = {k: c[k] / n if n else 0 for k in ("E→C", "C→E", "unparsed")}
            lines.append(f"| {arm} | {s} | {n / len(by_source[s]):.0f} | {share['E→C']:.2f} | {share['C→E']:.2f} | "
                         f"{share['unparsed']:.2f} | {statistics.fmean(acc):.3f} |")
    return lines


def training_suspects():
    """Training rows no untrained sample answers right (and, where present, ≤ 1 of 8 thinking samples)."""
    def right(path):
        return {r["id"]: (r["mode"], r["label"], sum(s["answer"] == r["label"] for s in r["samples"]))
                for r in map(json.loads, path.open())}

    untrained, think = right(ROLLOUTS / "untrained_ALL.jsonl"), right(ROLLOUTS / "think2048_A.jsonl")
    verdicts = {}
    for path in AUDITS:
        for r in map(json.loads, path.open()):
            verdicts[r["id"]] = r["verdict"]
    suspects = {i for i, (_, _, k) in untrained.items() if k == 0 and (i not in think or think[i][2] <= 1)}
    lines = ["", "## 3. Training-row suspects (pilot ALL, 1,922 rows)", "",
             "A suspect: 0 of 8 untrained samples right (answer prompt), and at most 1 of 8 with thinking at 2,048 "
             "tokens where that rollout exists. The suspect rate bounds label noise from above: it also counts rows "
             "the model gets wrong for its own reasons.", "",
             "| mode | rows | suspects | rate | Gemma agrees with gold (audited suspects) |", "|-|-|-|-|-|"]
    expected = {"Entailment": "YES", "Contradiction": "NO"}
    for mode in sorted({m for m, _, _ in untrained.values()}):
        ids = [i for i, (m, _, _) in untrained.items() if m == mode]
        sus = [i for i in ids if i in suspects]
        audited = [i for i in sus if i in verdicts]
        agree = sum(verdicts[i] == expected[untrained[i][1]] for i in audited)
        lines.append(f"| {mode} | {len(ids)} | {len(sus)} | {len(sus) / len(ids):.2f} | "
                     f"{f'{agree} of {len(audited)}' if audited else '–'} |")
    return lines, suspects, untrained


def section_consensus(data):
    """F2 test items that every arm (every seed) gets wrong."""
    failed, lines = {}, ["", "## 4. F2 test items every arm gets wrong", ""]
    for s in SOURCES:
        keys = set.intersection(*(set(run) for arm in data.values() for run, _ in arm[s]))
        failed[s] = {k for k in keys if not any(run[k]["hit"] for arm in data.values() for run, _ in arm[s])}
        lines.append(f"- {s}: {len(failed[s])} of {len(keys)} variants ({len(failed[s]) / len(keys):.1%}).")
    return lines, failed


def section_clean(data, failed) -> list[str]:
    lines = ["", "## 5. Held-out F2 minus BASE, all items vs without the all-fail items (95% bootstrap CI)", "",
             "| arm | source | all items | without all-fail |", "|-|-|-|-|"]
    rng = random.Random(SEED)
    for s in SOURCES:
        bases = sorted({k[0] for run, _ in data["BASE"][s] for k in run})
        samples = [[rng.choice(bases) for _ in bases] for _ in range(ITERATIONS)]

        def score(arm, sample, clean, s=s):
            weights = Counter(sample)
            vals = []
            for run, _ in data[arm][s]:
                hits = [(v["hit"], weights[k[0]]) for k, v in run.items()
                        if weights[k[0]] and not (clean and k in failed[s])]
                vals.append(sum(h * w for h, w in hits) / sum(w for _, w in hits))
            return statistics.fmean(vals)

        for arm in (a for a in data if a != "BASE"):
            cells = []
            for clean in (False, True):
                point = score(arm, bases, clean) - score("BASE", bases, clean)
                d = sorted(score(arm, x, clean) - score("BASE", x, clean) for x in samples)
                cells.append(f"{point:+.3f} [{d[int(.025 * ITERATIONS)]:+.3f}, {d[int(.975 * ITERATIONS) - 1]:+.3f}]")
            lines.append(f"| {arm} | {s} | {cells[0]} | {cells[1]} |")
    return lines


def lines_of(section) -> list[str]:
    if isinstance(section, str):
        section = ast.literal_eval(section) if section.startswith("[") else [section]
    return [line.strip() for line in section or []]


def render(n, doc, original, mode, gold):
    out = [f"### {n}", ""]
    for side in ("Primary", "Secondary"):
        body = lines_of(doc.get(f"{side}_Section"))
        if body:
            out += [f"**{side} trial, {doc.get('Section_id', '')}:**", ""] + [f"- {t}" for t in body if t] + [""]
    if mode != "original":
        out += [f"**Original statement:** {original}", ""]
    out += [f"**Statement to check:** {doc['Statement']}", "", f"**What was done:** {EDIT[mode]}", "",
            f"**Gold label:** {gold}", "", "Your mark: ok / odd / ?  —  note:", ""]
    return out


def sheet(suspects, untrained, failed, data, drop):
    from datasets import load_from_disk

    rng = random.Random(SEED)
    train = {r["id"]: r for r in load_from_disk(str(TRAIN))["train"]}
    picks = []
    for mode, n in SHEET_TRAIN.items():
        pool = sorted(i for i, (m, _, _) in untrained.items() if m == mode)
        sus = rng.sample([i for i in pool if i in suspects], n)
        ctl = rng.sample([i for i in pool if i not in suspects], n)
        picks += [("train suspect", i) for i in sus] + [("train control", i) for i in ctl]
    entries = []
    for group, i in picks:
        row = train[i]
        entries.append((group, {"id": i, "mode": row["augmentation"]["mode"], "untrained_right": untrained[i][2]},
                        row, train[row["source_id"]]["Statement"], row["augmentation"]["mode"], row["Label"]))
    for s in SOURCES:
        path = sorted(glob.glob(f"{fs.ROOT}ans-think2048/samples_{fs.TASK_OF[s]}_0-shot_*.jsonl"))[-1]
        docs = [json.loads(line)["doc"] for line in open(path)]
        groups = defaultdict(list)
        for doc in docs:
            if s == "official":
                causal = doc["Causal_type"]
                causal = ast.literal_eval(causal) if isinstance(causal, str) else causal
                base = doc["id"] if doc["Intervention"] == "Base_Statement" else causal[1]
            else:
                base = doc["source_id"]
            groups[base].append(doc)
        keys = set(next(iter(data["BASE"][s]))[0])
        failed_k = sorted(failed[s])
        random_k = sorted(k for k in keys if k not in failed[s])
        for group, chosen in (("test all-fail", rng.sample(failed_k, SHEET_TEST)),
                              ("test random", rng.sample(random_k, SHEET_TEST))):
            for key, j in chosen:
                doc = groups[key][j]
                assert kind(doc, s) in F2_KINDS, (key, j)  # positions match family_scores.read's grouping
                original = next(d for d in groups[key] if (d.get("Intervention") == "Base_Statement"
                                                           or d.get("id") == d.get("source_id")))
                mode = doc.get("Intervention") if s == "official" else doc["augmentation"]["mode"]
                answers = {a: [run[(key, j)]["pred"] for run, _ in data[a][s]] for a in data}
                entries.append((group, {"source": s, "key": key, "position": j, "answers": answers},
                                doc, original["Statement"], mode, doc["Label"]))
    rng.shuffle(entries)
    md = ["# NLI4CT data check: 48 items to read (blind)", "",
          "Mark each item `ok` (the label makes sense), `odd` (looks wrong) or `?` (can't tell), with an optional "
          "note. Reply in one line, e.g. `1 ok, 2 odd (definition changes meaning), 3 ?, ...`.", "",
          "- **Entailment**: the trial text supports the statement.",
          "- **Contradiction**: the trial text does not support it (it is false, or the text does not say so).",
          "- You are not asked to be a clinician: judge whether it reads sensibly.", ""]
    key = {}
    for n, (group, info, doc, original, mode, gold) in enumerate(entries, 1):
        md += render(n, doc, original, mode, gold)
        key[n] = {"group": group, **info}
    return md, key


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--key", type=Path)
    args = parser.parse_args()
    drop = frozenset(fs.unsupported_negations(fs.NEGATE_FALSE_VERDICTS))
    data = load_arms(drop)
    lines = [f"Arms: {', '.join(f'{a} ({len(ARMS[a])})' for a in data)}.", ""]
    lines += section_scores(data) + section_direction(data)
    train_lines, suspects, untrained = training_suspects()
    consensus_lines, failed = section_consensus(data)
    lines += train_lines + consensus_lines + section_clean(data, failed)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.sheet:
        md, key = sheet(suspects, untrained, failed, data, drop)
        args.sheet.write_text("\n".join(md) + "\n")
        args.key.write_text(json.dumps(key, indent=1) + "\n")
        print(f"sheet: {len(key)} items -> {args.sheet}")


if __name__ == "__main__":
    main()
