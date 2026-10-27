"""Ask an LLM judge whether each view of an Entailment statement still supports it, to check the rule labels
of the evidence-ablation set.

Gold evidence is annotated for sufficiency, not exhaustiveness, so a section with the evidence removed may
still support the statement through another line. The judge reads the statement and one view's section(s)
and answers SUPPORTED or NOT_SUPPORTED. The full view is judged too, as a check on the judge itself.
Verdicts are appended as they come, so a rerun only asks for the rows still missing; a row whose call keeps
failing is left out and retried on the next run. `--data` also takes an augmentation export with evidence views
(`OpenCTEval-Data-Pipeline/scripts/augmentation/build_evidence_views.py`); `--views` limits what is judged.

    python scripts/verify_evidence_views.py --env ../OpenCTEval-Data-Pipeline/configs/gemma4.env
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from lm_eval.opencteval.evidence import EVIDENCE_ONLY, EVIDENCE_REMOVED, FULL
from lm_eval.tasks.SemEval_NLI4CT.utils import format_sections


VIEWS = [FULL, EVIDENCE_ONLY, EVIDENCE_REMOVED]


PROMPT = """You check clinical trial statements against Clinical Trial Report (CTR) sections.

Section name: {section}

{premise}

Statement: {statement}

Using only the CTR text above, is the statement fully supported? A statement is not supported when the text
contradicts it or when the text does not contain the information needed to conclude it.
Answer with one word: SUPPORTED or NOT_SUPPORTED."""


def premise(row: dict) -> str:
    text = f"Primary trial CTR section:\n{format_sections(row['Primary_Section'])}"
    if row.get("Secondary_Section"):
        text += f"\n\nSecondary trial CTR section:\n{format_sections(row['Secondary_Section'])}"
    return text


def as_view_row(row: dict) -> dict:
    """A test-set row as is; an export row with `view`, `source_id` and `Gold_Label` filled in from `augmentation`."""
    if "view" in row:
        return row
    aug = row["augmentation"]
    view = FULL if aug["mode"] == "original" else aug["mode"]
    return {**row, "view": view, "Gold_Label": aug.get("source_label", row["Label"])}


def read_env(path: Path) -> dict:
    pairs = (
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )
    return {key.strip(): value.strip().strip("\"'") for key, value in pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/nli4ct_evidence_ablation/test.jsonl")
    )
    parser.add_argument(
        "--env", type=Path, required=True, help="File with BASE_URL, API_KEY, MODEL"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/nli4ct_evidence_ablation/judge.jsonl")
    )
    parser.add_argument(
        "--views", nargs="+", default=VIEWS, choices=VIEWS, help="Views to judge"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    env = read_env(args.env)
    client = OpenAI(base_url=env["BASE_URL"], api_key=env["API_KEY"], timeout=300)
    rows = [as_view_row(json.loads(line)) for line in args.data.open()]
    ablated = {
        row["source_id"]
        for row in rows
        if row["view"] in (EVIDENCE_ONLY, EVIDENCE_REMOVED)
    }
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.open()}
    todo = [
        r
        for r in rows
        if r["Gold_Label"] == "Entailment"
        and r["source_id"] in ablated
        and r["view"] in args.views
        and r["id"] not in done
    ]
    print(f"{len(todo)} rows to judge ({len(done)} already judged)", flush=True)
    write_lock = Lock()

    def judge(row: dict) -> dict:
        content = PROMPT.format(
            section=row["Section_id"], premise=premise(row), statement=row["Statement"]
        )
        for attempt in range(args.retries):
            try:
                reply = client.chat.completions.create(
                    model=env["MODEL"],
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=2048,
                )
                break
            except (APIConnectionError, APIStatusError, APITimeoutError) as error:
                print(
                    f"{row['id']}: attempt {attempt + 1} failed: {error!s:.120}",
                    flush=True,
                )
                time.sleep(30 * (attempt + 1))
        else:
            return None
        answer = (reply.choices[0].message.content or "").strip().upper()
        verdict = (
            "NOT_SUPPORTED"
            if "NOT_SUPPORTED" in answer
            else ("SUPPORTED" if "SUPPORTED" in answer else "UNPARSED")
        )
        result = {
            "id": row["id"],
            "source_id": row["source_id"],
            "view": row["view"],
            "verdict": verdict,
        }
        with write_lock, args.output.open("a") as out:
            out.write(json.dumps(result) + "\n")
        return result

    with ThreadPoolExecutor(args.workers) as pool:
        missing = sum(result is None for result in pool.map(judge, todo))
    verdicts = [json.loads(line) for line in args.output.open()]
    if missing:
        print(f"{missing} rows failed every attempt; rerun to retry them")
    for view in VIEWS:
        got = [v["verdict"] for v in verdicts if v["view"] == view]
        counts = {k: got.count(k) for k in ("SUPPORTED", "NOT_SUPPORTED", "UNPARSED")}
        print(f"{view:16} n={len(got)} {counts}")


if __name__ == "__main__":
    main()
