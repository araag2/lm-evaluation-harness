"""Untrained Qwen3-8B on TREC CT 2023 in the think style (HYPOTHESES.md, Check T1), and row-matched subsets of
earlier runs.

`eval` (GPU, Slurm): a seeded sample of `--per-patient` pairs per patient from TREC_CT_2023_0-shot. The prompt is
the 0-shot prompt without its trailing "Answer:" cue, with Qwen3's thinking mode and its thinking sampling. Answers
cut off by `--max-tokens` are budget-forced: close the thought, add "Answer:" and continue greedily for a few
tokens. It writes `samples_TREC_CT_2023_0-shot_<time>.jsonl`, which `scripts/trec_ct_scores.py` reads.

`subset` (CPU): copies of other runs' TREC_CT_2023 samples restricted to the rows of an `eval` output, so the
scorer compares runs on the same pairs.

    python scripts/trec_think_eval.py eval --out outputs/TREC_CT_Training/single-turn/0-shot/qwen3-8b-think2048
    python scripts/trec_think_eval.py subset --rows THINK_SAMPLES.jsonl --out DIR NAME=SAMPLES.jsonl ...
"""

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


HARNESS = Path(__file__).resolve().parents[1]
LABELS = ["not relevant", "possibly relevant", "definitely relevant"]
LABEL = re.compile(r"(not|possibly|definitely)\s+relevant", re.IGNORECASE)
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}  # Qwen3's thinking-mode sampling
FORCE_CUE = "Answer:"
FORCE_TOKENS = 8
CUT_MARGIN = 4


def parse(text: str) -> str:
    """The last label named after the thought; "" when the thought is cut off or no label is named."""
    if "<think>" in text and "</think>" not in text:
        return ""
    found = LABEL.findall(text.rsplit("</think>", maxsplit=1)[-1])
    return f"{found[-1].lower()} relevant" if found else ""


def load_docs(per_patient: int, seed: int) -> list[dict]:
    import os

    from lm_eval.tasks import TaskManager

    os.chdir(HARNESS)
    task = next(iter(TaskManager().load_task_or_group(["TREC_CT_2023_0-shot"]).values()))
    by_patient = defaultdict(list)
    for doc in task.test_docs():
        by_patient[str(doc["query_id"])].append(doc)
    rng = random.Random(seed)
    docs = []
    for patient in sorted(by_patient, key=int):
        rows = by_patient[patient]
        docs += rng.sample(rows, min(per_patient, len(rows)))
    return docs


def evaluate(args):
    import sys

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(HARNESS / "lm_eval/tasks/TREC_CT"))
    import utils  # the harness's TREC prompt

    docs = load_docs(args.per_patient, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory,
              max_model_len=args.max_model_len, seed=args.seed)

    def chat(doc):
        question = utils.doc_to_text(doc).rstrip().removesuffix("Answer:").rstrip()
        return tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)

    prompts = [chat(d) for d in docs]
    params = SamplingParams(n=1, max_tokens=args.max_tokens, seed=args.seed, **THINKING_SAMPLING)
    texts = [o.outputs[0].text for o in llm.generate(prompts, params)]
    answers = [parse(t) for t in texts]
    cut = [len(tokenizer(t, add_special_tokens=False).input_ids) >= args.max_tokens - CUT_MARGIN for t in texts]
    todo = [i for i, a in enumerate(answers) if not a and cut[i]]
    if todo:
        suffix = {i: ("\n</think>\n\n" if "</think>" not in texts[i] else "\n") + FORCE_CUE for i in todo}
        outs = llm.generate([prompts[i] + texts[i] + suffix[i] for i in todo],
                            SamplingParams(n=1, max_tokens=FORCE_TOKENS, temperature=0.0))
        for i, o in zip(todo, outs, strict=True):
            texts[i] = texts[i] + suffix[i] + o.outputs[0].text
            answers[i] = parse(texts[i])
    args.out.mkdir(parents=True, exist_ok=True)
    forced = set(todo)
    path = args.out / f"samples_TREC_CT_2023_0-shot_{datetime.now().isoformat()}.jsonl"
    with path.open("w") as out:
        for i, (doc, answer, text) in enumerate(zip(docs, answers, texts, strict=True)):
            out.write(json.dumps({"doc_id": i, "doc": doc, "filtered_resps": [answer], "text": text,
                                  "forced": i in forced}) + "\n")
    print(f"{path}: {len(docs)} rows, {sum(not a for a in answers)} unparsed, {len(todo)} forced, "
          f"{sum(cut)} cut off", flush=True)


def subset(args):
    keep = {json.loads(line)["doc"]["id"] for line in args.rows.open()}
    args.out.mkdir(parents=True, exist_ok=True)
    for item in args.runs:
        name, path = item.split("=", 1)
        target = args.out / f"{name}.jsonl"
        n = 0
        with target.open("w") as out, Path(path).open() as source:
            for line in source:
                if json.loads(line)["doc"]["id"] in keep:
                    out.write(line)
                    n += 1
        print(f"{target}: {n} of {len(keep)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--model", default="Qwen/Qwen3-8B")
    ev.add_argument("--per-patient", type=int, default=150)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--max-tokens", type=int, default=2048)
    ev.add_argument("--max-model-len", type=int, default=8192)
    ev.add_argument("--gpu-memory", type=float, default=0.90)
    sb = sub.add_parser("subset")
    sb.add_argument("--rows", type=Path, required=True)
    sb.add_argument("--out", type=Path, required=True)
    sb.add_argument("runs", nargs="+", help="NAME=SAMPLES.jsonl")
    args = parser.parse_args()
    evaluate(args) if args.command == "eval" else subset(args)


if __name__ == "__main__":
    main()
