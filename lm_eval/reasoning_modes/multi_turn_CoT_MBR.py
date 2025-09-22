import json
import warnings

from joblib import Parallel, delayed
from lm_eval.reasoning_modes.reasoning_utils import *
from mbrs.metrics import MetricBLEU, MetricBLEURT
from mbrs.decoders import DecoderMBR
from tqdm import tqdm


def MBR_reasoning_chains(reasoning_chains_per_document):
    def process_doc(doc_id, decoder, reasoning_chains):
        metric_output = decoder.decode(
            reasoning_chains[doc_id], reasoning_chains[doc_id]
        )
        return reasoning_chains[doc_id][metric_output.idx[0]]

    bleu = MetricBLEU(MetricBLEU.Config(num_workers=128))
    bleurt = MetricBLEURT(MetricBLEURT.Config(batch_size=64, model="lucadiliello/bleurt-tiny-512"))

    mbr_metrics = [("bleu", bleu, bleu.Config)] # ("bluert", bleurt, bleurt.Config)

    res = {"bleu" : []}

    for name, metric, config in mbr_metrics:
        decoder = DecoderMBR(config, metric)

        if name == "bleu":
            # ⚡ Parallelize across documents
            res[name] = Parallel(n_jobs=8)(
                delayed(process_doc)(doc_id, decoder, reasoning_chains_per_document)
                for doc_id in tqdm(reasoning_chains_per_document, desc="BLEU")
            )

        elif name == "bleurt":

            for doc_id in tqdm(reasoning_chains_per_document, desc="BLEURT"):
                # still per-doc, but BLEURT batches internally with batch_size=64
                metric_output = decoder.decode(
                    reasoning_chains_per_document[doc_id],
                    reasoning_chains_per_document[doc_id],
                )

                res[name].append(
                    reasoning_chains_per_document[doc_id][metric_output.idx[0]]
                )

    return res

def mode_multi_turn_CoT_MBR(args: argparse.Namespace) -> Dict:
    if len(args.reasoning_models) != 1 or len(args.answering_models) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning model and one answering model. {args.reasoning_models=} {args.answering_models=}")

    if len(args.reasoning_tasks) != 1 or len(args.answering_tasks) != 1:
        print(f"[WARNING] For multi-turn mode, please provide exactly one reasoning task and one answering task. {args.reasoning_tasks=} {args.answering_tasks=}")

    reasoning_model = args.reasoning_models[0]
    answering_model = args.answering_models[0]

    reasoning_task = args.reasoning_tasks[0]
    answering_task = args.answering_tasks[0]


    predictions_per_input_doc = None
    if args.vote_file is None:
        raise ValueError("For multi-turn CoT-MBR mode, please provide a vote_file to load reasoning samples from (Currently only stubbing functionality).")
        reasoning_outputs = run_reasoning(args)[reasoning_model][reasoning_task]
        reasoning_chains_per_document = extract_multiple_reasoning_chains_per_document(reasoning_outputs)
    else:    
        predictions_per_input_doc = json.load(open(args.vote_file, "r"))["samples"]

        reasoning_chains_per_document = {}
        for doc_id, info in predictions_per_input_doc.items():
            reasoning_chains_per_document[doc_id] = info["doc"]["Reasoning_Chains"]

    full_task_name = answering_task.replace(":", "_")
    task_def = tasks.get_task_dict([full_task_name])[full_task_name]
    doc_to_text_module = f"lm_eval.tasks.{parse_task_spec(answering_task)[0]}.utils"
    base_dataset = load_base_dataset_from_task(answering_task.replace(":", "_"))


    ### TO:DO Remove stubs after testing
    stub_mini_trial = {}
    stub_mini_trial["0"] = reasoning_chains_per_document["0"]
    stub_mini_trial["1"] = reasoning_chains_per_document["1"]
    mbr_reasoning_chains = MBR_reasoning_chains(stub_mini_trial)

    metrics_results = {"results" : {mbr_metric: {} for mbr_metric in mbr_reasoning_chains}}

    for mbr_metric in mbr_reasoning_chains:
        reasoning_chain_list = mbr_reasoning_chains[mbr_metric]

        dataset_with_reasoning = inject_reasoning_into_dataset(base_dataset, reasoning_chain_list)

        raw_output = run_answering_for_dataset(
            args=args,
            answering_model= answering_model,
            answering_task_name= full_task_name,
            dataset_with_reasoning= dataset_with_reasoning,
            doc_to_text_module= doc_to_text_module
        )

        metrics_results["results"][mbr_metric] = format_results_dict(raw_output["results"][full_task_name])

        for sample in raw_output["samples"][full_task_name]:
            doc_id = sample["doc_id"]
            if doc_id not in predictions_per_input_doc:
                predictions_per_input_doc[doc_id] = {
                    "doc" : copy.deepcopy(sample["doc"]),
                    "preds": [],
                    "pred_probs": [],
                }

                predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"] = []

            predictions_per_input_doc[doc_id]["doc"]["Reasoning_Chains"].append(sample["doc"]["Reasoning_Chain"])

            pred_probs = [prob[0][0] for prob in sample["resps"]]
            predictions_per_input_doc[doc_id]["pred_probs"].append(pred_probs)    
            predictions_per_input_doc[doc_id]["preds"].append(pred_probs.index(max(pred_probs)))

    return {
        "mode": "multi-turn_CoT-MBR",
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
        "reasoning_task": reasoning_task,
        "answering_task": answering_task,
        **metrics_results,
        "samples" : predictions_per_input_doc
    }