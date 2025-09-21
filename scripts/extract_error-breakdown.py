import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def analyze_label_errors(samples, use="majority"):
    total = Counter()
    mistakes = Counter()
    misclassified_as = defaultdict(Counter)

    if isinstance(samples, dict):
        for sample_id, sample in samples.items():
            true_label = sample["doc"]["Label"].strip()
            pred_label = sample.get(use)

            total[true_label] += 1
            if true_label != pred_label:
                mistakes[true_label] += 1
                misclassified_as[true_label][pred_label] += 1

    elif isinstance(samples, list):
        for sample in samples:
            true_label = sample["doc"]["Label"].strip()
            pred_label = sample.get(use)

            total[true_label] += 1
            if true_label != pred_label:
                mistakes[true_label] += 1
                misclassified_as[true_label][pred_label] += 1

    label_order = sorted(total.keys())

    # combine mistakes + accuracy
    errors_summary = {}
    misclass_summary = {}

    for lbl in label_order:
        if lbl in total:
            count = total[lbl]
            error_count = mistakes.get(lbl, 0)
            accuracy = 1 - error_count / count
            errors_summary[lbl] = {
                "total": count,
                "mistakes": error_count,
                "accuracy": round(accuracy, 4) * 100
            }

            # misclassification breakdown as % of mistakes
            if error_count > 0:
                misclass_summary[lbl] = []
                for wrong in sorted(misclassified_as[lbl].keys()):
                    wrong_count = misclassified_as[lbl][wrong]
                    misclass_summary[lbl].append({
                        "as": wrong,
                        "count": wrong_count,
                        "percent": f"{wrong_count / error_count * 100:.1f}%"
                    })

    return {
        "errors": errors_summary,
        "misclassifications": misclass_summary
    }

def process_folder(input_folder, use="majority"):
    """
    Recursively scan all JSON files in a folder and subfolders,
    process them, and return results as a dict keyed by file path.
    """
    results_by_file = {}
    for file_path in Path(input_folder).rglob("*.json"):
        try:
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            if "samples" in data:
                if len(data["samples"]) != 1:
                    results_by_file[str(file_path)] = analyze_label_errors(data["samples"], use=use)
                else:
                    # go one level deeper if only one sample (common in multi-turn CoT)
                    single_model = next(iter(data["samples"].keys()))
                    results_by_file[str(file_path)] = analyze_label_errors(data["samples"][single_model], use=use)
        except Exception as e:
            print(f"[Warning] Failed to process {file_path}: {e}")
    return results_by_file


def main():
    parser = argparse.ArgumentParser(description="Extract evaluation results from lm-harness runs recursively.")
    parser.add_argument("--input_folder", required=True, help="Folder containing JSON result files")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--output_name", required=True, help="Base name for output files (no extension)")
    parser.add_argument("--use", default="majority", help="Prediction key to use (default=majority)")

    args = parser.parse_args()

    print(f"Collecting results from folders: {args.input_folder}\nSaving to {args.output_dir}/{args.output_name}")

    results_by_file = process_folder(args.input_folder, use=args.use)

    with safe_open_w(os.path.join(args.output_dir, f"{args.output_name}.json")) as f:
        json.dump(results_by_file, f, indent=4)

    print(f"Results saved to {args.output_dir}/{args.output_name}.json")


if __name__ == "__main__":
    main()