import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def _extract_label_and_correctness(sample: dict, use: str):
    """Return (true_label_str, pred_label_str, is_wrong: bool) for one sample.

    Handles two label conventions:
    - Text labels  (most tasks): doc["Label"] == pred_label space → compare directly.
    - Integer index labels (e.g. NLI4PR): doc["Label"] is "0"/"1" while pred_label
      is the text form. In this case correctness is determined by comparing
      doc["Label"] against sample["preds"] (also an integer index).

    Returns None if the sample has no Label field.
    """
    doc = sample.get("doc", {})
    raw_label = doc.get("Label")
    if raw_label is None:
        return None  # task doesn't use a "Label" field — skip

    true_label = str(raw_label).strip()

    pred_label = sample.get("pred_label") or sample.get(use) or sample.get("target", "")
    pred_label = pred_label.strip() if isinstance(pred_label, str) else str(pred_label)

    if true_label.isdigit():
        # Integer-index convention: compare preds (index) against true_label (index)
        preds_idx = sample.get("preds")
        is_wrong = (str(preds_idx) != true_label) if preds_idx is not None else True
    else:
        is_wrong = (true_label != pred_label)

    return true_label, pred_label, is_wrong


def analyze_label_errors(samples, use="majority"):
    total = Counter()
    mistakes = Counter()
    misclassified_as = defaultdict(Counter)

    iter_samples = samples.values() if isinstance(samples, dict) else samples
    for sample in iter_samples:
        result = _extract_label_and_correctness(sample, use)
        if result is None:
            continue
        true_label, pred_label, is_wrong = result

        total[true_label] += 1
        if is_wrong:
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

def process_folder(
    input_folders: List[str],
    use: str = "majority",
    file_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Recursively scan all JSON files in one or more folders.

    Args:
        input_folders: Paths to scan recursively for JSON result files.
        use:           Prediction key to look up in each sample (e.g. "majority").
        file_filter:   If set, only process files whose name contains this string.

    Returns:
        Dict mapping absolute file path → error/misclassification analysis.
    """
    results_by_file: Dict[str, Any] = {}
    for folder in input_folders:
        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        for file_path in Path(folder).rglob("*.json"):
            if file_filter and file_filter not in file_path.name:
                continue
            try:
                with open(file_path, "r", encoding="utf8") as f:
                    data = json.load(f)
                if "samples" in data:
                    if len(data["samples"]) != 1:
                        results_by_file[str(file_path)] = analyze_label_errors(data["samples"], use=use)
                    else:
                        # go one level deeper if only one sample per key (common in multi-turn CoT)
                        single_model = next(iter(data["samples"].keys()))
                        results_by_file[str(file_path)] = analyze_label_errors(data["samples"][single_model], use=use)
            except Exception as e:
                print(f"[Warning] Failed to process {file_path}: {e}")
    return results_by_file


def extract_dataset_name(file_path: str) -> str:
    """Extract dataset name from file path.
    
    Expected structures:
    - .../output_dir/dataset_name/0-shot/model/...
    - .../output_dir/multi-turn_CoT/dataset_CoT/model/...
    """
    path = Path(file_path)
    parts = path.parts
    
    # Special case: multi-turn_CoT has dataset AFTER it
    for i, part in enumerate(parts):
        if part == 'multi-turn_CoT':
            # Dataset is the next directory (e.g., MedNLI_CoT, MedMCQA_CoT)
            if i + 1 < len(parts):
                dataset = parts[i + 1]
                # Clean up the _CoT suffix if present
                return dataset.replace('_CoT', '')
    
    # For regular shot types (0-shot, 1-shot, few-shot), dataset is BEFORE them
    shot_indicators = ['0-shot', '1-shot', 'few-shot']
    for i, part in enumerate(parts):
        if part in shot_indicators:
            # The dataset is the part immediately before the shot indicator
            if i > 0:
                return parts[i - 1]
    
    # Fallback: look for outputs directory and skip 2 levels
    for i, part in enumerate(parts):
        if part in ['outputs', 'lm_harness_run-outputs']:
            # Skip the output directory name itself and get to dataset
            if i + 2 < len(parts):
                return parts[i + 2]
    
    # Last resort fallback to parent directory
    return path.parent.name


def extract_model_name(file_path: str) -> str:
    """Extract model name from file path."""
    path = Path(file_path)
    parts = path.parts
    
    # Look for common model patterns in path
    for part in parts:
        if any(model in part for model in ['Llama', 'Qwen', 'DeepSeek', 'Fleming', 'Mistral', 'Gemma', 'GPT']):
            # Clean up the model name
            if '__' in part:
                return part.split('__')[1] if len(part.split('__')) > 1 else part.split('__')[0]
            return part.replace('_', '-')
    
    # Fallback to parent directory name
    return path.parent.name


def results_to_dataframe(results_by_file: Dict) -> pd.DataFrame:
    """Convert results dictionary to a pandas DataFrame."""
    rows = []
    
    for file_path, result in results_by_file.items():
        dataset_name = extract_dataset_name(file_path)
        model_name = extract_model_name(file_path)
        
        for label, stats in result["errors"].items():
            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Label": label,
                "Total": stats["total"],
                "Mistakes": stats["mistakes"],
                "Accuracy": stats["accuracy"],
                "File": file_path
            }
            
            # Add misclassification details
            if label in result.get("misclassifications", {}):
                for misclass in result["misclassifications"][label]:
                    row[f"Misclassified_as_{misclass['as']}"] = misclass['count']
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path):
    """Save error breakdown to CSV."""
    df.to_csv(path, index=False)
    print(f"    - Saved CSV to:     {path}")


def save_markdown(results_by_file: Dict, path: Path):
    """Save error breakdown to Markdown with separate tables per dataset and model."""
    # Group results by dataset and model
    results_by_dataset_model = defaultdict(list)
    
    for file_path, result in results_by_file.items():
        dataset_name = extract_dataset_name(file_path)
        model_name = extract_model_name(file_path)
        key = (dataset_name, model_name)
        results_by_dataset_model[key].append({
            "file": file_path,
            "result": result
        })
    
    markdown_content = ["# Error Breakdown Analysis\n\n"]
    
    # Sort by dataset first, then model
    for (dataset, model) in sorted(results_by_dataset_model.keys()):
        markdown_content.append(f"## {dataset} - {model}\n\n")
        
        # Aggregate errors across all files for this dataset-model combo
        label_stats = defaultdict(lambda: {"total": 0, "mistakes": 0, "misclassifications": Counter()})
        
        for item in results_by_dataset_model[(dataset, model)]:
            result = item["result"]
            for label, stats in result["errors"].items():
                label_stats[label]["total"] += stats["total"]
                label_stats[label]["mistakes"] += stats["mistakes"]
                
                if label in result.get("misclassifications", {}):
                    for misclass in result["misclassifications"][label]:
                        label_stats[label]["misclassifications"][misclass["as"]] += misclass["count"]
        
        # Create summary table
        summary_rows = []
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            accuracy = (1 - stats["mistakes"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            
            row = {
                "Label": label,
                "Total": stats["total"],
                "Mistakes": stats["mistakes"],
                "Accuracy (%)": f"{accuracy:.2f}"
            }
            summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            markdown_content.append("### Summary\n\n")
            markdown_content.append(summary_df.to_markdown(index=False, disable_numparse=True))
            markdown_content.append("\n\n")
        
        # Create misclassification breakdown
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            if stats["misclassifications"]:
                markdown_content.append(f"### {label} - Misclassification Breakdown\n\n")
                
                misclass_rows = []
                total_mistakes = stats["mistakes"]
                for wrong_label, count in sorted(stats["misclassifications"].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_mistakes * 100) if total_mistakes > 0 else 0
                    misclass_rows.append({
                        "Predicted as": wrong_label,
                        "Count": count,
                        "% of Mistakes": f"{percentage:.1f}%"
                    })
                
                if misclass_rows:
                    misclass_df = pd.DataFrame(misclass_rows)
                    markdown_content.append(misclass_df.to_markdown(index=False, disable_numparse=True))
                    markdown_content.append("\n\n")
        
        markdown_content.append("---\n\n")
    
    # Write to file
    with open(path, 'w') as f:
        f.write(''.join(markdown_content))
    
    print(f"    - Saved Markdown to: {path}")


def save_visualizations(results_by_file: Dict, output_dir: Path, output_name: str):
    """Generate visualizations grouped by dataset and model."""
    # Group results by dataset and model
    results_by_dataset_model = defaultdict(list)
    
    for file_path, result in results_by_file.items():
        dataset_name = extract_dataset_name(file_path)
        model_name = extract_model_name(file_path)
        key = (dataset_name, model_name)
        results_by_dataset_model[key].append(result)
    
    for (dataset, model) in sorted(results_by_dataset_model.keys()):
        # Aggregate errors
        label_stats = defaultdict(lambda: {"total": 0, "mistakes": 0})
        
        for result in results_by_dataset_model[(dataset, model)]:
            for label, stats in result["errors"].items():
                label_stats[label]["total"] += stats["total"]
                label_stats[label]["mistakes"] += stats["mistakes"]
        
        if not label_stats:
            continue
        
        # Create accuracy bar chart
        labels = sorted(label_stats.keys())
        accuracies = [(1 - label_stats[l]["mistakes"] / label_stats[l]["total"]) * 100 
                     if label_stats[l]["total"] > 0 else 0 for l in labels]
        
        try:
            plt.figure(figsize=(10, 6))
            colors = plt.cm.RdYlGn([(acc/100) for acc in accuracies])
            bars = plt.bar(labels, accuracies, color=colors)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.1f}%',
                        ha='center', va='bottom', fontsize=10)
            
            plt.xlabel('Label')
            plt.ylabel('Accuracy (%)')
            plt.title(f'{dataset} - {model}\nLabel-wise Accuracy', fontsize=14, fontweight='bold')
            plt.ylim(0, 105)
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_').replace(':', '_')
            safe_model_name = model.replace('/', '_').replace(' ', '_').replace(':', '_')
            chart_path = output_dir / f"{output_name}_{safe_dataset_name}_{safe_model_name}_accuracy.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    - Saved chart:       {chart_path}")
        except Exception as e:
            print(f"    - Failed to create chart for '{dataset} - {model}': {e}")


def generate_summary_report(results_by_file: Dict, output_path: Path):
    """Generate a text summary report of error analysis."""
    # Group by dataset and model
    results_by_dataset_model = defaultdict(list)
    
    for file_path, result in results_by_file.items():
        dataset_name = extract_dataset_name(file_path)
        model_name = extract_model_name(file_path)
        key = (dataset_name, model_name)
        results_by_dataset_model[key].append(result)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ERROR BREAKDOWN ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Dataset-Model Combinations: {len(results_by_dataset_model)}\n")
        f.write(f"Total Files Processed:            {len(results_by_file)}\n")
        f.write("\n" + "-"*80 + "\n\n")
        
        for (dataset, model) in sorted(results_by_dataset_model.keys()):
            f.write(f"📊 {dataset} - {model}\n")
            f.write("-" * 40 + "\n")
            
            # Aggregate statistics
            label_stats = defaultdict(lambda: {"total": 0, "mistakes": 0})
            
            for result in results_by_dataset_model[(dataset, model)]:
                for label, stats in result["errors"].items():
                    label_stats[label]["total"] += stats["total"]
                    label_stats[label]["mistakes"] += stats["mistakes"]
            
            # Calculate overall accuracy
            total_samples = sum(s["total"] for s in label_stats.values())
            total_mistakes = sum(s["mistakes"] for s in label_stats.values())
            overall_accuracy = (1 - total_mistakes / total_samples) * 100 if total_samples > 0 else 0
            
            f.write(f"  Overall Accuracy:  {overall_accuracy:.2f}%\n")
            f.write(f"  Total Samples:     {total_samples}\n")
            f.write(f"  Total Mistakes:    {total_mistakes}\n\n")
            
            f.write("  Label-wise Performance:\n")
            for label in sorted(label_stats.keys()):
                stats = label_stats[label]
                accuracy = (1 - stats["mistakes"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                f.write(f"    • {label:20s}: {accuracy:6.2f}% ({stats['total']} samples, {stats['mistakes']} mistakes)\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"    - Saved summary:     {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-label error breakdown from lm-harness evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_error-breakdown.py --input_folders ./outputs \\
                                    --output_dir ./analysis --output_name errors

  python extract_error-breakdown.py --input_folders ./outputs/multi-turn_CoT ./outputs/cross-consistency \\
                                    --output_dir ./analysis --output_name errors \\
                                    --pred_key majority --file_filter "MedNLI"
        """
    )

    parser.add_argument("--input_folders", nargs="+", required=True,
                       help="One or more folders to search recursively for JSON result files")
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save outputs")
    parser.add_argument("--output_name", required=True,
                       help="Base name for output files (no extension)")
    parser.add_argument("--pred_key", default="majority",
                       help="Prediction key to look up in each sample (default: majority)")
    parser.add_argument("--file_filter", default=None,
                       help="Only process JSON files whose name contains this string")
    parser.add_argument("--no-csv", action="store_true",
                       help="Skip CSV output")
    parser.add_argument("--no-markdown", action="store_true",
                       help="Skip Markdown output")
    parser.add_argument("--no-summary", action="store_true",
                       help="Skip plain-text summary report")
    parser.add_argument("--no-charts", action="store_true",
                       help="Skip chart generation")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ERROR BREAKDOWN EXTRACTION SCRIPT")
    print("="*80)
    print(f"\nInput folders: {', '.join(args.input_folders)}")
    print(f"Output:        {args.output_dir}/{args.output_name}.*")
    print(f"Pred key:      {args.pred_key}")
    if args.file_filter:
        print(f"Filter:        {args.file_filter}")
    print("\n" + "-"*80 + "\n")

    # Process files
    print(f"[INFO] Collecting results from: {', '.join(args.input_folders)}")
    results_by_file = process_folder(args.input_folders, use=args.pred_key, file_filter=args.file_filter)

    if not results_by_file:
        print("[ERROR] No valid results found! Check your input folders.")
        return

    print(f"[INFO] Processed {len(results_by_file)} files\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Generating outputs...\n")

    # Save JSON (always) - enhance with dataset and model information
    json_output = {}
    for file_path, result in results_by_file.items():
        dataset_name = extract_dataset_name(file_path)
        model_name = extract_model_name(file_path)
        json_output[file_path] = {
            "dataset": dataset_name,
            "model": model_name,
            "file_path": file_path,
            **result  # unpacks "errors" and "misclassifications"
        }

    json_path = output_dir / f"{args.output_name}.json"
    with safe_open_w(str(json_path)) as f:
        json.dump(json_output, f, indent=4)
    print(f"    - Saved JSON to:     {json_path}")

    # Save CSV
    if not args.no_csv:
        df = results_to_dataframe(results_by_file)
        save_csv(df, output_dir / f"{args.output_name}.csv")

    # Save Markdown
    if not args.no_markdown:
        save_markdown(results_by_file, output_dir / f"{args.output_name}.md")

    # Save summary report
    if not args.no_summary:
        generate_summary_report(results_by_file, output_dir / f"{args.output_name}_summary.txt")

    # Save visualizations
    if not args.no_charts:
        save_visualizations(results_by_file, output_dir, args.output_name)

    print("\n[SUCCESS] All outputs generated successfully!\n")
    print("="*80)
    print(f"✅ Error breakdown extracted successfully to {args.output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()