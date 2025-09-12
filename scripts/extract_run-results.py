import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path


def extract_metrics(result_dict, prefix=""):
    metrics = {}

    for k, v in result_dict.items():
        if "," in k and "_stderr" not in k:
            metric_name = k.split(",")[0]
            metrics[f"{prefix}{metric_name}"] = v
    return metrics

def collect_results(folders, file_filter=None):
    results = []

    for folder in folders:
        for root, _, files in os.walk(folder):

            for file in files:
                if not file.endswith(".json") or (not file.startswith("results_") and not file.startswith("Summary_")):
                    continue

                if file_filter and file_filter not in file:
                    continue   # Skip files that don’t match filter

                path = os.path.join(root, file)
                with open(path, "r") as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")
                        continue

                results_field = list(data.get("results", {}).keys())
                search_field = None
                if len(results_field) == 1:
                    search_field = data["results"][results_field[0]]

                for task_name, result_dict in search_field.items():
                    config = data.get("configs", {}).get(task_name, {})
                    model_args = data.get("config", {}).get("model_args", "N/A")
                    pretrained = config.get("metadata", {}).get("pretrained", "N/A")

                    result_entry = {
                        "Dataset": task_name,
                        "Model": pretrained,
                        "Model Args": model_args,
                        "Path": root
                    }

                    result_entry.update(extract_metrics(result_dict))
                    results.append(result_entry)

    return results


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"    - Saved CSV to:     {path}")


def save_markdown(df, path):
    df = df.drop(columns=["Model Args", "Path"], errors="ignore")

    # Multiply all numeric (float/int) columns by 100
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = round(df[numeric_cols] * 100, 2)

    df.to_markdown(path, index=False)
    print(f"    - Saved Markdown to:{path}")


def save_latex(df, path):
    df = df.drop(columns=["Model Args", "Path"], errors="ignore")
    
    # Multiply all numeric (float/int) columns by 100
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols] * 100

    latex_code = df.to_latex(index=False, float_format="%.2f", escape=False)
    with open(path, "w") as f:
        f.write(latex_code)
    print(f"    - Saved LaTeX to:   {path}")


def save_bar_charts(df, output_dir, output_name):
    # Identify potential metric columns
    reserved_cols = {"Dataset", "Model", "Model Args", "Path"}
    metric_columns = [col for col in df.columns if col not in reserved_cols and pd.api.types.is_numeric_dtype(df[col])]

    for metric in metric_columns:
        try:
            df_plot = df[["Dataset", "Model", metric]].dropna().copy()
            if df_plot.empty:
                print(f"[INFO] Skipping metric '{metric}': no valid data to plot.")
                continue

            # Scale values for plotting
            df_plot[metric] *= 100

            df_plot["Label"] = df_plot["Dataset"] + "\n" + df_plot["Model"]
            df_plot = df_plot.sort_values(metric, ascending=False)

            plt.figure(figsize=(max(10, 0.4 * len(df_plot)), 6))  # Scale width by number of bars
            plt.bar(df_plot["Label"], df_plot[metric], color="skyblue")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric)
            plt.title(f"{metric} by Dataset / Model")
            plt.tight_layout()

            chart_path = output_dir / f"{output_name}_{metric}_barchart.png"
            plt.savefig(chart_path)
            plt.close()
            print(f"    - Saved chart to:  {chart_path}")
        except Exception as e:
            print(f"    - Failed to plot metric '{metric}': {e}")

def save_outputs(results, output_dir, output_name, output_csv=True, output_md=True, output_latex=True, output_barcharts=True):
    df = pd.DataFrame(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_csv:
        save_csv(df, output_dir / f"{output_name}.csv")
    if output_md:
        save_markdown(df, output_dir / f"{output_name}.md")
    if output_latex:
        save_latex(df, output_dir / f"{output_name}.tex")
    if output_barcharts:
        save_bar_charts(df, output_dir, output_name)

def main():
    parser = argparse.ArgumentParser(description="Extract evaluation results from lm-harness runs to various formats.")

    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--output_name", required=True, help="Base name for output files (no extension)")
    parser.add_argument("--input_folders", nargs="+", required=True, help="List of folders to search for results")
    parser.add_argument("--file_filter", default=None, help="Only process JSON files containing this string in the name")


    args = parser.parse_args()

    print(f"Collecting results from folders: {args.input_folders}\nSaving to {args.output_dir}/{args.output_name}")

    results = collect_results(args.input_folders)
    save_outputs(results, args.output_dir, args.output_name)

if __name__ == "__main__":
    main()