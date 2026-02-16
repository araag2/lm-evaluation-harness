import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def normalize_metric_name(metric_name: str) -> str:
    """Normalize metric names by removing suffixes like ',none' and cleaning up."""
    # Remove common suffixes
    for suffix in [",none", ",all", "_score"]:
        if metric_name.endswith(suffix):
            metric_name = metric_name[:-len(suffix)]
    return metric_name.strip()

def extract_metrics(result_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Extract metrics from result dictionary with normalization."""
    metrics = {}

    for k, v in result_dict.items():
        # Skip normalized versions and stderr
        if 'norm' in k or 'stderr' in k:
            continue
        
        # Normalize metric name
        if "," in k:
            metric_name = normalize_metric_name(k.split(",")[0])
        else:
            metric_name = normalize_metric_name(k)
        
        metrics[f"{prefix}{metric_name}"] = v
    
    return metrics

def collect_results_from_default_file(path: str) -> List[Dict[str, Any]]:
    """Extract results from standard lm-eval results file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARNING] Invalid JSON in {path}: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return []
    
    results = []
    
    for task_name, result_dict in data.get("results", {}).items():
        config = data.get("configs", {}).get(task_name, {})
        model_args = data.get("config", {}).get("model_args", "N/A")
        pretrained = config.get("metadata", {}).get("pretrained", "N/A")
        
        # Extract a cleaner model name
        model_name = pretrained.split("/")[-1] if "/" in pretrained else pretrained
        
        # Get timestamp from file path or metadata
        timestamp = Path(path).parent.name if Path(path).parent.name.startswith("202") else "N/A"

        result_entry = {
            "Dataset": task_name,
            "Model": model_name,
            "Model_Full": pretrained,
            "Timestamp": timestamp,
            "Model Args": model_args,
            "Path": path
        }

        result_entry.update(extract_metrics(result_dict))
        results.append(result_entry)

    return results

def collect_results_from_summary_file(path: str) -> List[Dict[str, Any]]:
    """Extract results from multi-turn or aggregated summary files."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARNING] Invalid JSON in {path}: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return []
    
    results = []

    if "acc" in data.get("results", {}).keys() or "acc,none" in data.get("results", {}).keys():
        model_args = data.get("config", {}).get("model_args", "N/A")

        result_entry = {
            "Dataset": data.get("reasoning_task", {}),
            "Model": data.get("reasoning_model", {}).split(",")[0][11:],
            "Model Args": model_args,
            "Path": path
        }

        result_entry.update(extract_metrics(data.get("results", {})))
        results.append(result_entry)

    elif "majority" in data.get("results", {}).keys() or "rrf" in data.get("results", {}).keys() or "condorcet" in data.get("results", {}).keys() or "logits" in data.get("results", {}).keys() or "borda" in data.get("results", {}).keys():

        for voting_method, result_dict in data.get("results", {}).items():
            model_args = data.get("config", {}).get("model_args", "N/A")

            result_entry = {
                "Dataset": data.get("reasoning_task", {}),
                "Voting_Method": voting_method,
                "Model": data.get("reasoning_model", {}).split(",")[0][11:],
                "Model Args": model_args,
                "Path": path
            }

            result_entry.update(extract_metrics(result_dict))
            results.append(result_entry)     

    else:
        for task_name, result_dict in data.get("results", {}).items():
            model_args = data.get("config", {}).get("model_args", "N/A")

            result_entry = {
                "Dataset": task_name,
                "Model": data.get("reasoning_model", {}).split(",")[0][11:],
                "Model Args": model_args,
                "Path": path
            }

            result_entry.update(extract_metrics(result_dict))
            results.append(result_entry)        

    return results


def collect_results(folders: List[str], file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Recursively collect all results from specified folders."""
    results = []
    files_processed = 0
    files_failed = 0

    for folder in folders:
        if not os.path.exists(folder):
            print(f"[WARNING] Folder does not exist: {folder}")
            continue
            
        for root, _, files in os.walk(folder):
            for file in files:
                if not file.endswith(".json"):
                    continue
                    
                if file_filter and file_filter not in file:
                    continue
                
                file_path = os.path.join(root, file)
                
                if file.startswith("results_"):
                    file_results = collect_results_from_default_file(file_path)
                    if file_results:
                        results.extend(file_results)
                        files_processed += 1
                    else:
                        files_failed += 1
                        
                elif file.startswith("Summary_"):
                    file_results = collect_results_from_summary_file(file_path)
                    if file_results:
                        results.extend(file_results)
                        files_processed += 1
                    else:
                        files_failed += 1
    
    print(f"\n[INFO] Processed {files_processed} files successfully, {files_failed} failed")
    print(f"[INFO] Collected {len(results)} result entries\n")
    return results


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics (mean, median, std) for each model across all tasks."""
    if "Model" not in df.columns:
        return pd.DataFrame()
    
    # Identify numeric metric columns (exclude metadata)
    reserved_cols = {"Dataset", "Model", "Model_Full", "Timestamp", "Model Args", "Path", "Voting_Method"}
    numeric_cols = [col for col in df.columns 
                   if col not in reserved_cols 
                   and pd.api.types.is_numeric_dtype(df[col])
                   and not col.endswith("_stderr")]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    # Group by model and compute statistics
    summary = df.groupby("Model")[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    summary = summary.round(4)
    
    return summary

def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking columns for each metric (1 = best)."""
    reserved_cols = {"Dataset", "Model", "Model_Full", "Timestamp", "Model Args", "Path", "Voting_Method"}
    numeric_cols = [col for col in df.columns 
                   if col not in reserved_cols 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    for col in numeric_cols:
        # Assume higher is better for most metrics (acc, f1, etc.)
        # Lower is better for loss, perplexity
        if any(x in col.lower() for x in ['loss', 'perplexity', 'error']):
            df[f"{col}_rank"] = df.groupby("Dataset")[col].rank(ascending=True, method='min')
        else:
            df[f"{col}_rank"] = df.groupby("Dataset")[col].rank(ascending=False, method='min')
    
    return df

def save_csv(df: pd.DataFrame, path: Path):
    """Save results to CSV with proper ordering."""
    # Reorder columns: Dataset, Model, metrics, then metadata
    priority_cols = ["Dataset", "Model"]
    metadata_cols = ["Model_Full", "Timestamp", "Model Args", "Path"]
    
    existing_priority = [col for col in priority_cols if col in df.columns]
    existing_metadata = [col for col in metadata_cols if col in df.columns]
    metric_cols = [col for col in df.columns if col not in existing_priority + existing_metadata]
    
    ordered_cols = existing_priority + metric_cols + existing_metadata
    df = df[ordered_cols]
    
    df.to_csv(path, index=False)
    print(f"    - Saved CSV to:     {path}")


def save_markdown(df: pd.DataFrame, path: Path):
    """Save results to Markdown with separate tables per dataset."""
    df = df.copy()
    df = df.drop(columns=["Model Args", "Model_Full", "Timestamp"], errors="ignore")

    # Sort by model priority
    model_order = ["Fleming", "Panacea", "Qwen", "Llama", "deepseek", "DeepSeek", "Mistral", "Gemma"]

    def get_model_priority(model_name: str) -> int:
        for i, key in enumerate(model_order):
            if key in str(model_name):
                return i
        return len(model_order)
    
    def to_ordinal(n: float) -> str:
        """Convert a number to ordinal string (1st, 2nd, 3rd, etc.)."""
        if pd.isna(n):
            return "N/A"
        n = int(n)
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    if "Model" in df.columns:
        df = df.sort_values(
            by="Model",
            key=lambda col: col.map(get_model_priority),
        )

    # Identify ALL numeric columns (including ranks this time)
    numeric_cols = [col for col in df.select_dtypes(include=["float", "int"]).columns]
    # Separate non-rank numeric columns for formatting
    non_rank_numeric = [col for col in numeric_cols if not col.endswith("_rank")]
    
    # Build markdown content with separate tables per dataset
    markdown_content = ["# Evaluation Results\n\n"]
    
    if "Dataset" not in df.columns:
        # If no dataset column, create single table
        df_formatted = df.copy()
        
        # Filter out columns where all values are NaN
        cols_to_keep = []
        for col in df_formatted.columns:
            if col in numeric_cols:
                if not df_formatted[col].isna().all():
                    cols_to_keep.append(col)
            else:
                cols_to_keep.append(col)
        df_formatted = df_formatted[cols_to_keep]
        
        # Format non-rank numeric columns
        for col in non_rank_numeric:
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x * 100:.2f}" if pd.notna(x) else "N/A"
                )
        if "Path" in df_formatted.columns:
            df_formatted["Path"] = df_formatted["Path"].apply(
                lambda x: f"[📁]({x})" if pd.notna(x) else ""
            )
        
        # Format rank columns as ordinals (1st, 2nd, 3rd, etc.)
        rank_cols = [col for col in df_formatted.columns if col.endswith("_rank")]
        for rank_col in rank_cols:
            df_formatted[rank_col] = df_formatted[rank_col].apply(to_ordinal)
        
        markdown_content.append(df_formatted.to_markdown(index=False, disable_numparse=True))
    else:
        # Create separate table for each dataset
        for dataset in sorted(df["Dataset"].unique()):
            markdown_content.append(f"## {dataset}\n\n")
            
            dataset_df = df[df["Dataset"] == dataset].copy()
            
            # Filter out ALL columns where all values are NaN for this dataset (INCLUDING rank columns)
            cols_to_keep = []
            for col in dataset_df.columns:
                # Skip the grouping Dataset column - it's redundant with alias
                if col == "Dataset":
                    continue
                # For numeric columns (including ranks), check if all NaN
                if col in numeric_cols:
                    if not dataset_df[col].isna().all():
                        cols_to_keep.append(col)
                else:
                    # Keep all non-numeric columns
                    cols_to_keep.append(col)
            
            dataset_df = dataset_df[cols_to_keep]
            
            # Rename 'alias' to 'Dataset' if it exists
            if "alias" in dataset_df.columns:
                dataset_df = dataset_df.rename(columns={"alias": "Dataset"})
            
            # Reorder columns: Dataset (alias), Model, Path, then metrics
            ordered_cols = []
            if "Dataset" in dataset_df.columns:  # This is the renamed 'alias' column
                ordered_cols.append("Dataset")
            if "Model" in dataset_df.columns:
                ordered_cols.append("Model")
            if "Path" in dataset_df.columns:
                ordered_cols.append("Path")
            
            # Add remaining columns (metrics)
            for col in dataset_df.columns:
                if col not in ordered_cols:
                    ordered_cols.append(col)
            
            dataset_df = dataset_df[ordered_cols]
            
            # Highlight best values per metric (only for non-rank numeric columns)
            for metric in non_rank_numeric:
                if metric not in dataset_df.columns:
                    continue
                
                # Determine if higher or lower is better
                higher_is_better = not any(x in metric.lower() for x in ['loss', 'perplexity', 'error'])
                
                subset = dataset_df[metric]
                
                # Skip if no valid numeric values
                if subset.isna().all() or len(subset) == 0:
                    continue
                
                try:
                    # Find best index
                    if higher_is_better:
                        best_idx = subset.idxmax()
                    else:
                        best_idx = subset.idxmin()
                    
                    # Check if best_idx is valid (not NaN)
                    if pd.isna(best_idx):
                        # Just format without highlighting
                        for idx in subset.index:
                            val = dataset_df.loc[idx, metric]
                            if pd.notna(val):
                                dataset_df.loc[idx, metric] = f"{val * 100:.2f}"
                            else:
                                dataset_df.loc[idx, metric] = "N/A"
                        continue
                    
                    # Format all values
                    for idx in subset.index:
                        val = dataset_df.loc[idx, metric]
                        if pd.notna(val):
                            if idx == best_idx:
                                dataset_df.loc[idx, metric] = f"**{val * 100:.2f}**"
                            else:
                                dataset_df.loc[idx, metric] = f"{val * 100:.2f}"
                        else:
                            dataset_df.loc[idx, metric] = "N/A"
                except (ValueError, TypeError):
                    # If idxmax/idxmin fails, just format the values
                    for idx in subset.index:
                        val = dataset_df.loc[idx, metric]
                        if pd.notna(val):
                            dataset_df.loc[idx, metric] = f"{val * 100:.2f}"
                        else:
                            dataset_df.loc[idx, metric] = "N/A"
            
            # Add links to result files if Path column exists
            if "Path" in dataset_df.columns:
                dataset_df["Path"] = dataset_df["Path"].apply(
                    lambda x: f"[📁]({x})" if pd.notna(x) else ""
                )
            
            # Format rank columns as ordinals (1st, 2nd, 3rd, etc.)
            rank_cols = [col for col in dataset_df.columns if col.endswith("_rank")]
            for rank_col in rank_cols:
                dataset_df[rank_col] = dataset_df[rank_col].apply(to_ordinal)
            
            markdown_content.append(dataset_df.to_markdown(index=False, disable_numparse=True))
            markdown_content.append("\n\n")
    
    # Write to file
    with open(path, 'w') as f:
        f.write(''.join(markdown_content))
    
    print(f"    - Saved Markdown to:{path}")


def save_latex(df, path):
    df = df.drop(columns=["Model Args", "Path"], errors="ignore")
    
    # Multiply all numeric (float/int) columns by 100
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols] * 100

    # Create LaTeX table with Datasets in Columns, with metrics as sub-columns, and Models as rows, with Voting_Method as sub-rows if exists
    if "Voting_Method" in df.columns:
        df = df.pivot_table(index=["Model", "Voting_Method"], columns="Dataset", aggfunc='first')
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
    else:
        df = df.pivot_table(index="Model", columns="Dataset", aggfunc='first')
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()

    latex_code = df.to_latex(index=False, float_format="%.2f", escape=False)


    with open(path, "w") as f:
        f.write(latex_code)
    print(f"    - Saved LaTeX to:   {path}")


def save_visualizations(df: pd.DataFrame, output_dir: Path, output_name: str):
    """Generate visualizations grouped by dataset."""
    reserved_cols = {"Dataset", "Model", "Model_Full", "Timestamp", "Model Args", "Path", "Voting_Method"}
    metric_columns = [col for col in df.columns 
                     if col not in reserved_cols 
                     and pd.api.types.is_numeric_dtype(df[col])
                     and not col.endswith("_rank")]

    if not metric_columns:
        print("[INFO] No numeric metrics found for visualization")
        return

    if "Dataset" not in df.columns or "Model" not in df.columns:
        print("[INFO] Missing Dataset or Model columns for visualization")
        return

    # 1. Combined charts per dataset (all metrics in one figure)
    for dataset in sorted(df["Dataset"].unique()):
        dataset_df = df[df["Dataset"] == dataset]
        
        # Find metrics that have at least one valid value for this dataset
        available_metrics = []
        for metric in metric_columns:
            if metric in dataset_df.columns and not dataset_df[metric].isna().all():
                available_metrics.append(metric)
        
        if not available_metrics:
            continue
        
        try:
            # Create subplots: one row per metric
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
            
            # Handle single metric case
            if n_metrics == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, available_metrics):
                df_plot = dataset_df[["Model", metric]].dropna().copy()
                if df_plot.empty:
                    ax.text(0.5, 0.5, 'No data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{metric}")
                    continue
                
                df_plot[metric] *= 100
                df_plot = df_plot.sort_values(metric, ascending=False)
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))
                ax.bar(df_plot["Model"], df_plot[metric], color=colors)
                ax.set_ylabel(f'{metric} (%)')
                ax.set_title(f"{metric}")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
                
                # Format x-axis labels
                ax.set_xticklabels(df_plot["Model"], rotation=45, ha='right')
            
            # Add overall title
            fig.suptitle(f'{dataset}', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_').replace(':', '_')
            chart_path = output_dir / f"{output_name}_{safe_dataset_name}_combined.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    - Saved chart:         {chart_path}")
        except Exception as e:
            print(f"    - Failed to create chart for '{dataset}': {e}")

    # 2. Heatmap per Dataset (Models as rows, Metrics as columns)
    try:
        for dataset in sorted(df["Dataset"].unique()):
            dataset_df = df[df["Dataset"] == dataset]
            
            # Find metrics with valid data
            available_metrics = [m for m in metric_columns 
                               if m in dataset_df.columns and not dataset_df[m].isna().all()]
            
            if not available_metrics:
                continue
            
            # Pivot: Models as rows, Metrics as columns
            pivot = dataset_df.pivot_table(
                values=available_metrics,
                index='Model',
                aggfunc='first'
            )
            
            if pivot.empty:
                continue
            
            pivot = pivot * 100  # Convert to percentage
            
            # Dynamic figure size based on content
            fig_width = max(12, len(pivot.columns) * 1.5)
            fig_height = max(6, len(pivot) * 0.8)
            
            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Score (%)'}, 
                       vmin=0, vmax=100, linewidths=0.5)
            plt.title(f'{dataset}\n(Models × Metrics)', fontsize=14, fontweight='bold')
            plt.xlabel('Metrics')
            plt.ylabel('Models')
            plt.tight_layout()
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_').replace(':', '_')
            heatmap_path = output_dir / f"{output_name}_heatmap_{safe_dataset_name}.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    - Saved heatmap:       {heatmap_path}")
    except Exception as e:
        print(f"    - Failed to create heatmaps: {e}")

def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate a text summary report of key findings."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write(f"Total Evaluations: {len(df)}\n")
        if "Model" in df.columns:
            f.write(f"Models Evaluated:  {df['Model'].nunique()}\n")
            f.write(f"  - {', '.join(df['Model'].unique())}\n")
        if "Dataset" in df.columns:
            f.write(f"Datasets:          {df['Dataset'].nunique()}\n")
        f.write("\n" + "-"*80 + "\n\n")
        
        # Best performing model per dataset
        if "Dataset" in df.columns and "Model" in df.columns:
            reserved_cols = {"Dataset", "Model", "Model_Full", "Timestamp", "Model Args", "Path"}
            metric_cols = [col for col in df.columns 
                          if col not in reserved_cols 
                          and pd.api.types.is_numeric_dtype(df[col])
                          and not col.endswith("_rank")]
            
            if metric_cols:
                f.write("TOP PERFORMING MODELS BY DATASET\n")
                f.write("-"*80 + "\n\n")
                
                for dataset in sorted(df["Dataset"].unique()):
                    f.write(f"📊 {dataset}:\n")
                    dataset_df = df[df["Dataset"] == dataset]
                    
                    for metric in metric_cols[:3]:  # Top 3 metrics
                        if metric not in dataset_df.columns:
                            continue
                        
                        # Skip if all values are NaN
                        if dataset_df[metric].isna().all():
                            continue
                        
                        best_idx = dataset_df[metric].idxmax()
                        
                        # Check if best_idx is valid (not NaN)
                        if pd.isna(best_idx):
                            continue
                        
                        best_model = dataset_df.loc[best_idx, "Model"]
                        best_score = dataset_df.loc[best_idx, metric] * 100
                        f.write(f"  • {metric:20s}: {best_model:30s} ({best_score:.2f}%)\n")
                    f.write("\n")
        
        f.write("\n" + "-"*80 + "\n\n")
        
        # Overall best models
        summary = compute_summary_statistics(df)
        if not summary.empty:
            f.write("OVERALL MODEL RANKINGS (by mean performance)\n")
            f.write("-"*80 + "\n\n")
            
            # Get mean scores for each metric
            for metric in summary.columns.get_level_values(0).unique()[:5]:
                f.write(f"\n{metric}:\n")
                metric_means = summary[(metric, 'mean')].sort_values(ascending=False)
                for i, (model, score) in enumerate(metric_means.head(5).items(), 1):
                    f.write(f"  {i}. {model:30s}: {score*100:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"    - Saved summary:    {output_path}")

def save_outputs(results: List[Dict], output_dir: str, output_name: str, 
                output_csv: bool = True, output_md: bool = True, 
                output_latex: bool = True, output_charts: bool = True,
                output_summary: bool = True):
    """Save results in multiple formats with comprehensive reporting."""
    if not results:
        print("[WARNING] No results to save!")
        return
    
    df = pd.DataFrame(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[INFO] Generating outputs...\n")
    
    # Add rankings
    df = add_rankings(df)
    
    # Save main outputs
    if output_csv:
        save_csv(df, output_dir / f"{output_name}.csv")
    
    if output_md:
        save_markdown(df, output_dir / f"{output_name}.md")
    
    if output_latex:
        save_latex(df, output_dir / f"{output_name}.tex")
    
    # Save summary statistics
    summary = compute_summary_statistics(df)
    if not summary.empty:
        summary.to_csv(output_dir / f"{output_name}_summary_stats.csv")
        print(f"    - Saved statistics: {output_dir / f'{output_name}_summary_stats.csv'}")
    
    if output_summary:
        generate_summary_report(df, output_dir / f"{output_name}_summary.txt")
    
    if output_charts:
        save_visualizations(df, output_dir, output_name)
    
    print("\n[SUCCESS] All outputs generated successfully!\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation results from lm-harness runs to various formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_run-results.py --input_folders ./outputs/exp1 ./outputs/exp2 \\
                                --output_dir ./results --output_name my_benchmark
  
  python extract_run-results.py --input_folders ./outputs --output_dir ./results \\
                                --output_name filtered --file_filter "Qwen"
        """
    )

    parser.add_argument("--output_dir", required=True, 
                       help="Directory to save outputs")
    parser.add_argument("--output_name", required=True, 
                       help="Base name for output files (no extension)")
    parser.add_argument("--input_folders", nargs="+", required=True, 
                       help="List of folders to search for results")
    parser.add_argument("--file_filter", default=None, 
                       help="Only process JSON files containing this string in the name")
    parser.add_argument("--no-csv", action="store_true", 
                       help="Skip CSV output")
    parser.add_argument("--no-markdown", action="store_true", 
                       help="Skip Markdown output")
    parser.add_argument("--no-latex", action="store_true", 
                       help="Skip LaTeX output")
    parser.add_argument("--no-charts", action="store_true", 
                       help="Skip chart generation")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("RESULTS EXTRACTION SCRIPT")
    print("="*80)
    print(f"\nInput folders: {', '.join(args.input_folders)}")
    print(f"Output:        {args.output_dir}/{args.output_name}.*")
    if args.file_filter:
        print(f"Filter:        {args.file_filter}")
    print("\n" + "-"*80 + "\n")

    results = collect_results(args.input_folders, args.file_filter)
    
    if not results:
        print("[ERROR] No results found! Check your input folders and file patterns.")
        return
    
    save_outputs(
        results, 
        args.output_dir, 
        args.output_name,
        output_csv=not args.no_csv,
        output_md=not args.no_markdown,
        output_latex=not args.no_latex,
        output_charts=not args.no_charts
    )
    
    print("="*80)
    print(f"✅ Results extracted successfully to {args.output_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()