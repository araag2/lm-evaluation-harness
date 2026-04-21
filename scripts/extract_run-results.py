import os
import json
import argparse
from numbers import Number
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

def is_flat_scalar_metrics_dict(results_dict: Dict[str, Any]) -> bool:
    """Return True when summary results are a single flat metrics mapping.

    Flat summaries map metric names directly to numeric scalars (e.g. rouge_l).
    Nested summaries map task/voting names to dictionaries.
    """
    if not isinstance(results_dict, dict) or not results_dict:
        return False

    for value in results_dict.values():
        if isinstance(value, dict):
            return False
        if isinstance(value, bool):
            return False
        if value is None:
            continue
        if not isinstance(value, Number):
            return False

    return True

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

    # Try to infer the evaluation mode from the folder path
    known_modes = ["multi-turn_CoT", "self-refine_CoT", "cross-consistency",
                   "multi-turn_CoT-SC", "multi-turn_CoT-MBR"]
    path_str = str(path)
    inferred_mode = "0-shot"
    for m in known_modes:
        if m in path_str:
            inferred_mode = m
            break
    
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
            "Mode": inferred_mode,
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

    mode = data.get("mode", "N/A")
    reasoning_task = data.get("reasoning_task", "")
    # Extract base dataset name (e.g. "MedNLI" from "MedNLI:CoT")
    base_dataset = reasoning_task.split(":")[0] if ":" in reasoning_task else reasoning_task

    # Strip HuggingFace org prefix (e.g. "unsloth/Qwen3-8B" → "Qwen3-8B")
    raw_model = data.get("reasoning_model", "").split(",")[0][11:]
    model_name = raw_model.split("/")[-1] if "/" in raw_model else raw_model

    results_dict = dict(data.get("results", {}))

    # Some summary files store placeholder values for these two fields.
    # Replace with useful metadata during extraction.
    if results_dict.get("name") == 0.0:
        results_dict["name"] = reasoning_task or base_dataset or "N/A"
    if results_dict.get("sample_len") == 0.0:
        samples = data.get("samples", {})
        if isinstance(samples, dict):
            results_dict["sample_len"] = len(samples)
        elif isinstance(samples, list):
            results_dict["sample_len"] = len(samples)
        else:
            results_dict["sample_len"] = 0

    if is_flat_scalar_metrics_dict(results_dict) or "acc" in results_dict.keys() or "acc,none" in results_dict.keys():
        model_args = data.get("config", {}).get("model_args", "N/A")

        result_entry = {
            "Dataset": base_dataset,
            "Mode": mode,
            "Model": model_name,
            "Model Args": model_args,
            "Path": path
        }

        result_entry.update(extract_metrics(results_dict))
        results.append(result_entry)

    elif "majority" in results_dict.keys() or "rrf" in results_dict.keys() or "condorcet" in results_dict.keys() or "logits" in results_dict.keys() or "borda" in results_dict.keys():

        for voting_method, result_dict in results_dict.items():
            model_args = data.get("config", {}).get("model_args", "N/A")

            result_entry = {
                "Dataset": base_dataset,
                "Mode": mode,
                "Voting_Method": voting_method,
                "Model": model_name,
                "Model Args": model_args,
                "Path": path
            }

            if not isinstance(result_dict, dict):
                print(f"[WARNING] Unexpected non-dict voting result in {path}: {voting_method}")
                continue
            result_entry.update(extract_metrics(result_dict))
            results.append(result_entry)

    else:
        for task_name, result_dict in results_dict.items():
            model_args = data.get("config", {}).get("model_args", "N/A")

            result_entry = {
                "Dataset": task_name,
                "Mode": mode,
                "Model": model_name,
                "Model Args": model_args,
                "Path": path
            }

            if not isinstance(result_dict, dict):
                print(f"[WARNING] Unexpected non-dict task result in {path}: {task_name}")
                continue
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
                   and not col.endswith("_stderr")
                   and not col.endswith("_rank")]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    # Group by model and compute statistics
    summary = df.groupby("Model")[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    summary = summary.round(4)
    
    return summary

def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking columns for each metric (1 = best)."""
    reserved_cols = {"Dataset", "Mode", "Model", "Model_Full", "Timestamp", "Model Args", "Path", "Voting_Method"}
    numeric_cols = [col for col in df.columns 
                   if col not in reserved_cols 
                   and pd.api.types.is_numeric_dtype(df[col])]

    # Group rankings within the same dataset+mode combination when available
    group_cols = [c for c in ["Dataset", "Mode", "Voting_Method"] if c in df.columns]
    if not group_cols:
        group_cols = ["Dataset"]
    
    for col in numeric_cols:
        # Assume higher is better for most metrics (acc, f1, etc.)
        # Lower is better for loss, perplexity
        if any(x in col.lower() for x in ['loss', 'perplexity', 'error']):
            df[f"{col}_rank"] = df.groupby(group_cols)[col].rank(ascending=True, method='min')
        else:
            df[f"{col}_rank"] = df.groupby(group_cols)[col].rank(ascending=False, method='min')
    
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


def _format_group_table(
    group_df: pd.DataFrame,
    numeric_cols: List[str],
    non_rank_numeric: List[str],
    drop_cols: List[str],
) -> pd.DataFrame:
    """Format a single group dataframe for Markdown output (in-place formatting)."""

    def to_ordinal(n: float) -> str:
        if pd.isna(n):
            return "N/A"
        n = int(n)
        suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    group_df = group_df.copy()

    # Always drop Dataset (redundant — table header carries that info) plus caller-specified cols
    group_df = group_df.drop(columns=list(set(drop_cols) | {"Dataset"}), errors="ignore")

    # Filter columns where all values are NaN
    cols_to_keep = []
    for col in group_df.columns:
        if col in numeric_cols:
            if not group_df[col].isna().all():
                cols_to_keep.append(col)
        else:
            cols_to_keep.append(col)
    group_df = group_df[cols_to_keep]

    # Drop 'alias' (lm-eval task alias) — table header already carries dataset context
    group_df = group_df.drop(columns=["alias"], errors="ignore")

    # Column ordering: Model first, then Path, then metrics
    fixed_first = [c for c in ["Model"] if c in group_df.columns]
    fixed_last  = [c for c in ["Path"] if c in group_df.columns]
    metric_order = [c for c in group_df.columns if c not in fixed_first + fixed_last]
    group_df = group_df[fixed_first + metric_order + fixed_last]

    # Highlight best per metric and format as percentage
    current_non_rank = [c for c in non_rank_numeric if c in group_df.columns]
    for metric in current_non_rank:
        higher_is_better = not any(x in metric.lower() for x in ['loss', 'perplexity', 'error'])
        subset = group_df[metric]
        if subset.isna().all() or len(subset) == 0:
            continue
        try:
            best_idx = subset.idxmax() if higher_is_better else subset.idxmin()
            if pd.isna(best_idx):
                raise ValueError
            for idx in subset.index:
                val = group_df.loc[idx, metric]
                if pd.notna(val):
                    fmt = f"{val * 100:.2f}"
                    group_df.loc[idx, metric] = f"**{fmt}**" if idx == best_idx else fmt
                else:
                    group_df.loc[idx, metric] = "N/A"
        except (ValueError, TypeError):
            for idx in subset.index:
                val = group_df.loc[idx, metric]
                group_df.loc[idx, metric] = f"{val * 100:.2f}" if pd.notna(val) else "N/A"

    # Format paths as links
    if "Path" in group_df.columns:
        group_df["Path"] = group_df["Path"].apply(
            lambda x: f"[📁]({x})" if pd.notna(x) else ""
        )

    # Format rank columns as ordinals
    for rank_col in [c for c in group_df.columns if c.endswith("_rank")]:
        group_df[rank_col] = group_df[rank_col].apply(to_ordinal)

    return group_df


def save_markdown(df: pd.DataFrame, path: Path):
    """Save results to Markdown with separate tables per (Dataset, Mode) group,
    followed by combined per-dataset tables."""
    df = df.copy()
    df = df.drop(columns=["Model Args", "Model_Full", "Timestamp"], errors="ignore")

    # ── model sort order ─────────────────────────────────────────────────────
    model_order = ["Fleming", "Panacea", "Qwen", "Llama", "deepseek", "DeepSeek", "Mistral", "Gemma"]

    def get_model_priority(model_name: str) -> int:
        for i, key in enumerate(model_order):
            if key in str(model_name):
                return i
        return len(model_order)

    # Sort entire dataframe: Dataset → Mode → Model
    sort_keys = [c for c in ["Dataset", "Mode"] if c in df.columns]
    if "Model" in df.columns:
        df = df.sort_values(
            by=sort_keys + ["Model"],
            key=lambda col: col.map(get_model_priority) if col.name == "Model" else col,
        )

    # ── numeric column sets ───────────────────────────────────────────────────
    numeric_cols = list(df.select_dtypes(include=["float", "int"]).columns)
    non_rank_numeric = [c for c in numeric_cols if not c.endswith("_rank")]

    markdown_content = ["# Evaluation Results\n\n"]

    if "Dataset" not in df.columns:
        # Fallback: single table
        formatted = _format_group_table(df, numeric_cols, non_rank_numeric,
                                        drop_cols=[])
        markdown_content.append(formatted.to_markdown(index=False, disable_numparse=True))
        markdown_content.append("\n\n")
    else:
        has_mode = "Mode" in df.columns

        # ── Section 1: one table per (Dataset, Mode) ─────────────────────────
        markdown_content.append("## Results by Mode\n\n")

        if has_mode:
            groups = df.groupby(["Dataset", "Mode"], sort=False)
            group_keys = sorted(groups.groups.keys())  # sort by (dataset, mode)
        else:
            groups = df.groupby(["Dataset"], sort=False)
            group_keys = sorted(groups.groups.keys())

        for key in group_keys:
            group_df = groups.get_group(key).copy()
            if has_mode:
                dataset, mode = key
                header = f"### {dataset} — {mode}"
                drop_cols = ["Dataset", "Mode"]
            else:
                dataset = key
                header = f"### {dataset}"
                drop_cols = ["Dataset"]

            markdown_content.append(f"{header}\n\n")
            formatted = _format_group_table(
                group_df, numeric_cols, non_rank_numeric, drop_cols=drop_cols
            )
            markdown_content.append(formatted.to_markdown(index=False, disable_numparse=True))
            markdown_content.append("\n\n")

        # ── Section 2: combined table per base Dataset (strips _0-shot suffix) ─
        markdown_content.append("---\n\n## Combined Results by Dataset\n\n")

        def _base_dataset(name: str) -> str:
            """Normalise dataset name: strip trailing _0-shot so CoT and 0-shot rows share a group."""
            return name.removesuffix("_0-shot") if isinstance(name, str) else name

        df["_BaseDataset"] = df["Dataset"].apply(_base_dataset)

        rank_cols_global = [c for c in df.columns if c.endswith("_rank")]

        for base_ds in sorted(df["_BaseDataset"].unique()):
            markdown_content.append(f"### {base_ds} (all modes)\n\n")
            combined_df = df[df["_BaseDataset"] == base_ds].copy()

            # Drop stale per-mode ranks and recompute fresh across all rows in this group
            combined_df = combined_df.drop(columns=rank_cols_global, errors="ignore")
            combined_df = combined_df.drop(columns=["_BaseDataset"], errors="ignore")
            fresh_metric_cols = [
                c for c in combined_df.columns
                if c not in {"Dataset", "Mode", "Model", "Model_Full", "Timestamp",
                             "Model Args", "Path", "Voting_Method"}
                and pd.api.types.is_numeric_dtype(combined_df[c])
            ]
            for col in fresh_metric_cols:
                ascending = any(x in col.lower() for x in ['loss', 'perplexity', 'error'])
                combined_df[f"{col}_rank"] = combined_df[col].rank(
                    ascending=ascending, method='min'
                )

            # Recompute numeric_cols / non_rank_numeric for this combined slice
            comb_numeric = list(combined_df.select_dtypes(include=["float", "int"]).columns)
            comb_non_rank = [c for c in comb_numeric if not c.endswith("_rank")]

            # Keep Mode column so rows are distinguishable
            formatted = _format_group_table(
                combined_df, comb_numeric, comb_non_rank,
                drop_cols=[]  # Dataset already dropped inside _format_group_table
            )
            # Ensure Mode is the first column after Model
            if "Mode" in formatted.columns and "Model" in formatted.columns:
                other_cols = [c for c in formatted.columns if c not in ("Model", "Mode")]
                formatted = formatted[["Model", "Mode"] + other_cols]

            markdown_content.append(formatted.to_markdown(index=False, disable_numparse=True))
            markdown_content.append("\n\n")

        # Clean up helper column
        df.drop(columns=["_BaseDataset"], errors="ignore", inplace=True)

    # Write to file
    with open(path, 'w') as f:
        f.write(''.join(markdown_content))

    print(f"    - Saved Markdown to:{path}")


def save_latex(df, path):
    # ── Drop metadata / rank columns that mustn't appear in the table ────────
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    df = df.drop(
        columns=["Model Args", "Path", "Timestamp", "Mode"] + rank_cols,
        errors="ignore",
    )

    # Multiply only true metric columns (float/int) by 100 → percentages
    numeric_cols = list(df.select_dtypes(include=["float", "int"]).columns)
    df[numeric_cols] = df[numeric_cols] * 100

    has_voting = "Voting_Method" in df.columns

    def _pivot_and_format(sub: pd.DataFrame, index_cols: list) -> str:
        """Pivot sub-df and return a LaTeX tabular string."""
        # Remove index cols from the pivot value columns
        value_cols = [c for c in sub.columns if c not in index_cols + ["Dataset"]]
        piv = sub.pivot_table(
            index=index_cols,
            columns="Dataset",
            values=value_cols,
            aggfunc="first",
        )
        # Flatten MultiIndex columns: metric_Dataset
        piv.columns = [f"{metric}_{ds}" for metric, ds in piv.columns]
        piv = piv.reset_index()
        return piv.to_latex(index=False, float_format="%.2f", escape=False, na_rep="---")

    blocks: list[str] = []

    if has_voting:
        # ── Table 1: aggregated / voting results ──────────────────────────────
        # Rows without Voting_Method are non-voting runs; split them out.
        voting_df    = df[df["Voting_Method"].notna()].copy()
        no_voting_df = df[df["Voting_Method"].isna()].drop(columns=["Voting_Method"], errors="ignore").copy()

        if not voting_df.empty:
            # Include Mode in the index so the same (Model, Voting_Method) pair
            # from different pipelines (CoT-SC vs cross-consistency) stays distinct.
            index_cols = []
            for c in ["Model", "Mode", "Voting_Method"]:
                if c in voting_df.columns:
                    index_cols.append(c)
            blocks.append("% --- Voting / aggregation results ---\n")
            blocks.append(_pivot_and_format(voting_df, index_cols))

        if not no_voting_df.empty:
            index_cols = [c for c in ["Model"] if c in no_voting_df.columns]
            blocks.append("\n% --- Single-pass results (no voting) ---\n")
            blocks.append(_pivot_and_format(no_voting_df, index_cols))

    else:
        # No Voting_Method column at all — single table indexed by Model
        index_cols = [c for c in ["Model"] if c in df.columns]
        blocks.append(_pivot_and_format(df, index_cols))

    latex_output = "\n".join(blocks)

    with open(path, "w") as f:
        f.write(latex_output)
    print(f"    - Saved LaTeX to:   {path}")


def save_visualizations(df: pd.DataFrame, output_dir: Path, output_name: str):
    """Generate charts and heatmaps.

    Charts   → output_dir/charts/      one bar chart per base dataset (all modes, sorted descending)
    Heatmaps → output_dir/heatmaps/    one heatmap per (base_dataset × mode) + one global heatmap
    """
    charts_dir   = output_dir / "charts"
    heatmaps_dir = output_dir / "heatmaps"
    charts_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    reserved_cols = {"Dataset", "Mode", "Model", "Model_Full", "Timestamp",
                     "Model Args", "Path", "Voting_Method", "_BaseDataset", "_Label", "_ModeRank"}
    metric_columns = [
        col for col in df.columns
        if col not in reserved_cols
        and pd.api.types.is_numeric_dtype(df[col])
        and not col.endswith("_rank")
    ]

    if not metric_columns:
        print("[INFO] No numeric metrics found for visualization")
        return
    if "Dataset" not in df.columns or "Model" not in df.columns:
        print("[INFO] Missing Dataset or Model columns for visualization")
        return

    MODE_ORDER = ["0-shot", "multi-turn_CoT", "multi-turn_CoT-SC",
                  "multi-turn_CoT-MBR", "cross-consistency", "self-refine_CoT"]

    def mode_rank(mode: str) -> int:
        try:
            return MODE_ORDER.index(mode)
        except ValueError:
            return len(MODE_ORDER)

    def base_dataset(name: str) -> str:
        return name.removesuffix("_0-shot") if isinstance(name, str) else name

    def safe(s: str) -> str:
        return s.replace('/', '_').replace(' ', '_').replace(':', '_')

    def _heatmap(pivot: pd.DataFrame, title: str, path: Path):
        fig_w = max(4, len(pivot.columns) * 1.6)
        fig_h = max(3, len(pivot) * 0.55)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                    cbar_kws={'label': 'Score (%)', 'shrink': 0.6},
                    vmin=0, vmax=100, linewidths=0.4,
                    annot_kws={"size": 8})
        plt.title(title, fontsize=11, fontweight='bold', pad=8)
        plt.xlabel('Metric', fontsize=9)
        plt.ylabel('')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8, rotation=0)
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches='tight')
        plt.close()

    df = df.copy()
    df["_BaseDataset"] = df["Dataset"].apply(base_dataset)
    has_mode = "Mode" in df.columns

    if has_mode:
        df["_Label"]    = df["Model"] + " — " + df["Mode"].fillna("")
        df["_ModeRank"] = df["Mode"].apply(mode_rank)
    else:
        df["_Label"]    = df["Model"]
        df["_ModeRank"] = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Charts: one figure per base dataset, bars sorted descending per metric
    # ─────────────────────────────────────────────────────────────────────────
    for base_ds in sorted(df["_BaseDataset"].unique()):
        group_df = df[df["_BaseDataset"] == base_ds].copy()
        group_df = group_df.sort_values(["_ModeRank", "Model"]).reset_index(drop=True)

        avail = [m for m in metric_columns
                 if m in group_df.columns and not group_df[m].isna().all()]
        if not avail:
            continue

        try:
            n = len(avail)
            n_bars = len(group_df)
            bar_w  = max(0.5, min(0.8, 6 / n_bars))        # narrower bars when many entries
            fig_w  = max(10, n_bars * 1.1)
            fig_h  = 4.5 * n

            fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h))
            if n == 1:
                axes = [axes]

            for ax, metric in zip(axes, avail):
                plot_df = group_df[["_Label", metric]].dropna().copy()
                plot_df[metric] = plot_df[metric] * 100
                plot_df = plot_df.sort_values(metric, ascending=False).reset_index(drop=True)

                cmap   = plt.cm.RdYlGn
                norm   = plt.Normalize(vmin=plot_df[metric].min(), vmax=plot_df[metric].max())
                colors = [cmap(norm(v)) for v in plot_df[metric]]

                bars = ax.bar(plot_df["_Label"], plot_df[metric],
                              color=colors, width=bar_w, edgecolor='white', linewidth=0.5)

                # Value labels on top of bars
                for bar, val in zip(bars, plot_df[metric]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=7.5, fontweight='bold')

                ax.set_ylabel(f'{metric} (%)', fontsize=9)
                ax.set_title(metric, fontsize=10, fontweight='bold')
                ax.set_ylim(0, 108)
                ax.set_xticks(range(len(plot_df)))
                ax.set_xticklabels(plot_df["_Label"], rotation=40, ha='right', fontsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(axis='y', alpha=0.25, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            fig.suptitle(base_ds, fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            path = charts_dir / f"{output_name}_{safe(base_ds)}.png"
            plt.savefig(path, dpi=180, bbox_inches='tight')
            plt.close()
            print(f"    - Saved chart:    {path}")
        except Exception as e:
            plt.close('all')
            print(f"    - Failed chart for '{base_ds}': {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmaps: one per (base_dataset × mode), rows = Model, cols = metrics
    # ─────────────────────────────────────────────────────────────────────────
    group_cols = ["_BaseDataset", "Mode"] if has_mode else ["_BaseDataset"]
    for key, grp in df.groupby(group_cols, sort=False):
        base_ds, mode = (key[0], key[1]) if has_mode else (key, "")
        avail = [m for m in metric_columns
                 if m in grp.columns and not grp[m].isna().all()]
        if not avail:
            continue
        try:
            pivot = grp.set_index("Model")[avail].copy()
            pivot = pivot.sort_index()
            title = f"{base_ds} — {mode}" if mode else base_ds
            path  = heatmaps_dir / f"{output_name}_heatmap_{safe(base_ds)}_{safe(mode)}.png"
            _heatmap(pivot, title, path)
            print(f"    - Saved heatmap:  {path}")
        except Exception as e:
            plt.close('all')
            print(f"    - Failed heatmap for '{base_ds} / {mode}': {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Global heatmap: rows = "Model — Mode", cols = "Dataset / metric"
    # ─────────────────────────────────────────────────────────────────────────
    try:
        global_rows = []
        for _, row in df.sort_values(["_BaseDataset", "_ModeRank", "Model"]).iterrows():
            avail = [m for m in metric_columns if pd.notna(row.get(m))]
            for m in avail:
                global_rows.append({
                    "row_label": row["_Label"],
                    "col_label": f"{row['_BaseDataset']} / {m}",
                    "value":     row[m],
                })

        if global_rows:
            gdf   = pd.DataFrame(global_rows)
            pivot = gdf.pivot_table(index="row_label", columns="col_label",
                                    values="value", aggfunc="first")
            # Preserve row order: mode rank then model
            ordered_labels = (
                df.sort_values(["_ModeRank", "Model"])["_Label"].drop_duplicates().tolist()
            )
            pivot = pivot.reindex([l for l in ordered_labels if l in pivot.index])

            path = heatmaps_dir / f"{output_name}_heatmap_GLOBAL.png"
            _heatmap(pivot, "All Datasets × All Modes", path)
            print(f"    - Saved global heatmap: {path}")
    except Exception as e:
        plt.close('all')
        print(f"    - Failed global heatmap: {e}")

    df.drop(columns=["_BaseDataset", "_Label", "_ModeRank"], errors="ignore", inplace=True)

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
        charts_dir   = output_dir / "charts"
        heatmaps_dir = output_dir / "heatmaps"
        charts_dir.mkdir(parents=True, exist_ok=True)
        heatmaps_dir.mkdir(parents=True, exist_ok=True)
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

  python extract_run-results.py --input_folders ./outputs \\
                                --output_dir ./results --output_name filtered \\
                                --file_filter "Qwen"
        """
    )

    parser.add_argument("--input_folders", nargs="+", required=True,
                       help="One or more folders to search recursively for JSON result files")
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save outputs")
    parser.add_argument("--output_name", required=True,
                       help="Base name for output files (no extension)")
    parser.add_argument("--file_filter", default=None,
                       help="Only process JSON files whose name contains this string")
    parser.add_argument("--no-csv", action="store_true",
                       help="Skip CSV output")
    parser.add_argument("--no-markdown", action="store_true",
                       help="Skip Markdown output")
    parser.add_argument("--no-latex", action="store_true",
                       help="Skip LaTeX output")
    parser.add_argument("--no-summary", action="store_true",
                       help="Skip plain-text summary report")
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
        output_summary=not args.no_summary,
        output_charts=not args.no_charts,
    )
    
    print("="*80)
    print(f"✅ Results extracted successfully to {args.output_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()