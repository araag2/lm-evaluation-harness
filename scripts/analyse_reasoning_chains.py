#!/usr/bin/env python3
"""
Reasoning Chain Quality Analyser
==================================
Measures the quality of reasoning chains produced by multi_turn_CoT /
multi_turn_CoT_SC runs and correlates those metrics with label accuracy.

Metrics computed per reasoning chain
--------------------------------------
  - Flesch Reading Ease          (textstat)
  - Flesch-Kincaid Grade Level   (textstat)
  - Gunning Fog Index            (textstat)
  - Word Count
  - Log-Likelihood Sum           (vLLM model; same infrastructure as the benchmark)
  - Perplexity                   (derived from the log-likelihood)

Outputs
-------
  <output_dir>/<name>_metrics.csv             per-sample metrics table
  <output_dir>/<name>_avg_per_label.md        avg metric per predicted label (markdown)
  <output_dir>/<name>_correlations.csv        Pearson r per metric vs is_correct
  <output_dir>/<name>_correlation_plot.png    scatter + regression grid
  <output_dir>/<name>_summary.txt             plain-text summary

Usage
-----
  python analyse_reasoning_chains.py \\
      --input_folders ./outputs/multi-turn_CoT \\
      --output_dir    ./outputs/analysis \\
      --output_name   qwen3-8b_chain_quality \\
      --model_args    pretrained=unsloth/Qwen3-8B,max_length=15000,gpu_memory_utilization=0.8,dtype=float16

  # Skip the (slow) perplexity / log-likelihood calculation:
  python analyse_reasoning_chains.py ... --no-perplexity
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import textstat

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Language-model helpers (vLLM via lm_eval infrastructure)
# ---------------------------------------------------------------------------

def load_scoring_model(provider: str, model_args: str):
    """Load a model using lm_eval's registry — same path as the benchmark."""
    from lm_eval.api.registry import get_model

    print(f"[INFO] Loading scoring model: provider={provider}  args={model_args[:80]}…")
    kwargs: Dict[str, Any] = {}
    for part in model_args.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            # Cast numeric values
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            kwargs[k.strip()] = v

    model_cls = get_model(provider)
    lm = model_cls(**kwargs)
    print("[INFO] Scoring model loaded.")
    return lm


def compute_logliks_batch(
    lm,
    chains: List[str],
) -> List[Tuple[float, float]]:
    """Score all *chains* in one batched call.

    Returns a list of (loglik_sum, perplexity) pairs, one per chain.
    Empty / whitespace-only chains return (nan, nan).
    """
    from lm_eval.api.instance import Instance

    results: List[Tuple[float, float]] = []
    valid_indices: List[int] = []
    requests: List[Instance] = []

    # Build lm_eval Instance objects for non-empty chains
    for i, chain in enumerate(chains):
        if chain and chain.strip():
            inst = Instance(
                request_type="loglikelihood",
                doc={},
                arguments=("", chain),   # (context="", continuation=chain)
                idx=len(requests),
            )
            requests.append(inst)
            valid_indices.append(i)
        else:
            results.append((float("nan"), float("nan")))

    if not requests:
        return results

    print(f"[INFO] Scoring {len(requests)} chains via loglikelihood …")
    raw = lm.loglikelihood(requests)   # list of (ll_sum, is_greedy)

    # Build a full results list aligned with the original *chains* order
    scored: Dict[int, Tuple[float, float]] = {}
    for req_i, (ll_sum, _) in enumerate(raw):
        original_i = valid_indices[req_i]
        chain      = chains[original_i]
        n_words    = max(len(chain.split()), 1)
        # Perplexity as exp(-ll_sum / n_words): word-level proxy
        # (token-level would need the tokenizer; this is consistent across models)
        ppl = math.exp(-ll_sum / n_words) if not math.isnan(ll_sum) else float("nan")
        scored[original_i] = (ll_sum, ppl)

    # Merge back — preserve the (nan,nan) entries already in results
    combined: List[Tuple[float, float]] = []
    nan_iter = iter(results)          # the (nan,nan) entries for empty chains
    valid_set = set(valid_indices)
    for i in range(len(chains)):
        if i in valid_set:
            combined.append(scored[i])
        else:
            combined.append(next(nan_iter))

    return combined


# ---------------------------------------------------------------------------
# Readability metrics
# ---------------------------------------------------------------------------

def compute_readability(text: str) -> Dict[str, float]:
    """Return a dict of textstat readability scores for *text*."""
    if not text or not text.strip():
        return {
            "flesch_reading_ease":   float("nan"),
            "flesch_kincaid_grade":  float("nan"),
            "gunning_fog":           float("nan"),
            "word_count":            0.0,
        }
    return {
        "flesch_reading_ease":   textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade":  textstat.flesch_kincaid_grade(text),
        "gunning_fog":           textstat.gunning_fog(text),
        "word_count":            float(len(text.split())),
    }


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def _get_model_shortname(model_str: str) -> str:
    """Extract a readable model name from a model_args string."""
    # e.g. "pretrained=Qwen/Qwen3-8B,max_length=..." → "Qwen3-8B"
    for part in model_str.split(","):
        if part.startswith("pretrained="):
            path = part.split("=", 1)[1]
            return path.split("/")[-1]
    return model_str[:40]


def load_samples_from_file(path: str) -> List[Dict[str, Any]]:
    """Parse one summary JSON and return a list of per-sample dicts (no LM scoring yet)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load {path}: {e}")
        return []

    samples = data.get("samples", {})
    if not samples:
        return []

    mode            = data.get("mode", "unknown")
    reasoning_model = _get_model_shortname(data.get("reasoning_model", "unknown"))
    reasoning_task  = data.get("reasoning_task", "unknown")

    rows: List[Dict[str, Any]] = []
    iter_samples = samples.values() if isinstance(samples, dict) else samples

    for sample in iter_samples:
        doc        = sample.get("doc", {})
        chain      = doc.get("Reasoning_Chain", "") or ""
        gt_label   = str(doc.get("Label", ""))
        pred_label = str(sample.get("pred_label", ""))
        is_correct = int(gt_label == pred_label) if (gt_label and pred_label) else None

        if not chain or not chain.strip():
            continue

        row: Dict[str, Any] = {
            "file":        path,
            "mode":        mode,
            "model":       reasoning_model,
            "task":        reasoning_task,
            "gt_label":    gt_label,
            "pred_label":  pred_label,
            "is_correct":  is_correct,
            "chain_text":  chain,
            # LM scores filled later
            "loglik_sum":  float("nan"),
            "perplexity":  float("nan"),
        }
        row.update(compute_readability(chain))
        rows.append(row)

    return rows


def collect_all_samples(
    folders: List[str],
    file_filter: Optional[str],
    scoring_lm=None,
) -> List[Dict[str, Any]]:
    """Walk *folders* recursively, collect readability metrics, then batch-score
    all chains through *scoring_lm* (if provided) in a single model pass.
    """
    all_rows: List[Dict[str, Any]] = []
    n_files = 0

    for folder in folders:
        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        for root, _, files in os.walk(folder):
            for fname in sorted(files):
                if not fname.endswith(".json"):
                    continue
                if file_filter and file_filter not in fname:
                    continue
                full = os.path.join(root, fname)
                rows = load_samples_from_file(full)
                if rows:
                    all_rows.extend(rows)
                    n_files += 1
                    print(f"  [OK] {full}  ({len(rows)} samples)")

    print(f"\n[INFO] Loaded {len(all_rows)} samples from {n_files} files.")

    # --- single batched LM pass -------------------------------------------
    if scoring_lm is not None and all_rows:
        chains = [r["chain_text"] for r in all_rows]
        scored = compute_logliks_batch(scoring_lm, chains)
        for row, (ll, ppl) in zip(all_rows, scored):
            row["loglik_sum"] = ll
            row["perplexity"] = ppl
        print("[INFO] Log-likelihood scoring complete.")

    print()
    return all_rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "word_count",
    "loglik_sum",
    "perplexity",
]

METRIC_LABELS = {
    "flesch_reading_ease":  "Flesch Reading Ease ↑",
    "flesch_kincaid_grade": "Flesch–Kincaid Grade ↓",
    "gunning_fog":          "Gunning Fog Index ↓",
    "word_count":           "Word Count",
    "loglik_sum":           "Log-Likelihood Sum ↑",
    "perplexity":           "Perplexity ↓",
}


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r and p-value for each metric vs is_correct, grouped by task+model."""
    rows = []
    sub = df.dropna(subset=["is_correct"])

    for (task, model), grp in sub.groupby(["task", "model"]):
        for metric in METRIC_COLS:
            col = grp[metric].dropna()
            correct = grp.loc[col.index, "is_correct"]
            if len(col) < 5 or correct.std() == 0:
                r, p = float("nan"), float("nan")
            else:
                r, p = stats.pearsonr(col, correct)
            rows.append({
                "task":   task,
                "model":  model,
                "metric": metric,
                "pearson_r": round(r, 4),
                "p_value":   round(p, 4) if not math.isnan(p) else float("nan"),
                "n":         len(col),
            })

    return pd.DataFrame(rows)


def avg_per_label(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric grouped by (task, model, pred_label)."""
    available = [m for m in METRIC_COLS if df[m].notna().any()]
    return (
        df.groupby(["task", "model", "pred_label"])[available]
        .mean()
        .round(3)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Output: Markdown table
# ---------------------------------------------------------------------------

def save_avg_per_label_md(df_avg: pd.DataFrame, path: Path):
    """Write average-per-label table as a Markdown file."""
    available = [m for m in METRIC_COLS if m in df_avg.columns and df_avg[m].notna().any()]

    lines = ["# Reasoning Chain Quality — Average Metrics per Predicted Label\n\n"]

    for (task, model), grp in df_avg.groupby(["task", "model"]):
        lines.append(f"## {task}  ·  `{model}`\n\n")

        # Header
        header = "| Predicted Label | " + " | ".join(METRIC_LABELS[m] for m in available) + " |"
        sep    = "|:---| " + " | ".join(":---:" for _ in available) + " |"
        lines.append(header)
        lines.append(sep)

        for _, row in grp.iterrows():
            cells = [str(row["pred_label"])]
            for m in available:
                v = row[m]
                cells.append(f"{v:.3f}" if not (isinstance(v, float) and math.isnan(v)) else "—")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"    - Saved avg-per-label:  {path}")


# ---------------------------------------------------------------------------
# Output: Correlation plot
# ---------------------------------------------------------------------------

def save_correlation_plot(df: pd.DataFrame, path: Path):
    """Grid of scatter plots (metric vs is_correct), one subplot per metric×task."""
    sub = df.dropna(subset=["is_correct"]).copy()
    available = [m for m in METRIC_COLS if sub[m].notna().any()]

    tasks  = sorted(sub["task"].unique())
    n_rows = len(tasks)
    n_cols = len(available)

    if n_rows == 0 or n_cols == 0:
        print("    - [SKIP] No data for correlation plot.")
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for row_i, task in enumerate(tasks):
        task_df = sub[sub["task"] == task]

        for col_i, metric in enumerate(available):
            ax = axes[row_i][col_i]
            m_vals = task_df[metric].dropna()
            c_vals = task_df.loc[m_vals.index, "is_correct"]

            # Jitter correct/incorrect on y-axis for readability
            jitter = np.random.default_rng(0).uniform(-0.04, 0.04, size=len(c_vals))
            ax.scatter(m_vals, c_vals + jitter, alpha=0.25, s=8, color="steelblue")

            # Regression line
            if len(m_vals) >= 5 and c_vals.std() > 0:
                slope, intercept, r, p, _ = stats.linregress(m_vals, c_vals)
                x_lin = np.linspace(m_vals.min(), m_vals.max(), 100)
                ax.plot(x_lin, slope * x_lin + intercept, color="firebrick", lw=1.5,
                        label=f"r={r:.2f} p={p:.3f}")
                ax.legend(fontsize=7, loc="upper right")

            ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=8)
            ax.set_ylabel("Correct (1/0)", fontsize=8)
            ax.set_title(f"{task[:30]}", fontsize=8, style="italic")
            ax.tick_params(labelsize=7)

    fig.suptitle("Reasoning Chain Metrics vs Accuracy", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    - Saved correlation plot: {path}")


# ---------------------------------------------------------------------------
# Output: Summary text
# ---------------------------------------------------------------------------

def save_summary(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    path: Path,
):
    available = [m for m in METRIC_COLS if df[m].notna().any()]

    with open(path, "w") as f:
        sep = "=" * 80
        f.write(sep + "\n")
        f.write("REASONING CHAIN QUALITY — SUMMARY\n")
        f.write(sep + "\n\n")

        f.write(f"Total samples analysed : {len(df)}\n")
        f.write(f"Tasks                  : {', '.join(sorted(df['task'].unique()))}\n")
        f.write(f"Models                 : {', '.join(sorted(df['model'].unique()))}\n")
        if "is_correct" in df and df["is_correct"].notna().any():
            acc = df["is_correct"].dropna().mean()
            f.write(f"Overall accuracy       : {acc:.3f}\n")
        f.write("\n" + "-" * 80 + "\n\n")

        # Global mean metrics
        f.write("Global metric averages (all samples)\n\n")
        for m in available:
            col = df[m].dropna()
            if len(col):
                f.write(f"  {METRIC_LABELS.get(m, m):<35}  mean={col.mean():.3f}  "
                        f"std={col.std():.3f}  min={col.min():.3f}  max={col.max():.3f}\n")

        f.write("\n" + "-" * 80 + "\n\n")

        # Strongest correlations
        f.write("Correlations with accuracy (Pearson r)\n\n")
        best = corr_df.dropna(subset=["pearson_r"]).copy()
        best = best.sort_values("pearson_r", key=abs, ascending=False)
        for _, row in best.head(20).iterrows():
            sig = "**" if (not math.isnan(row["p_value"]) and row["p_value"] < 0.05) else "  "
            f.write(f"  {sig} {row['task'][:25]:<26} {row['model'][:20]:<21} "
                    f"{METRIC_LABELS.get(row['metric'], row['metric']):<35} "
                    f"r={row['pearson_r']:+.3f}  p={row['p_value']:.3f}  n={int(row['n'])}\n")

        f.write("\n" + sep + "\n")

    print(f"    - Saved summary:          {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse reasoning chain quality from multi_turn_CoT outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folders", nargs="+", required=True,
                        help="Top-level folders to search recursively for JSON result files.")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write outputs into.")
    parser.add_argument("--output_name", required=True,
                        help="Base filename prefix for all outputs (no extension).")
    parser.add_argument("--file_filter", default=None,
                        help="Only process JSON files whose name contains this string.")
    parser.add_argument("--provider", default="vllm",
                        help="lm_eval model provider (default: vllm).")
    parser.add_argument("--model_args", default=None,
                        help="Model args string passed to lm_eval, e.g. "
                             "'pretrained=unsloth/Qwen3-8B,max_length=15000,"
                             "gpu_memory_utilization=0.8,dtype=float16'. "
                             "If omitted, log-likelihood / perplexity are skipped.")
    parser.add_argument("--no-perplexity", action="store_true",
                        help="Skip log-likelihood / perplexity scoring even if "
                             "--model_args is provided.")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("REASONING CHAIN QUALITY ANALYSER")
    print("=" * 80)
    print(f"Input folders : {', '.join(args.input_folders)}")
    print(f"Output        : {args.output_dir}/{args.output_name}.*")
    use_lm = bool(args.model_args) and not args.no_perplexity
    if use_lm:
        print(f"Scoring model : provider={args.provider}  args={args.model_args[:80]}…")
    else:
        print("LM scoring    : OFF (pass --model_args to enable)")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 1. Load scoring model (once, before collecting)
    # ------------------------------------------------------------------
    scoring_lm = None
    if use_lm:
        scoring_lm = load_scoring_model(args.provider, args.model_args)

    # ------------------------------------------------------------------
    # 2. Collect + batch-score
    # ------------------------------------------------------------------
    rows = collect_all_samples(
        args.input_folders,
        args.file_filter,
        scoring_lm=scoring_lm,
    )

    if not rows:
        print("[ERROR] No samples found. Check your input folders.")
        return

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Analyse
    # ------------------------------------------------------------------
    corr_df  = compute_correlations(df)
    df_avg   = avg_per_label(df)

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n   = args.output_name

    print("[INFO] Saving outputs …\n")

    # CSV — full per-sample metrics (drop raw chain text to keep size manageable)
    csv_df = df.drop(columns=["chain_text"], errors="ignore")
    csv_path = out / f"{n}_metrics.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"    - Saved metrics CSV:      {csv_path}")

    # Correlations CSV
    corr_path = out / f"{n}_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"    - Saved correlations CSV: {corr_path}")

    # Markdown table
    save_avg_per_label_md(df_avg, out / f"{n}_avg_per_label.md")

    # Correlation plot
    save_correlation_plot(df, out / f"{n}_correlation_plot.png")

    # Summary
    save_summary(df, corr_df, out / f"{n}_summary.txt")

    print("\n" + "=" * 80)
    print(f"✅  Done. Results in {out}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
