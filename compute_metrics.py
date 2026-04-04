"""
Compute Table 1 / Table 2 / Figure 2 metrics from jsonl output files.

Metrics (paper definitions):
  r  = rejection rate = (sum(accept_lengths) - len(accept_lengths)) / total_draft_tokens
         equivalently: (accepted_tokens) / (draft_tokens)
         where accepted_tokens = sum of net-accepted per step (accept_lengths - 1 per step)

  τ  = speedup = mean_tokens_per_sec(method) / mean_tokens_per_sec(baseline AR)

Usage:
    # Compute τ for one pair of jsonl files:
    python compute_metrics.py \\
        --method-file test/webquestions/.../via-sd-3tier.jsonl \\
        --base-file   test/webquestions/.../gemma2-9b-ar.jsonl

    # Summarize all results for Table 1:
    python compute_metrics.py --table1 --task-dir test/table1

    # Summarize all results for Table 2:
    python compute_metrics.py --table2 --result-dir test/
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_jsonl(path: str) -> list:
    """Load a jsonl file, skipping the trailing summary line (no 'choices' key)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "choices" in obj:
                records.append(obj)
    return records


def compute_speed_metrics(method_path: str, base_path: str, data_num: int = None):
    """
    Compute speedup τ and rejection rate r from two jsonl files.

    Returns: dict with keys τ, r, mean_accept_len, n_samples
    """
    method_data = load_jsonl(method_path)
    base_data   = load_jsonl(base_path)

    if data_num is not None:
        method_data = method_data[:data_num]
        base_data   = base_data[:data_num]

    n = min(len(method_data), len(base_data))
    if n == 0:
        raise ValueError(f"Empty jsonl files: {method_path}, {base_path}")

    # Speedup τ = mean(tokens/sec method) / mean(tokens/sec baseline)
    method_speeds = []
    base_speeds   = []
    accept_lengths_all = []
    draft_tokens_all = 0

    for i in range(n):
        mc = method_data[i]["choices"][0]
        bc = base_data[i]["choices"][0]

        m_toks  = sum(mc["new_tokens"])
        m_time  = sum(mc["wall_time"])
        b_toks  = sum(bc["new_tokens"])
        b_time  = sum(bc["wall_time"])

        if m_time > 0:
            method_speeds.append(m_toks / m_time)
        if b_time > 0:
            base_speeds.append(b_toks / b_time)

        accept_lengths_all.extend(mc.get("accept_lengths", []))

    # Acceptance rate r from stored value (or recompute from accept_lengths)
    r_values = [
        d["choices"][0].get("acceptance_rate", None)
        for d in method_data[:n]
        if d["choices"][0].get("acceptance_rate") is not None
    ]
    if r_values:
        r_mean = float(np.mean(r_values))
    elif accept_lengths_all:
        # Fallback: approximate from mean accept length
        # r ≈ (mean_accept_length - 1) / gamma  (paper Eq.)
        r_mean = float(np.mean(accept_lengths_all) - 1)
    else:
        r_mean = float("nan")

    tau = float(np.mean(method_speeds) / np.mean(base_speeds)) if base_speeds else float("nan")

    return {
        "tau":             tau,
        "r":               r_mean,
        "mean_accept_len": float(np.mean(accept_lengths_all)) if accept_lengths_all else float("nan"),
        "method_tok_per_sec": float(np.mean(method_speeds)),
        "base_tok_per_sec":   float(np.mean(base_speeds)),
        "n_samples":       n,
    }


def print_table(rows: list, headers: list, title: str = ""):
    """Simple ASCII table printer."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def summarize_table1(task_dir: str, data_num: int = 200):
    """Print Table 1: tier comparison across tasks."""
    tasks = ["webquestions", "nq", "triviaqa"]
    tiers = [2, 3, 4, 5]
    pair  = "2b-9b"   # adjust if running other pairs

    rows = []
    for task in tasks:
        for tier in tiers:
            method_glob = os.path.join(
                task_dir, task, f"*tier{tier}*", "model_answer", "*", f"*tier{tier}*.jsonl"
            )
            base_glob = os.path.join(
                task_dir, task, "*", "model_answer", f"*{pair}*ar*", "*.jsonl"
            )
            method_files = glob.glob(method_glob, recursive=True)
            base_files   = glob.glob(base_glob,   recursive=True)

            if not method_files or not base_files:
                rows.append([task, f"{tier}-tier", "N/A", "N/A", "N/A"])
                continue

            try:
                m = compute_speed_metrics(method_files[0], base_files[0], data_num)
                rows.append([
                    task, f"{tier}-tier",
                    f"{m['r']:.3f}",
                    f"{m['tau']:.2f}×",
                    f"{m['n_samples']}",
                ])
            except Exception as e:
                rows.append([task, f"{tier}-tier", f"ERR: {e}", "", ""])

    print_table(rows,
                headers=["Task", "# Tiers", "r (reject)", "τ (speedup)", "n"],
                title="Table 1: Multi-Tier Speculative Decoding")


def summarize_table2(result_dir: str, data_num: int = 200):
    """Print Table 2: full method comparison."""
    tasks   = ["webquestions", "nq", "triviaqa"]
    methods = {
        "Vanilla AR":        "ar",
        "Spec. Dec. (SD)":   "sd-2b",
        "VIA-SD (3-tier)":   "via-sd-2b",
    }
    pairs = ["9b", "27b"]

    for pair in pairs:
        rows = []
        for task in tasks:
            for method_name, key in methods.items():
                pattern = os.path.join(
                    result_dir, "**", f"*{key}*{pair}*.jsonl"
                )
                files = glob.glob(pattern, recursive=True)
                if not files:
                    rows.append([task, method_name, "N/A", "N/A"])
                    continue
                method_file = files[0]

                base_pattern = os.path.join(result_dir, "**", f"*{pair}*ar*.jsonl")
                base_files   = glob.glob(base_pattern, recursive=True)
                if not base_files:
                    rows.append([task, method_name, "N/A", "N/A"])
                    continue

                try:
                    m = compute_speed_metrics(method_file, base_files[0], data_num)
                    rows.append([task, method_name,
                                 f"{m['r']:.3f}", f"{m['tau']:.2f}×"])
                except Exception as e:
                    rows.append([task, method_name, f"ERR", str(e)[:30]])

        print_table(rows,
                    headers=["Task", "Method", "r", "τ"],
                    title=f"Table 2: Full Comparison (2B→{pair.upper()})")


def summarize_figure2(result_dir: str, data_num: int = 200):
    """Print Figure 2: skip-layer vs independent intermediate."""
    tasks = ["xsum", "wmt14", "cnndm"]
    rows = []
    for task in tasks:
        for label, key in [("Skip-layer q' (45%)", "skip45"), ("Independent 12B", "12b")]:
            method_files = glob.glob(
                os.path.join(result_dir, "**", f"*{key}*{task}*.jsonl"), recursive=True
            )
            base_files = glob.glob(
                os.path.join(result_dir, "**", f"*ar*{task}*.jsonl"), recursive=True
            )
            if not method_files or not base_files:
                rows.append([task, label, "N/A"])
                continue
            try:
                m = compute_speed_metrics(method_files[0], base_files[0], data_num)
                rows.append([task, label, f"{m['tau']:.2f}×"])
            except Exception as e:
                rows.append([task, label, f"ERR: {e}"])

    print_table(rows,
                headers=["Task", "Intermediate", "τ (speedup)"],
                title="Figure 2: Skip-Layer q' vs Independent Intermediate Model")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compute VIA-SD paper metrics")
    p.add_argument("--method-file", type=str, default=None,
                   help="Path to method jsonl (for single-pair speedup)")
    p.add_argument("--base-file",   type=str, default=None,
                   help="Path to baseline AR jsonl")
    p.add_argument("--data-num",    type=int, default=200)
    p.add_argument("--table1",      action="store_true",
                   help="Summarize Table 1 results")
    p.add_argument("--table2",      action="store_true",
                   help="Summarize Table 2 results")
    p.add_argument("--figure2",     action="store_true",
                   help="Summarize Figure 2 results")
    p.add_argument("--task-dir",    type=str, default="test/table1")
    p.add_argument("--result-dir",  type=str, default="test")
    args = p.parse_args()

    if args.method_file and args.base_file:
        m = compute_speed_metrics(args.method_file, args.base_file, args.data_num)
        print("\n── Single-pair metrics ──────────────────────────────────")
        print(f"  τ (speedup):           {m['tau']:.4f}×")
        print(f"  r (acceptance rate):   {m['r']:.4f}")
        print(f"  mean accept length:    {m['mean_accept_len']:.2f}")
        print(f"  method tokens/sec:     {m['method_tok_per_sec']:.2f}")
        print(f"  baseline tokens/sec:   {m['base_tok_per_sec']:.2f}")
        print(f"  n samples:             {m['n_samples']}")
        return

    if args.table1:
        summarize_table1(args.task_dir, args.data_num)
    if args.table2:
        summarize_table2(args.result_dir, args.data_num)
    if args.figure2:
        summarize_figure2(args.result_dir, args.data_num)

    if not (args.table1 or args.table2 or args.figure2 or args.method_file):
        p.print_help()


if __name__ == "__main__":
    main()
