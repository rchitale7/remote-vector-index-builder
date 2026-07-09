#!/usr/bin/env python3
"""Aggregate factor-sweep results into p50 / p90 / range for download & upload.

Reads results/<label>/run_*.csv produced by run_factor_sweep.sh (each is the
per-dataset output of remote_build_csv.py) and, for every (label, dataset),
computes the median (p50), 90th percentile (p90), and range (max - min) of
`vector_download_time` and `upload_time` across the repetitions.

Timing cells look like "3.35 seconds"; the trailing unit is stripped before
parsing. Rows whose status is not COMPLETED are dropped from the stats.

Usage:
    python aggregate_factors.py --results-dir results [--output results/factor_summary.csv]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Order labels sensibly regardless of filesystem ordering.
LABEL_ORDER = ["baseline", "f1", "f1_5", "f2", "f2_5", "f3"]

SECONDS_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_seconds(val):
    """Turn a cell like '3.35 seconds' (or 3.35, or NaN) into a float or NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    m = SECONDS_RE.search(str(val))
    return float(m.group()) if m else np.nan


def label_sort_key(label):
    return (LABEL_ORDER.index(label) if label in LABEL_ORDER else len(LABEL_ORDER), label)


def load_runs(results_dir):
    """Load every run_*.csv under results_dir/<label>/, tagged with its label."""
    frames = []
    for label_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        for csv_path in sorted(glob.glob(os.path.join(label_dir, "run_*.csv"))):
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                print(f"WARN: empty CSV skipped: {csv_path}", file=sys.stderr)
                continue
            if df.empty:
                continue
            df["label"] = label
            df["source_csv"] = os.path.basename(csv_path)
            frames.append(df)
    if not frames:
        raise SystemExit(f"No run_*.csv found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    df = df.copy()
    df["download_s"] = df["vector_download_time"].map(parse_seconds)
    df["upload_s"] = df["upload_time"].map(parse_seconds)

    # Keep only successful builds when a status column is present.
    if "status" in df.columns:
        completed = df["status"].astype(str).str.contains("COMPLETED", na=False)
        dropped = (~completed).sum()
        if dropped:
            print(f"NOTE: dropping {dropped} non-COMPLETED row(s) from stats", file=sys.stderr)
        df = df[completed]

    rows = []
    for (label, dataset), grp in df.groupby(["label", "dataset"]):
        for metric, col in (("download", "download_s"), ("upload", "upload_s")):
            vals = grp[col].dropna().to_numpy()
            if vals.size == 0:
                continue
            rows.append({
                "label": label,
                "dataset": dataset,
                "metric": metric,
                "n": int(vals.size),
                "p50": round(float(np.percentile(vals, 50)), 3),
                "p90": round(float(np.percentile(vals, 90)), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "range": round(float(vals.max() - vals.min()), 3),
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No parseable download/upload timings found.")
    summary["_lk"] = summary["label"].map(label_sort_key)
    summary = summary.sort_values(
        ["metric", "dataset", "_lk"]
    ).drop(columns="_lk").reset_index(drop=True)
    return summary


def print_tables(summary):
    for metric in ["download", "upload"]:
        sub = summary[summary["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n=== {metric.upper()} TIME (seconds) — p50 / p90 / range across reps ===")
        for dataset, grp in sub.groupby("dataset"):
            print(f"\n  dataset: {dataset}")
            print(f"    {'label':<10} {'n':>3} {'p50':>8} {'p90':>8} {'min':>8} {'max':>8} {'range':>8}")
            for _, r in grp.iterrows():
                print(f"    {r['label']:<10} {r['n']:>3} {r['p50']:>8} {r['p90']:>8} "
                      f"{r['min']:>8} {r['max']:>8} {r['range']:>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results",
                    help="Directory containing <label>/run_*.csv (default: results)")
    ap.add_argument("--output", default=None,
                    help="Summary CSV path (default: <results-dir>/factor_summary.csv)")
    args = ap.parse_args()

    output = args.output or os.path.join(args.results_dir, "factor_summary.csv")

    df = load_runs(args.results_dir)
    summary = summarize(df)
    summary.to_csv(output, index=False)
    print_tables(summary)
    print(f"\nSummary written to {output}")


if __name__ == "__main__":
    main()
