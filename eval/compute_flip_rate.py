"""
Compute the ensemble flip rate from one or more eval CSVs produced by
run_autoffs.py.

A "flip" means the held-out ensemble's post-transformation probability
(`avg_prob_after`) crossed 0.5 toward the target class:
    target_class == 1 → flip if prob > 0.5  (feminization, M → F)
    target_class == 0 → flip if prob < 0.5  (masculinization, F → M)

Rates are reported overall and separately for both directions. When
multiple CSVs are given, a side-by-side comparison table is printed so
ablations like single-classifier vs. ensemble optimization can be read
at a glance.

Usage:
    python compute_flip_rate.py \\
        --csv ./deformed_images/ensemble6/eval_ensemble6.csv \\
              ./deformed_images/single_resnet34/eval_single_resnet34.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


ENSEMBLE_COLUMN = "avg_prob_after"


def flip_rate(probs, targets):
    flipped = ((targets == 1) & (probs > 0.5)) | ((targets == 0) & (probs < 0.5))
    n = len(targets)
    return {"n": int(n), "n_flipped": int(flipped.sum()), "flip_rate": float(flipped.mean()) if n else float("nan")}


def analyze_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "target_class" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'target_class' column.")
    if ENSEMBLE_COLUMN not in df.columns:
        raise ValueError(f"{csv_path}: missing '{ENSEMBLE_COLUMN}' column.")
    targets = df["target_class"].astype(int)
    probs = df[ENSEMBLE_COLUMN]

    fem_mask = targets == 1
    masc_mask = targets == 0

    return {
        "csv_path": str(csv_path),
        "n_samples": int(len(df)),
        "ensemble": {
            "overall": flip_rate(probs, targets),
            "feminization": flip_rate(probs[fem_mask], targets[fem_mask]),
            "masculinization": flip_rate(probs[masc_mask], targets[masc_mask]),
        },
    }


def fmt_rate(stats):
    return f"{stats['flip_rate']*100:5.1f}%"


def print_single(summary):
    print(f"\n=== {summary['csv_path']} ===")
    print(f"  n = {summary['n_samples']}")
    e = summary["ensemble"]
    print(f"  overall          : {fmt_rate(e['overall'])}  (n={e['overall']['n']})")
    print(f"  feminization  M→F: {fmt_rate(e['feminization'])}  (n={e['feminization']['n']})")
    print(f"  masculinization F→M: {fmt_rate(e['masculinization'])}  (n={e['masculinization']['n']})")


def print_comparison(summaries):
    exp_labels = [Path(s["csv_path"]).stem.replace("eval_", "") for s in summaries]
    rows = [
        ("overall", "overall"),
        ("feminization (M→F)", "feminization"),
        ("masculinization (F→M)", "masculinization"),
    ]
    print("\n=== Comparison: ensemble flip rate ===")
    header = f"  {'direction':<24}  " + "  ".join(f"{label:<20}" for label in exp_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key in rows:
        cells = [fmt_rate(s["ensemble"][key]) for s in summaries]
        print(f"  {label:<24}  " + "  ".join(f"{c:<20}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="Compute held-out ensemble flip rate from run_autoffs.py CSVs")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="One or more eval CSV paths produced by run_autoffs.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional explicit output JSON path. Default: alongside each CSV as flip_rates_{exp}.json")
    args = parser.parse_args()

    summaries = [analyze_csv(p) for p in args.csv]

    for summary in summaries:
        print_single(summary)
        csv_path = Path(summary["csv_path"])
        if args.output and len(args.csv) == 1:
            out_json = Path(args.output)
        else:
            stem = csv_path.stem.replace("eval_", "")
            out_json = csv_path.parent / f"flip_rates_{stem}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → saved to {out_json}")

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()
