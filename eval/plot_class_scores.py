"""
Plot held-out ensemble class probabilities after transformation, one boxplot
and one paired histogram per eval CSV produced by run_autoffs.py.

For each CSV the script writes:
    {csv_stem}_box.pdf   — boxplot grouped by transformation direction
    {csv_stem}_hist.pdf  — side-by-side histograms for M→F and F→M

Usage:
    python plot_class_scores.py \\
        --csv ./deformed_images/ensemble6/eval_ensemble6.csv \\
              ./deformed_images/single_resnet34/eval_single_resnet34.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PROB_COLUMN = "avg_prob_after"  # held-out ensemble average
LABEL_MAP = {0: "male (0)", 1: "female (1)"}
DIR_M2F = "male (0) → female (1)"
DIR_F2M = "female (1) → male (0)"
COLOR_M2F = (0.647, 0.855, 0.965)
COLOR_F2M = (0.745, 0.894, 0.659)


def _apply_rc():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def _prepare(df):
    df = df.copy()
    df["orig_label_name"] = df["orig_label"].map(LABEL_MAP)
    df["target_class_name"] = df["target_class"].map(LABEL_MAP)
    df["direction"] = df["orig_label_name"] + " → " + df["target_class_name"]
    return df


def _plot_box(df, out_path):
    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x="direction",
        y=PROB_COLUMN,
        hue="orig_label_name",
        palette="pastel",
        order=[DIR_M2F, DIR_F2M],
    )
    plt.title(r"Average Ensemble Class Probability After Transformation")
    plt.xlabel(r"Transformation Direction")
    plt.ylabel(
        "Avg. Class Prob. After Transform\n"
        r"$(0 = \mathrm{male}, 1 = \mathrm{female})$"
    )
    plt.legend(title="Original Label", loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def _hist_panel(ax, values, color, legend_loc):
    bins = np.linspace(0, 1, 31)
    counts, _ = np.histogram(values, bins=bins)
    ylim = max(15, int(counts.max() * 1.2)) if len(counts) else 15
    text_y = 0.93 * ylim

    in_male = int((values < 0.5).sum())
    in_female = int((values >= 0.5).sum())

    ax.text(
        0.25, text_y, f"Male\nn={in_male}",
        ha="center", va="top", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3", alpha=0.9),
    )
    ax.text(
        0.75, text_y, f"Female\nn={in_female}",
        ha="center", va="top", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3", alpha=0.9),
    )
    ax.axvspan(0, 0.5, alpha=0.5, color="lightgray")
    ax.axvspan(0.5, 1, alpha=0.5, color="darkgray")
    ax.grid(axis="y", alpha=0.5)
    ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=1.0)
    if len(values):
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=2,
                   label=f"Mean: {values.mean():.3f}")
        ax.axvline(values.median(), color="red", linestyle=":", linewidth=2,
                   label=f"Median: {values.median():.3f}")
    ax.set_xlabel("Class Probability After Transformation")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, ylim)
    ax.set_xlim(0, 1)
    ax.legend(loc=legend_loc)

    return in_male, in_female


def _plot_hist(df, out_path):
    m2f = df.loc[df["direction"] == DIR_M2F, PROB_COLUMN]
    f2m = df.loc[df["direction"] == DIR_F2M, PROB_COLUMN]

    fig, axes = plt.subplots(1, 2, figsize=(12, 2))
    m2f_male, m2f_female = _hist_panel(axes[0], m2f, COLOR_M2F, legend_loc=3)
    f2m_male, f2m_female = _hist_panel(axes[1], f2m, COLOR_F2M, legend_loc=4)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    return {
        "m2f": {"n": len(m2f), "in_male": m2f_male, "in_female": m2f_female},
        "f2m": {"n": len(f2m), "in_male": f2m_male, "in_female": f2m_female},
    }


def _print_stats(csv_path, stats):
    print(f"\n=== Class Region Statistics ({csv_path}) ===")
    m2f, f2m = stats["m2f"], stats["f2m"]
    n_m2f = max(m2f["n"], 1)
    n_f2m = max(f2m["n"], 1)
    print(f"\nMale (0) → Female (1):")
    print(f"  In Male region (< 0.5):   {m2f['in_male']} ({m2f['in_male']/n_m2f*100:.1f}%)")
    print(f"  In Female region (≥ 0.5): {m2f['in_female']} ({m2f['in_female']/n_m2f*100:.1f}%)")
    print(f"\nFemale (1) → Male (0):")
    print(f"  In Male region (< 0.5):   {f2m['in_male']} ({f2m['in_male']/n_f2m*100:.1f}%)")
    print(f"  In Female region (≥ 0.5): {f2m['in_female']} ({f2m['in_female']/n_f2m*100:.1f}%)")


def plot_one(csv_path):
    csv_path = Path(csv_path)
    df = _prepare(pd.read_csv(csv_path))
    if PROB_COLUMN not in df.columns:
        raise ValueError(f"{csv_path}: missing '{PROB_COLUMN}' column.")

    box_path = csv_path.with_name(csv_path.stem + "_box.pdf")
    hist_path = csv_path.with_name(csv_path.stem + "_hist.pdf")

    _plot_box(df, box_path)
    stats = _plot_hist(df, hist_path)

    print(f"  → saved {box_path}")
    print(f"  → saved {hist_path}")
    _print_stats(csv_path, stats)


def main():
    parser = argparse.ArgumentParser(description="Plot class-score distributions from run_autoffs.py CSVs")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more eval CSV paths")
    args = parser.parse_args()

    _apply_rc()
    for p in args.csv:
        plot_one(p)


if __name__ == "__main__":
    main()
