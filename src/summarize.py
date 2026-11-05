"""Collect the results of all three models into one table and one figure.

Reads the per-run result files written by baseline.py, evaluate_cnn.py and
pretrained.py, and computes mean and standard deviation over runs (seeds) in
the same way for every model.

Usage (from the repo root):
    python src/summarize.py

Output:
    results/summary_all.csv     one row per model and test condition
    figures/degradation.png     accuracy and retained accuracy vs. severity
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Result files per model: one file per run (seed), each with "test_conditions"
RUNS = {
    "Baseline (PCA + LogReg)": ["results/baseline/metrics.json"],
    "CNN from scratch": [f"results/cnn/e80_seed{s}/corruptions.json" for s in (42, 43, 44)],
    "ResNet-18 + MLP head": [f"results/pretrained/seed{s}/metrics.json" for s in (42, 43, 44)],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
KINDS = {"blur": "Gaussian blur", "noise": "Gaussian noise", "jpeg": "JPEG compression"}
DDOF = 1         # sample standard deviation over runs (n - 1)
CHANCE = 1 / 37  # random guessing over 37 breeds


def load_accuracies(paths):
    """Return {condition: [accuracy per run]}."""
    runs = [json.load(open(p))["test_conditions"] for p in paths]
    return {c: [r[c]["accuracy"] for r in runs] for c in runs[0]}


def sd(values):
    return float(np.std(values, ddof=DDOF)) if len(values) > 1 else float("nan")


def main():
    rows = []
    for model, paths in RUNS.items():
        acc = load_accuracies(paths)
        clean = np.array(acc["clean"])
        for cond, values in acc.items():
            values = np.array(values)
            retained = values / clean  # accuracy as a fraction of the same run's clean accuracy
            kind, level = ("clean", 0) if cond == "clean" else (cond.split("_")[0], int(cond.split("_")[1]))
            rows.append({"model": model, "condition": cond, "kind": kind, "level": level,
                         "runs": len(values),
                         "accuracy_mean": values.mean(), "accuracy_sd": sd(values),
                         "retained_mean": retained.mean(), "retained_sd": sd(retained)})

    Path("results").mkdir(exist_ok=True)
    with open("results/summary_all.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Figure: top row accuracy, bottom row accuracy relative to clean; one column per corruption
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey="row")
    for col, (kind, title) in enumerate(KINDS.items()):
        for i, model in enumerate(RUNS):
            pts = sorted((r for r in rows if r["model"] == model and r["kind"] in ("clean", kind)),
                         key=lambda r: r["level"])
            x = [r["level"] for r in pts]
            for row, key in enumerate(["accuracy", "retained"]):
                y = [100 * r[f"{key}_mean"] for r in pts]
                err = [0 if np.isnan(r[f"{key}_sd"]) else 100 * r[f"{key}_sd"] for r in pts]
                axes[row, col].errorbar(x, y, yerr=err, color=COLORS[i], marker=MARKERS[i],
                                        markersize=6, linewidth=2, capsize=3, label=model)
        axes[0, col].axhline(100 * CHANCE, color="#888888", linestyle="--", linewidth=1)
        axes[0, col].set_title(title)
        axes[1, col].set_xticks([0, 1, 2, 3], ["clean", "1", "2", "3"])
        axes[1, col].set_xlabel("Severity level")
    axes[0, 0].text(0.6, 100 * CHANCE + 1.2, "chance", ha="left", fontsize=8, color="#666666")
    axes[0, 0].set_ylabel("Test accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy retained\n(% of clean accuracy)")
    for ax in axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.005, f"Mean over runs; error bars show the standard deviation over seeds "
             f"(ddof={DDOF}). The baseline is a single run.", ha="center", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/degradation.png", dpi=200)

    print("Saved results/summary_all.csv and figures/degradation.png")
    for model in RUNS:
        line = "  ".join(f"{r['condition']}={100 * r['accuracy_mean']:.1f}"
                         for r in rows if r["model"] == model)
        print(f"{model}: {line}")


if __name__ == "__main__":
    main()
