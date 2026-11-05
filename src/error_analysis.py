"""Error analysis for the CNN and the pretrained model, using saved predictions.

1. Which breeds lose most accuracy under moderate corruption (level 2)?
2. Which breeds are confused with each other, on clean and corrupted images?
3. Example images that the pretrained model gets right when clean and wrong
   when corrupted.
4. How often a cat is predicted as a dog or the other way around, per condition,
   and how much the predictions collapse onto a few breeds (the share of all
   test images given the two most predicted breeds).

The baseline is left out: at about 9 % accuracy (chance is 2.7 %) its errors
are close to random and say little about which breeds are hard.
No model is run here; everything comes from the prediction files.

Usage (from the repo root):
    python src/error_analysis.py

Output:
    results/error_analysis/breed_drop.csv    accuracy per breed, clean vs. level 2
    results/error_analysis/confusions.csv    most common (true -> predicted) pairs
    results/error_analysis/species_errors.csv  cat <-> dog mistakes per condition
    figures/error_examples.png               clean vs. corrupted example images
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from corruptions import load_image

PREDICTIONS = {
    "CNN from scratch": [f"results/cnn/e80_seed{s}/predictions_corrupted.csv" for s in (42, 43, 44)],
    "ResNet-18 + MLP head": [f"results/pretrained/seed{s}/predictions.csv" for s in (42, 43, 44)],
}
KINDS = ["blur", "noise", "jpeg"]
CONDITIONS = ["clean"] + [f"{k}_{lvl}" for k in KINDS for lvl in (1, 2, 3)]
LEVEL = 2        # moderate corruption, used for the breed and confusion analysis
TOP_PAIRS = 10   # confusion pairs to keep per model and condition
OUT = Path("results/error_analysis")


def read_predictions(path):
    """Return {condition: {image: (label, pred)}}."""
    preds = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            preds[r["condition"]][r["image"]] = (int(r["label"]), int(r["pred"]))
    return preds


def main():
    with open("splits/test.csv") as f:
        breeds = {int(r["label"]): r["breed"] for r in csv.DictReader(f)}
    runs = {m: [read_predictions(p) for p in paths] for m, paths in PREDICTIONS.items()}
    corrupted = [f"{k}_{LEVEL}" for k in KINDS]
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy per breed: clean vs. level 2 (mean over seeds and over the 3 corruptions)
    breed_rows = []
    for model, seeds in runs.items():
        for label, breed in sorted(breeds.items(), key=lambda kv: kv[1].lower()):
            def acc(cond):
                vals = [np.mean([p == y for y, p in run[cond].values() if y == label]) for run in seeds]
                return float(np.mean(vals))
            clean = acc("clean")
            level2 = float(np.mean([acc(c) for c in corrupted]))
            breed_rows.append({"model": model, "breed": breed, "clean_acc": clean,
                               "level2_acc": level2, "drop": clean - level2})
    with open(OUT / "breed_drop.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=breed_rows[0].keys())
        writer.writeheader()
        writer.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
                         for r in breed_rows)

    # 2. Most common confusions (true -> predicted), counted per seed and averaged
    pair_rows = []
    for model, seeds in runs.items():
        for name, conds in [("clean", ["clean"]), (f"level {LEVEL} (all corruptions)", corrupted)]:
            counts = Counter()
            for run in seeds:
                for cond in conds:
                    counts.update((y, p) for y, p in run[cond].values() if y != p)
            for (y, p), n in counts.most_common(TOP_PAIRS):
                pair_rows.append({"model": model, "condition": name, "true": breeds[y],
                                  "predicted": breeds[p],
                                  "count_per_seed": n / len(seeds) / len(conds)})
    with open(OUT / "confusions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows({k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in r.items()}
                         for r in pair_rows)

    # 3. Examples: pretrained model (seed 42) right on clean, wrong at level 2.
    #    Two images per corruption type, picked at random with a fixed seed.
    run = runs["ResNet-18 + MLP head"][0]
    rng = np.random.default_rng(0)
    examples = []
    for kind in KINDS:
        cond = f"{kind}_{LEVEL}"
        candidates = sorted(img for img, (y, p) in run["clean"].items()
                            if y == p and run[cond][img][1] != y)
        for img in rng.choice(candidates, size=2, replace=False):
            examples.append((img, kind, breeds[run[cond][img][0]], breeds[run[cond][img][1]]))

    image_dir = Path("data/oxford-iiit-pet/images")
    fig, axes = plt.subplots(2, len(examples), figsize=(2.2 * len(examples), 5.8),
                             layout="constrained")
    for col, (img, kind, true, pred) in enumerate(examples):
        for row, (k, title) in enumerate([("clean", f"clean\n{true}"),
                                          (kind, f"{kind} level {LEVEL}\npred: {pred}")]):
            axes[row, col].imshow(load_image(image_dir / f"{img}.jpg", k, 0 if k == "clean" else LEVEL))
            axes[row, col].set_title(title.replace("_", " "), fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("ResNet-18 + MLP head (seed 42): correct on clean images, wrong after corruption",
                 fontsize=10)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/error_examples.png", dpi=200)

    # 4. Species errors: in Oxford-IIIT Pets, cat breed names start with a capital
    #    letter and dog breed names do not (12 cat and 25 dog breeds).
    is_cat = {label: breed[0].isupper() for label, breed in breeds.items()}
    assert sum(is_cat.values()) == 12, "expected 12 cat breeds"
    species_rows = []
    for model, seeds in runs.items():
        for cond in CONDITIONS:
            rates, shares, top2 = [], [], []
            predicted = Counter()
            for run in seeds:
                pairs = list(run[cond].values())
                wrong = [(y, p) for y, p in pairs if y != p]
                cross = sum(is_cat[y] != is_cat[p] for y, p in wrong)
                rates.append(cross / len(pairs))
                shares.append(cross / len(wrong) if wrong else 0.0)
                counts = Counter(p for _, p in pairs)
                predicted.update(counts)
                top2.append(sum(n for _, n in counts.most_common(2)) / len(pairs))
            species_rows.append({"model": model, "condition": cond,
                                 "species_error_rate": float(np.mean(rates)),
                                 "share_of_errors": float(np.mean(shares)),
                                 "top2_share": float(np.mean(top2)),
                                 "most_predicted": ", ".join(breeds[k] for k, _ in
                                                             predicted.most_common(2))})
    with open(OUT / "species_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=species_rows[0].keys())
        writer.writeheader()
        writer.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
                         for r in species_rows)

    # Short printout
    for model in runs:
        rows = sorted((r for r in breed_rows if r["model"] == model), key=lambda r: -r["drop"])
        print(f"\n{model}: largest drops (clean -> level {LEVEL})")
        for r in rows[:5]:
            print(f"  {r['breed']:28s} {100 * r['clean_acc']:5.1f} -> {100 * r['level2_acc']:5.1f}")
        print(f"{model}: most common confusions on clean images")
        for r in [r for r in pair_rows if r["model"] == model and r["condition"] == "clean"][:5]:
            print(f"  {r['true']:28s} -> {r['predicted']:28s} {r['count_per_seed']:.1f} per seed")
        print(f"{model}: cat <-> dog mistakes (% of images / % of errors), "
              f"share given the 2 most predicted breeds")
        for r in [r for r in species_rows if r["model"] == model]:
            print(f"  {r['condition']:8s} {100 * r['species_error_rate']:5.1f} %  "
                  f"{100 * r['share_of_errors']:5.1f} %   top-2 {100 * r['top2_share']:5.1f} %"
                  f"  ({r['most_predicted']})")
    print("\nExamples:", ", ".join(e[0] for e in examples))
    print(f"Saved {OUT}/breed_drop.csv, {OUT}/confusions.csv, {OUT}/species_errors.csv "
          f"and figures/error_examples.png")


if __name__ == "__main__":
    main()
