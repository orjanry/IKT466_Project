"""Baseline: resized pixels -> standardise -> PCA -> logistic regression.

Hyperparameters (number of PCA components and regularisation C) are chosen
on the validation set. The selected model is then evaluated once on the
clean test set and on the 9 corrupted test sets.

All images are loaded through corruptions.load_image() (resize + centre crop
to 224x224, then the corruption), and only afterwards shrunk to size x size.
This way the corruption is applied at the same resolution for every model.

Usage (from the repo root):
    python src/baseline.py

Output:
    results/baseline/metrics.json      config, validation grid, test metrics
    results/baseline/predictions.csv   test predictions for every condition
    results/baseline/summary.json      accuracy and drop per condition
    checkpoints/baseline.joblib        fitted pipeline
"""

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from corruptions import CONDITIONS, condition_name, load_image


def read_split(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_images(rows, image_dir, size, kind="clean", level=0):
    """Load images through load_image (224x224, optionally corrupted),
    shrink to size x size and flatten to vectors in [0, 1]."""
    X = np.empty((len(rows), size * size * 3), dtype=np.float32)
    for i, r in enumerate(rows):
        img = load_image(image_dir / f"{r['image']}.jpg", kind, level)
        img = img.resize((size, size), Image.BILINEAR)
        X[i] = np.asarray(img, dtype=np.float32).ravel() / 255.0
    y = np.array([int(r["label"]) for r in rows])
    return X, y


def make_pipeline(n_components, C, seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--size", type=int, default=32, help="resize images to size x size")
    parser.add_argument("--components", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/baseline")
    args = parser.parse_args()

    image_dir = Path(args.data_root) / "oxford-iiit-pet" / "images"
    splits = Path(args.splits)
    train_rows = read_split(splits / "train.csv")
    val_rows = read_split(splits / "val.csv")
    test_rows = read_split(splits / "test.csv")

    print(f"Loading images at {args.size}x{args.size} ...")
    X_train, y_train = load_images(train_rows, image_dir, args.size)
    X_val, y_val = load_images(val_rows, image_dir, args.size)

    # Model selection on validation data (fitted on train only)
    grid = []
    best = None
    for n in args.components:
        for C in args.C:
            start = time.time()
            model = make_pipeline(n, C, args.seed).fit(X_train, y_train)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            grid.append({"components": n, "C": C, "val_acc": val_acc,
                         "fit_seconds": round(time.time() - start, 1)})
            print(f"components={n:4d}  C={C:<6}  val_acc={val_acc:.4f}")
            if best is None or val_acc > best[0]:
                best = (val_acc, n, C, model)

    best_val, best_n, best_C, model = best
    explained = float(model.named_steps["pca"].explained_variance_ratio_.sum())
    print(f"Best on val: components={best_n}, C={best_C}, val_acc={best_val:.4f} "
          f"(PCA keeps {explained:.1%} of variance)")

    # Final evaluation, once, on the clean test set and every corrupted test set
    breeds = {int(r["label"]): r["breed"] for r in test_rows}
    results, all_preds = {}, {}
    print("\nTest set:")
    for kind, level in CONDITIONS:
        name = condition_name(kind, level)
        X_test, y_test = load_images(test_rows, image_dir, args.size, kind, level)
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "per_class_accuracy": {breeds[c]: float((pred[y_test == c] == c).mean())
                                   for c in sorted(breeds)},
        }
        all_preds[name] = pred
        print(f"  {name:8s} acc={results[name]['accuracy']:.4f}")

    clean_acc = results["clean"]["accuracy"]
    for name in results:
        acc = results[name]["accuracy"]
        results[name]["absolute_drop"] = clean_acc - acc
        results[name]["relative_drop"] = (clean_acc - acc) / clean_acc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config": vars(args),
            "selected": {"components": best_n, "C": best_C, "val_acc": best_val,
                         "pca_explained_variance": explained},
            "validation_grid": grid,
            "test": {"accuracy": results["clean"]["accuracy"],
                     "macro_f1": results["clean"]["macro_f1"]},
            "test_conditions": results,
        }, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "image", "label", "pred", "correct"])
        for name, pred in all_preds.items():
            for r, p in zip(test_rows, pred):
                writer.writerow([name, r["image"], r["label"], int(p),
                                 int(r["label"]) == int(p)])

    # Same format as the other models' summary.json. Logistic regression is
    # deterministic, so there is one run and the standard deviation is zero.
    summary = {name: {"accuracy_mean": r["accuracy"], "accuracy_std": 0.0,
                      "relative_drop_mean": r["relative_drop"], "relative_drop_std": 0.0}
               for name, r in results.items()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"seeds": [args.seed], "conditions": summary}, f, indent=2)

    Path("checkpoints").mkdir(exist_ok=True)
    joblib.dump({"model": model, "size": args.size}, "checkpoints/baseline.joblib")
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()