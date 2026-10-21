"""Baseline: resized pixels -> standardise -> PCA -> logistic regression.

Hyperparameters (number of PCA components and regularisation C) are chosen
on the validation set. The test set is only used once, for the final model.

Usage (from the repo root):
    python src/baseline.py

Output:
    results/baseline/metrics.json      config, validation grid, test metrics
    results/baseline/predictions.csv   test predictions, for error analysis
    checkpoints/baseline.joblib        fitted pipeline (reused for corruption tests)
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


def read_split(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_images(rows, image_dir, size):
    """Load images, resize to size x size RGB and flatten to vectors in [0, 1]."""
    X = np.empty((len(rows), size * size * 3), dtype=np.float32)
    for i, r in enumerate(rows):
        img = Image.open(image_dir / f"{r['image']}.jpg").convert("RGB")
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
    X_test, y_test = load_images(test_rows, image_dir, args.size)

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
    print(f"Best on val: components={best_n}, C={best_C}, val_acc={best_val:.4f}")

    # Final evaluation on test, once
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    explained = float(model.named_steps["pca"].explained_variance_ratio_.sum())
    print(f"Test accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}  "
          f"(PCA keeps {explained:.1%} of variance)")

    # Per-breed accuracy, useful for the error analysis
    breeds = {int(r["label"]): r["breed"] for r in test_rows}
    per_class = {breeds[c]: float((test_pred[y_test == c] == c).mean())
                 for c in sorted(breeds)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config": vars(args),
            "selected": {"components": best_n, "C": best_C, "val_acc": best_val,
                         "pca_explained_variance": explained},
            "validation_grid": grid,
            "test": {"accuracy": test_acc, "macro_f1": test_f1},
            "test_per_class_accuracy": per_class,
        }, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "pred", "correct"])
        for r, p in zip(test_rows, test_pred):
            writer.writerow([r["image"], r["label"], int(p), int(r["label"]) == int(p)])

    Path("checkpoints").mkdir(exist_ok=True)
    joblib.dump({"model": model, "size": args.size}, "checkpoints/baseline.joblib")
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()