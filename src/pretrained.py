"""Frozen pretrained ResNet-18 features + a small MLP classifier head.

Step 1 - feature extraction (slow part, done once and cached):
    Every image goes through ResNet-18 pretrained on ImageNet, with the final
    classification layer removed. The output is a 512-number feature vector
    per image. The ResNet weights are never changed ("frozen"). Features are
    extracted for train, val, and the clean + 9 corrupted versions of test.

Step 2 - MLP head (fast part):
    A small neural network (512 -> hidden -> 37) is trained on the clean train
    features with cross-entropy loss and the Adam optimiser. The settings are
    chosen on the validation set, and the epoch with the best validation
    accuracy is kept. This is repeated for several seeds.

Step 3 - evaluation:
    Each trained head is evaluated once on the clean test set and on all 9
    corrupted test sets, so the drop in accuracy can be compared with the
    other models.

All images are loaded through corruptions.load_image(), so this model sees
exactly the same clean and corrupted pixels as the other models.

Usage (from the repo root):
    python src/pretrained.py

Output:
    data/features/*.pt                       cached features (ignored by git)
    results/pretrained/seed42/metrics.json   config, validation grid, test metrics
    results/pretrained/seed42/predictions.csv  test predictions for every condition
    results/pretrained/summary.json          mean and std over seeds, per condition
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from corruptions import CONDITIONS, condition_name, load_image

NUM_CLASSES = 37
# ImageNet statistics: the pretrained ResNet expects inputs normalised this way
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Step 1: feature extraction
# ---------------------------------------------------------------------------

class PetsImages(Dataset):
    """Returns (normalised 224x224 image tensor, label) for one condition."""

    def __init__(self, rows, image_dir, kind, level):
        self.rows, self.image_dir = rows, Path(image_dir)
        self.kind, self.level = kind, level
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(self.image_dir / f"{r['image']}.jpg", self.kind, self.level)
        return self.to_tensor(img), int(r["label"])


def build_encoder(device):
    """ResNet-18 pretrained on ImageNet, with the last layer removed."""
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()  # output the 512 features instead of 1000 classes
    encoder.eval()              # frozen: batch norm uses its stored statistics
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


@torch.no_grad()
def extract(encoder, rows, image_dir, kind, level, device, args):
    loader = DataLoader(PetsImages(rows, image_dir, kind, level),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    feats, labels = [], []
    for x, y in loader:
        feats.append(encoder(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def get_features(name, rows, kind, level, encoder_holder, device, args):
    """Load cached features if they exist, otherwise extract and cache them."""
    path = Path(args.feature_dir) / f"resnet18_{name}.pt"
    if path.exists():
        data = torch.load(path)
        return data["x"], data["y"]
    if encoder_holder["model"] is None:
        encoder_holder["model"] = build_encoder(device)
    image_dir = Path(args.data_root) / "oxford-iiit-pet" / "images"
    start = time.time()
    x, y = extract(encoder_holder["model"], rows, image_dir, kind, level, device, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, path)
    print(f"  extracted {name:14s} {len(y):5d} images in {time.time() - start:6.1f} s")
    return x, y


# ---------------------------------------------------------------------------
# Step 2: MLP head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """512 features -> hidden layer with ReLU and dropout -> 37 class scores."""

    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)  # raw scores (logits); softmax is inside the loss


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x).argmax(1)


def train_head(x_tr, y_tr, x_va, y_va, hidden, weight_decay, seed, args):
    """Train one head; return the weights from the best validation epoch."""
    set_seed(seed)
    model = MLPHead(x_tr.shape[1], hidden, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)

    history, best_val, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y_tr), generator=g)
        total_loss = 0.0
        for i in range(0, len(order), args.head_batch_size):
            idx = order[i:i + args.head_batch_size]
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_acc = float((predict(model, x_tr) == y_tr).float().mean())
        val_acc = float((predict(model, x_va) == y_va).float().mean())
        history.append({"epoch": epoch, "train_loss": total_loss / len(y_tr),
                        "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_val:  # keep the weights from the best validation epoch
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val, best_epoch, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--feature-dir", default="data/features")
    parser.add_argument("--batch-size", type=int, default=64, help="for feature extraction")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-4, 1e-3])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out-dir", default="results/pretrained")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device_name}")
    splits = Path(args.splits)
    rows = {s: read_split(splits / f"{s}.csv") for s in ("train", "val", "test")}

    # --- Step 1: features (cached after the first run) ---
    print("Feature extraction (cached files are reused):")
    start = time.time()
    holder = {"model": None}
    x_tr, y_tr = get_features("train", rows["train"], "clean", 0, holder, device, args)
    x_va, y_va = get_features("val", rows["val"], "clean", 0, holder, device, args)
    test = {}
    for kind, level in CONDITIONS:
        name = condition_name(kind, level)
        test[name] = get_features(f"test_{name}", rows["test"], kind, level,
                                  holder, device, args)
    extract_minutes = (time.time() - start) / 60

    # Standardise features with statistics from the training set only
    mean, std = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_va = (x_tr - mean) / std, (x_va - mean) / std
    test = {k: ((x - mean) / std, y) for k, (x, y) in test.items()}

    # --- Step 2a: choose hidden size and weight decay on validation ---
    print("\nValidation grid (seed {}):".format(args.seeds[0]))
    grid, best = [], None
    for hidden in args.hidden:
        for wd in args.weight_decay:
            _, val_acc, epoch, _ = train_head(x_tr, y_tr, x_va, y_va, hidden, wd,
                                              args.seeds[0], args)
            grid.append({"hidden": hidden, "weight_decay": wd,
                         "val_acc": val_acc, "best_epoch": epoch})
            print(f"  hidden={hidden:4d}  weight_decay={wd:<7}  "
                  f"val_acc={val_acc:.4f}  (best epoch {epoch})")
            if best is None or val_acc > best["val_acc"]:
                best = grid[-1]
    print(f"Selected: hidden={best['hidden']}, weight_decay={best['weight_decay']}")

    # --- Step 2b + 3: train one head per seed, evaluate on every test condition ---
    breeds = {int(r["label"]): r["breed"] for r in rows["test"]}
    per_seed = {}
    for seed in args.seeds:
        start = time.time()
        model, val_acc, best_epoch, history = train_head(
            x_tr, y_tr, x_va, y_va, best["hidden"], best["weight_decay"], seed, args)
        train_seconds = time.time() - start

        results, all_preds = {}, {}
        for name, (x, y) in test.items():
            pred = predict(model, x)
            y_np, p_np = y.numpy(), pred.numpy()
            results[name] = {
                "accuracy": float((p_np == y_np).mean()),
                "macro_f1": float(f1_score(y_np, p_np, average="macro")),
                "per_class_accuracy": {breeds[c]: float((p_np[y_np == c] == c).mean())
                                       for c in sorted(breeds)},
            }
            all_preds[name] = p_np
        clean_acc = results["clean"]["accuracy"]
        for name in results:
            acc = results[name]["accuracy"]
            results[name]["absolute_drop"] = clean_acc - acc
            results[name]["relative_drop"] = (clean_acc - acc) / clean_acc

        print(f"\nSeed {seed}: best epoch {best_epoch}, val_acc={val_acc:.4f}")
        for name in test:
            r = results[name]
            print(f"  {name:8s} acc={r['accuracy']:.4f}  "
                  f"relative drop={r['relative_drop']:6.1%}")

        out = Path(args.out_dir) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        with open(out / "metrics.json", "w") as f:
            json.dump({
                "config": {**vars(args), "seed": seed},
                "device": device_name,
                "feature_extraction_minutes": round(extract_minutes, 1),
                "head_train_seconds": round(train_seconds, 1),
                "validation_grid": grid,
                "selected": {"hidden": best["hidden"],
                             "weight_decay": best["weight_decay"]},
                "best_epoch": best_epoch,
                "best_val_acc": val_acc,
                "test": {"accuracy": results["clean"]["accuracy"],
                         "macro_f1": results["clean"]["macro_f1"]},
                "test_conditions": results,
            }, f, indent=2)
        with open(out / "predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "image", "label", "pred", "correct"])
            for name, p_np in all_preds.items():
                for r, p in zip(rows["test"], p_np):
                    writer.writerow([name, r["image"], r["label"], int(p),
                                     int(r["label"]) == int(p)])
        per_seed[seed] = results

    # --- Summary over seeds: mean and standard deviation per condition ---
    summary = {}
    for name in test:
        accs = [per_seed[s][name]["accuracy"] for s in args.seeds]
        rel = [per_seed[s][name]["relative_drop"] for s in args.seeds]
        summary[name] = {"accuracy_mean": float(np.mean(accs)),
                         "accuracy_std": float(np.std(accs)),
                         "relative_drop_mean": float(np.mean(rel)),
                         "relative_drop_std": float(np.std(rel))}
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump({"seeds": args.seeds, "conditions": summary}, f, indent=2)

    print(f"\nMean over seeds {args.seeds}:")
    for name, s in summary.items():
        print(f"  {name:8s} acc={s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}  "
              f"relative drop={s['relative_drop_mean']:6.1%}")
    print(f"Saved results to {args.out_dir}/")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()