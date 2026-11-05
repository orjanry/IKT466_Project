"""Train a small CNN from scratch on Oxford-IIIT Pets.

The model with the best validation accuracy (checked after every epoch) is
kept and evaluated once on the clean test set.

Usage (from the repo root):
    python src/train_cnn.py --seed 42
    python src/train_cnn.py --seed 42 --epochs 80 --run-name e80_seed42

Output (per run; the run name defaults to seed<seed>):
    results/cnn/seed42/history.csv      loss and accuracy per epoch
    results/cnn/seed42/metrics.json     config, best epoch, test metrics
    results/cnn/seed42/predictions.csv  test predictions, for error analysis
    checkpoints/cnn_seed42.pt           weights of the best epoch
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
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet statistics, a common default for normalising natural images
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 37


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PetsDataset(Dataset):
    """Reads one of the split CSVs and returns (image tensor, label)."""

    def __init__(self, split_csv, image_dir, transform):
        with open(split_csv) as f:
            self.rows = list(csv.DictReader(f))
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.image_dir / f"{r['image']}.jpg").convert("RGB")
        return self.transform(img), int(r["label"])


def make_transforms(size):
    # Training: random crops and flips (data augmentation)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Validation/test: deterministic resize and centre crop
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # halves height and width
    )


class SmallCNN(nn.Module):
    """Four conv blocks (32-64-128-256 channels), global average pooling,
    dropout and a linear layer that outputs one score per breed."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)  # raw scores (logits); softmax is inside the loss


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return total_loss / len(labels), float((preds == labels).mean()), preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--size", type=int, default=128, help="input image size")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None,
                        help="name of the output folder and checkpoint (default: seed<seed>)")
    args = parser.parse_args()
    run_name = args.run_name or f"seed{args.seed}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    image_dir = Path(args.data_root) / "oxford-iiit-pet" / "images"
    splits = Path(args.splits)
    train_tf, eval_tf = make_transforms(args.size)
    train_ds = PetsDataset(splits / "train.csv", image_dir, train_tf)
    val_ds = PetsDataset(splits / "val.csv", image_dir, eval_tf)
    test_ds = PetsDataset(splits / "test.csv", image_dir, eval_tf)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda",
                       persistent_workers=args.workers > 0)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = SmallCNN(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SmallCNN with {n_params:,} parameters")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Learning rate decreases smoothly from lr to ~0 over the training run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path("results/cnn") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("checkpoints") / f"cnn_{run_name}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)

    history, best_val, best_epoch = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        marker = ""
        if va_acc > best_val:  # keep the weights from the best validation epoch
            best_val, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
            marker = "  *"
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.3f}  train_acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.3f}  val_acc={va_acc:.3f}{marker}")
    train_minutes = (time.time() - start) / 60

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Final evaluation on test, once, with the best checkpoint
    # Shut down the persistent train/val workers before starting test workers.
    # Otherwise Windows can run out of page file space (WinError 1455), since
    # every worker process loads its own copy of the CUDA libraries.
    del train_loader, val_loader
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc, test_pred, test_y = run_epoch(model, test_loader, loss_fn, device)
    test_f1 = f1_score(test_y, test_pred, average="macro")
    print(f"Best epoch {best_epoch} (val_acc={best_val:.4f}). "
          f"Test accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}  "
          f"({train_minutes:.1f} min training)")

    breeds = {int(r["label"]): r["breed"] for r in test_ds.rows}
    per_class = {breeds[c]: float((test_pred[test_y == c] == c).mean())
                 for c in sorted(breeds)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config": vars(args),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "parameters": n_params,
            "train_minutes": round(train_minutes, 1),
            "best_epoch": best_epoch,
            "best_val_acc": best_val,
            "test": {"accuracy": test_acc, "macro_f1": test_f1},
            "test_per_class_accuracy": per_class,
        }, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "pred", "correct"])
        for r, p in zip(test_ds.rows, test_pred):
            writer.writerow([r["image"], r["label"], int(p), int(r["label"]) == int(p)])
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()