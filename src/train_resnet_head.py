"""Train an MLP head on frozen ImageNet-pretrained ResNet-50 features.

The ResNet-50 backbone is frozen: its weights never change. Every image is
passed through it once, the 2048-dimensional feature vectors are cached, and
only a small MLP head is trained on those vectors. The head with the best
validation accuracy is evaluated once on the clean test set.

Note: the training images are not augmented here, because the features are
computed once from a fixed resize and centre crop. This differs from the CNN
and should be mentioned when the two are compared.

Usage (from the repo root):
    python src/train_resnet_head.py --seed 42

Output (per run; the run name defaults to seed<seed>):
    results/resnet_head/seed42/history.csv      loss and accuracy per epoch
    results/resnet_head/seed42/metrics.json     config, best epoch, test metrics
    results/resnet_head/seed42/predictions.csv  test predictions, for error analysis
    checkpoints/resnet_head_seed42.pt           head weights of the best epoch
    checkpoints/features/resnet50_<split>.pt    cached features (recreated if missing)
    checkpoints/torch_hub/                      downloaded ResNet-50 weights (~100 MB)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

# Reuse the dataset, seeding and training loop from the CNN script
from train_cnn import MEAN, NUM_CLASSES, STD, PetsDataset, run_epoch, set_seed

FEATURE_DIM = 2048  # size of ResNet-50's pooled feature vector
WEIGHTS_NAME = "IMAGENET1K_V2"


def eval_transform():
    """Standard ImageNet evaluation preprocessing: resize, centre crop 224, normalise."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_backbone(device):
    """ResNet-50 with ImageNet weights, final layer removed, all weights frozen."""
    weights = models.ResNet50_Weights[WEIGHTS_NAME]
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()  # output the 2048 pooled features, not 1000 ImageNet scores
    for p in model.parameters():
        p.requires_grad = False
    return model.eval().to(device)


class MLPHead(nn.Module):
    """One hidden layer with ReLU and dropout, then one score per breed."""

    def __init__(self, in_dim=FEATURE_DIM, hidden=512, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits; softmax is inside the loss


@torch.no_grad()
def extract_features(backbone, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def get_features(args, device):
    """Load cached features, or compute and cache them if any split is missing."""
    feature_dir = Path(args.feature_dir)
    paths = {s: feature_dir / f"resnet50_{s}.pt" for s in ("train", "val", "test")}
    missing = [s for s, p in paths.items() if not p.exists()]
    extract_minutes = None

    if missing:
        torch.hub.set_dir(args.hub_dir)  # keep the download out of C:\Users\...\.cache
        backbone = load_backbone(device)
        image_dir = Path(args.data_root) / "oxford-iiit-pet" / "images"
        feature_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        for split in missing:
            ds = PetsDataset(Path(args.splits) / f"{split}.csv", image_dir, eval_transform())
            print(f"Extracting ResNet-50 features for {split} ({len(ds)} images) ...")
            f, y = extract_features(backbone, ds, device, args.extract_batch_size, args.workers)
            torch.save({"features": f, "labels": y}, paths[split])
        extract_minutes = round((time.time() - start) / 60, 1)
        del backbone

    data = {s: torch.load(p) for s, p in paths.items()}
    return ({s: (d["features"], d["labels"]) for s, d in data.items()}, extract_minutes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--feature-dir", default="checkpoints/features")
    parser.add_argument("--hub-dir", default="checkpoints/torch_hub")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None,
                        help="name of the output folder and checkpoint (default: seed<seed>)")
    args = parser.parse_args()
    run_name = args.run_name or f"seed{args.seed}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device_name}")

    feats, extract_minutes = get_features(args, device)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = feats["train"], feats["val"], feats["test"]
    print(f"Features: train {tuple(X_train.shape)}, val {tuple(X_val.shape)}, test {tuple(X_test.shape)}")

    # Features are small and already in memory, so no worker processes are needed
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                              shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    head = MLPHead(hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"MLP head with {n_params:,} trainable parameters (backbone frozen)")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path("results/resnet_head") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("checkpoints") / f"resnet_head_{run_name}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)

    history, best_val, best_epoch = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(head, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc, _, _ = run_epoch(head, val_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        marker = ""
        if va_acc > best_val:  # keep the head from the best validation epoch
            best_val, best_epoch = va_acc, epoch
            torch.save({"state_dict": head.state_dict(), "hidden": args.hidden,
                        "dropout": args.dropout, "backbone": "resnet50",
                        "weights": WEIGHTS_NAME}, ckpt_path)
            marker = "  *"
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.3f}  train_acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.3f}  val_acc={va_acc:.3f}{marker}")
    train_minutes = (time.time() - start) / 60

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Final evaluation on test, once, with the best head
    head.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    _, test_acc, test_pred, test_y = run_epoch(head, test_loader, loss_fn, device)
    test_f1 = f1_score(test_y, test_pred, average="macro")
    print(f"Best epoch {best_epoch} (val_acc={best_val:.4f}). "
          f"Test accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}  "
          f"({train_minutes:.2f} min head training)")

    with open(Path(args.splits) / "test.csv") as f:
        test_rows = list(csv.DictReader(f))
    breeds = {int(r["label"]): r["breed"] for r in test_rows}
    per_class = {breeds[c]: float((test_pred[test_y == c] == c).mean())
                 for c in sorted(breeds)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config": vars(args),
            "backbone": f"torchvision resnet50, weights {WEIGHTS_NAME}, frozen",
            "device": device_name,
            "trainable_parameters": n_params,
            "feature_extraction_minutes": extract_minutes,  # None if cached features were used
            "train_minutes": round(train_minutes, 2),
            "best_epoch": best_epoch,
            "best_val_acc": best_val,
            "test": {"accuracy": test_acc, "macro_f1": test_f1},
            "test_per_class_accuracy": per_class,
        }, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "pred", "correct"])
        for r, p in zip(test_rows, test_pred):
            writer.writerow([r["image"], r["label"], int(p), int(r["label"]) == int(p)])
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()
