"""Evaluate trained CNN checkpoints on the clean and corrupted test sets.

train_cnn.py trains the CNN and saves the weights of the best validation
epoch. This script loads those weights and tests them on all 10 conditions
(clean + 3 corruptions x 3 levels). No training happens here.

Images are loaded through corruptions.load_image() (resize + centre crop to
224x224, then the corruption) and then resized to the CNN's input size. The
clean condition goes through exactly the same steps, so every accuracy drop
is caused by the corruption alone.

Usage (from the repo root), after training with train_cnn.py:
    python src/evaluate_cnn.py                  # seeds 42 43 44 (skips missing ones)
    python src/evaluate_cnn.py --seeds 42

Input:
    checkpoints/cnn_seed42.pt          weights from train_cnn.py
    results/cnn/seed42/metrics.json    used to read the training config

Output:
    results/cnn/seed42/corruptions.json            metrics per condition
    results/cnn/seed42/predictions_corrupted.csv   predictions for every condition
    results/cnn/summary.json                       mean and std over seeds
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from corruptions import CONDITIONS, condition_name, load_image
from train_cnn import MEAN, STD, SmallCNN


class CorruptedPets(Dataset):
    """Test images for one condition, resized to the CNN's input size."""

    def __init__(self, rows, image_dir, kind, level, size):
        self.rows, self.image_dir = rows, Path(image_dir)
        self.kind, self.level = kind, level
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),  # 224x224 crop -> size x size
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(self.image_dir / f"{r['image']}.jpg", self.kind, self.level)
        return self.tf(img), int(r["label"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    image_dir = Path(args.data_root) / "oxford-iiit-pet" / "images"
    with open(Path(args.splits) / "test.csv") as f:
        test_rows = list(csv.DictReader(f))
    breeds = {int(r["label"]): r["breed"] for r in test_rows}

    per_seed = {}
    for seed in args.seeds:
        run_dir = Path("results/cnn") / f"seed{seed}"
        ckpt = Path("checkpoints") / f"cnn_seed{seed}.pt"
        if not ckpt.exists() or not (run_dir / "metrics.json").exists():
            print(f"Seed {seed}: no checkpoint or metrics found, skipping "
                  f"(train it first with: python src/train_cnn.py --seed {seed})")
            continue
        with open(run_dir / "metrics.json") as f:
            config = json.load(f)["config"]

        model = SmallCNN(dropout=config["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        results, all_preds = {}, {}
        print(f"\nSeed {seed} (input size {config['size']}):")
        for kind, level in CONDITIONS:
            name = condition_name(kind, level)
            loader = DataLoader(CorruptedPets(test_rows, image_dir, kind, level,
                                              config["size"]),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
            pred, y = predict(model, loader, device)
            results[name] = {
                "accuracy": float((pred == y).mean()),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "per_class_accuracy": {breeds[c]: float((pred[y == c] == c).mean())
                                       for c in sorted(breeds)},
            }
            all_preds[name] = pred
            print(f"  {name:8s} acc={results[name]['accuracy']:.4f}")

        clean_acc = results["clean"]["accuracy"]
        for name in results:
            acc = results[name]["accuracy"]
            results[name]["absolute_drop"] = clean_acc - acc
            results[name]["relative_drop"] = (clean_acc - acc) / clean_acc

        with open(run_dir / "corruptions.json", "w") as f:
            json.dump({"seed": seed, "checkpoint": str(ckpt),
                       "test_conditions": results}, f, indent=2)
        with open(run_dir / "predictions_corrupted.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "image", "label", "pred", "correct"])
            for name, pred in all_preds.items():
                for r, p in zip(test_rows, pred):
                    writer.writerow([name, r["image"], r["label"], int(p),
                                     int(r["label"]) == int(p)])
        per_seed[seed] = results

    if not per_seed:
        print("No checkpoints evaluated.")
        return

    # Mean and standard deviation over the seeds that were evaluated
    seeds = sorted(per_seed)
    summary = {}
    for kind, level in CONDITIONS:
        name = condition_name(kind, level)
        accs = [per_seed[s][name]["accuracy"] for s in seeds]
        rel = [per_seed[s][name]["relative_drop"] for s in seeds]
        summary[name] = {"accuracy_mean": float(np.mean(accs)),
                         "accuracy_std": float(np.std(accs)),
                         "relative_drop_mean": float(np.mean(rel)),
                         "relative_drop_std": float(np.std(rel))}
    with open(Path("results/cnn") / "summary.json", "w") as f:
        json.dump({"seeds": seeds, "conditions": summary}, f, indent=2)

    print(f"\nMean over seeds {seeds}:")
    for name, s in summary.items():
        print(f"  {name:8s} acc={s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}  "
              f"relative drop={s['relative_drop_mean']:6.1%}")
    print("Saved results to results/cnn/")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()