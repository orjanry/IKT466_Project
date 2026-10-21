"""Download Oxford-IIIT Pets and create fixed train/val/test splits.
 
The official test split is kept as is. A stratified validation set is
carved out of the official trainval split with a fixed seed, so every
breed appears in train, val and test.
 
Usage (from the repo root):
    python src/make_splits.py
 
Output:
    splits/train.csv, splits/val.csv, splits/test.csv
    Each row: image (file name without .jpg), label (0-36), breed
"""
 
import argparse
import csv
from collections import Counter
from pathlib import Path
 
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet
 
 
def read_annotation_file(path):
    """Read an official Pets annotation file (trainval.txt or test.txt).
 
    Each line looks like: 'Abyssinian_100 1 1 1'
    (image name, class id 1-37, species, breed id).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            image, class_id, *_ = line.split()
            breed = image.rsplit("_", 1)[0]
            rows.append({"image": image, "label": int(class_id) - 1, "breed": breed})
    return rows
 
 
def write_split(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "breed"])
        writer.writeheader()
        writer.writerows(rows)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
 
    # Download the dataset (about 800 MB) into data/oxford-iiit-pet/
    if not args.no_download:
        OxfordIIITPet(args.data_root, split="trainval", download=True)
        OxfordIIITPet(args.data_root, split="test", download=True)
 
    ann_dir = Path(args.data_root) / "oxford-iiit-pet" / "annotations"
    trainval = read_annotation_file(ann_dir / "trainval.txt")
    test = read_annotation_file(ann_dir / "test.txt")
 
    # Stratified split: each breed keeps the same proportion in train and val
    train, val = train_test_split(
        trainval,
        test_size=args.val_fraction,
        stratify=[r["label"] for r in trainval],
        random_state=args.seed,
    )
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        write_split(rows, out_dir / f"{name}.csv")
 
    # Sanity checks
    n_classes = len({r["label"] for r in trainval})
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(r["label"] for r in rows)
        assert len(counts) == n_classes, f"{name} is missing classes"
        print(f"{name:5s}: {len(rows):5d} images, "
              f"{min(counts.values())}-{max(counts.values())} per class")
 
    overlap = {r["image"] for r in train} & {r["image"] for r in val}
    assert not overlap, "train and val overlap"
    print(f"Splits saved to {out_dir}/ (seed={args.seed})")
 
 
if __name__ == "__main__":
    main()
 
