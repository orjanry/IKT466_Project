"""Image corruptions shared by all models: Gaussian blur, Gaussian noise, JPEG.

Every model loads its images through `load_image()`, so the baseline, the CNN
and the pretrained model all see exactly the same (corrupted) pixels:

    original image -> resize + centre crop to 224x224 -> corruption
                   -> (model's own resizing, e.g. 32x32 or 128x128)

Corruptions are applied at 224x224 *before* any model-specific downsizing.
This matters because shrinking an image (e.g. to 32x32 for the baseline)
averages neighbouring pixels and would hide much of the noise and blur.

Noise is random, so it is seeded per image (from the image name). The same
image therefore gets the same noise in every script and on every run,
regardless of the order in which images are loaded.

Usage (from the repo root), to save example images for checking the levels:
    python src/corruptions.py

Output:
    figures/corruption_examples.png

Use in other scripts:
    from corruptions import CONDITIONS, load_image
    for kind, level in CONDITIONS:            # ("clean", 0), ("blur", 1), ...
        img = load_image(path, kind, level)   # PIL image, 224x224 RGB
"""

import argparse
import csv
import io
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

BASE_SIZE = 224   # every image is resized to this before corruption
NOISE_SEED = 42   # base seed for the Gaussian noise

# Severity levels 1 (mild), 2 (moderate), 3 (severe)
LEVELS = {
    "blur": {1: 1.0, 2: 2.0, 3: 4.0},     # Gaussian blur, standard deviation in pixels
    "noise": {1: 0.05, 2: 0.10, 3: 0.20}, # Gaussian noise std, pixel values in [0, 1]
    "jpeg": {1: 30, 2: 15, 3: 5},          # JPEG quality (100 = best, 1 = worst)
}

# All test conditions: clean + 3 corruption types x 3 levels = 10 conditions
CONDITIONS = [("clean", 0)] + [(kind, level) for kind in LEVELS for level in (1, 2, 3)]


def condition_name(kind, level):
    """Short name used in result files, e.g. 'clean', 'blur_1', 'jpeg_3'."""
    return "clean" if kind == "clean" else f"{kind}_{level}"


def gaussian_blur(img, sigma):
    """Blur the image. Removes fine detail such as fur texture and edges."""
    # In Pillow, the GaussianBlur radius is the standard deviation (sigma).
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(img, std, seed):
    """Add random noise to every pixel, like sensor noise in low light."""
    rng = np.random.default_rng(seed)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x + rng.normal(0.0, std, size=x.shape).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)  # keep valid pixel values
    return Image.fromarray((x * 255.0).round().astype(np.uint8))


def jpeg_compress(img, quality):
    """Save and reload as JPEG. Low quality gives blocky 8x8 artefacts."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def corrupt(img, kind, level, image_name=""):
    """Apply one corruption at one severity level to a 224x224 RGB image."""
    if kind == "clean":
        return img
    value = LEVELS[kind][level]
    if kind == "blur":
        return gaussian_blur(img, value)
    if kind == "noise":
        # Seed from the image name, so each image always gets the same noise
        seed = NOISE_SEED + zlib.crc32(image_name.encode()) + level
        return gaussian_noise(img, value, seed)
    if kind == "jpeg":
        return jpeg_compress(img, value)
    raise ValueError(f"Unknown corruption: {kind}")


def resize_and_crop(img, size=BASE_SIZE):
    """Resize the shorter side to 256 and take the central 224x224 crop.

    This keeps the animal's proportions (no squashing) and is the standard
    evaluation preprocessing for ImageNet models such as ResNet-18. It uses the
    same ratio (1.14) as the evaluation transform in train_cnn.py, so the CNN
    sees images framed the same way as during its validation.
    """
    img = TF.resize(img, int(round(size * 256 / 224)),
                    interpolation=TF.InterpolationMode.BILINEAR)
    return TF.center_crop(img, size)


def load_image(path, kind="clean", level=0):
    """Load an image, resize and crop to 224x224, and apply a corruption.

    This is the single entry point all models should use, so that clean and
    corrupted images go through exactly the same steps.
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")
    img = resize_and_crop(img)
    return corrupt(img, kind, level, image_name=path.stem)


def save_examples(image_paths, out_path):
    """Save a grid: one block per image, rows = corruption type, cols = level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = list(LEVELS)
    n_rows = len(image_paths) * len(kinds)
    fig, axes = plt.subplots(n_rows, 4, figsize=(10, 2.6 * n_rows))

    for i, path in enumerate(image_paths):
        for j, kind in enumerate(kinds):
            row = i * len(kinds) + j
            for col, level in enumerate([0, 1, 2, 3]):
                k = "clean" if level == 0 else kind
                img = load_image(path, k, level)
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if level == 0:
                    ax.set_title(f"{Path(path).stem}\nclean", fontsize=8)
                else:
                    ax.set_title(f"{kind}, level {level} ({LEVELS[kind][level]})",
                                 fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--breeds", nargs="+",
                        default=["american_pit_bull_terrier", "staffordshire_bull_terrier"],
                        help="show the first test image of each of these breeds")
    parser.add_argument("--out", default="figures/corruption_examples.png")
    args = parser.parse_args()

    image_dir = Path(args.data_root) / "oxford-iiit-pet" / "images"
    with open(Path(args.splits) / "test.csv") as f:
        rows = list(csv.DictReader(f))

    paths = []
    for breed in args.breeds:
        match = next((r for r in rows if r["breed"] == breed), None)
        if match is None:
            raise ValueError(f"No test image found for breed '{breed}'")
        paths.append(image_dir / f"{match['image']}.jpg")

    save_examples(paths, Path(args.out))
    print(f"Saved examples to {args.out}")
    print("Conditions:", ", ".join(condition_name(k, l) for k, l in CONDITIONS))


if __name__ == "__main__":
    main()