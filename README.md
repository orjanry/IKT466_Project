# IKT466 Final Project: Robustness of image classifiers to degraded images

Track A (computer vision), Oxford-IIIT Pets, University of Agder.

**Question.** How does classification accuracy on Oxford-IIIT Pets degrade under
blur, noise and JPEG compression for a linear pixel baseline, a CNN trained from
scratch, and an MLP head on frozen pretrained features, relative to each model's
clean accuracy?

## Models

| Model | Script | What is trained |
|---|---|---|
| Baseline: 32×32 pixels → standardise → PCA → logistic regression | `src/baseline.py` | PCA and logistic regression, hyperparameters chosen on validation |
| CNN from scratch: 4 conv blocks, ~1.18M parameters, 128×128 input | `src/train_cnn.py` | All weights, 80 epochs, 3 seeds |
| ResNet-18 (ImageNet, frozen) + MLP head | `src/pretrained.py` | Only the MLP head, 3 seeds |

Every model is tested on 10 conditions: clean, plus Gaussian blur, Gaussian noise
and JPEG compression at 3 severity levels each (`src/corruptions.py`).

## Main results

- Combined table: `results/summary_all.csv` (mean and standard deviation over seeds)
- Main figure: `figures/degradation.png`
- Error analysis: `results/error_analysis/` and `figures/error_examples.png`

Clean test accuracy: baseline 9.3 %, CNN 58.5 %, ResNet-18 + MLP head 88.4 %
(chance is 2.7 %).

## Setup

Tested with Python 3.13 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

NVIDIA GPU users: reinstall torch with CUDA support afterwards.

```powershell
pip uninstall -y torch torchvision
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
```

## Data

The dataset is downloaded automatically (about 800 MB) into `data/`, which is
ignored by git:

```powershell
python src/make_splits.py
```

The official test split (3,669 images) is kept unchanged. The official trainval
split is divided into train (2,944) and validation (736) with a stratified 80/20
split, seed 42. The split definitions are committed in `splits/`.

## Reproducing the results

Run from the repository root, in this order. Times are from our runs on an
NVIDIA RTX 3060 Laptop GPU (CNN) and an RTX 3050 6GB Laptop GPU (pretrained).

```powershell
# 1. Baseline, trained and evaluated on all 10 conditions
python src/baseline.py

# 2. CNN, 80 epochs, three seeds (about 11 min each)
python src/train_cnn.py --seed 42 --epochs 80 --run-name e80_seed42
python src/train_cnn.py --seed 43 --epochs 80 --run-name e80_seed43
python src/train_cnn.py --seed 44 --epochs 80 --run-name e80_seed44

# 3. CNN on all 10 conditions (about 5 min per seed)
python src/evaluate_cnn.py --run-names e80_seed42 e80_seed43 e80_seed44

# 4. Pretrained ResNet-18 + MLP head, all seeds and conditions
#    (feature extraction about 6 min the first time, then cached in data/features/)
python src/pretrained.py

# 5. Combined table, main figure and error analysis
python src/summarize.py
python src/error_analysis.py

# Optional: example images of every corruption level
python src/corruptions.py
```

The choice of 80 epochs was made on validation accuracy. The 40-epoch run used
for that decision is kept in `results/cnn/seed42/`, and the history of the first
80-epoch run is in `results/cnn/epoch_check/`.

## Notes

- **Preprocessing.** All test images go through `corruptions.load_image()`:
  resize the short side to 256, centre crop 224×224, apply the corruption, then
  resize to the model's input size. Corruptions are therefore applied at the same
  resolution for every model. For this reason the CNN's clean accuracy in
  `results/cnn/summary.json` (58.5 %) differs slightly from the training script's
  own test evaluation (60.0 %), which resizes directly to 128×128.
- **Seeds and statistics.** CNN and pretrained results use seeds 42, 43 and 44.
  `summarize.py` reports the sample standard deviation (n − 1). The per-model
  `summary.json` files use the population standard deviation (n).
- **Checkpoints** are not stored in git (see `.gitignore`), except files placed
  in `checkpoints/final/`. Retrain with the commands above to recreate them.
- **Windows.** If a run fails with `WinError 1455` (page file too small), close
  other programs, run with `--workers 2`, or increase the Windows page file.
- **Pretrained weights.** torchvision `ResNet18_Weights.IMAGENET1K_V1`. They are
  downloaded to the torch cache on first use (set `TORCH_HOME` to change where).

## References

- O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, "Cats and Dogs",
  CVPR 2012 (Oxford-IIIT Pet dataset).
- K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image
  Recognition", CVPR 2016 (ResNet).
