# ASL Alphabet Recognition

Real-time American Sign Language (ASL) alphabet recognition from webcam using MediaPipe hand landmarks and a trained MLP classifier. Letters are buffered into words and sentences as you sign.

**Scope:** Static letter recognition (A–Z + del + space). J and Z are motion letters in real ASL — see [Known Limitations](#known-limitations).

---

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install it once with:
```bash
pip install uv
```

Then install all dependencies into an isolated virtual environment:
```bash
uv sync
```

All other files (`models/model.pt`, `models/hand_landmarker.task`, `data/`) are already in the repo — no downloads needed.

---

## Usage

Prefix every command with `uv run` so it uses the project's virtual environment.

### Live demo
```bash
uv run python demo.py
```
Opens a webcam window. Hold a hand sign steady for ~0.5 s to commit a letter. Remove your hand for ~0.7 s to insert a word space. Sign `del` to backspace. Press `q` to quit.

### Train from scratch
```bash
uv run python train.py
```
Trains MLP + Random Forest on `data/splits.npz`. Saves `models/model.pt` and `results.json`. Takes ~2 min on CPU.

### Evaluation notebook
```bash
uv run jupyter notebook evaluation.ipynb
```
Generates confusion matrix, per-class accuracy chart, and model comparison table. Plots are saved to `docs/`.

---

## Repo structure

```
data/                    raw data artifacts (landmarks, splits, label map)
models/                  trained model checkpoint + MediaPipe task file
notebooks/               data pipeline notebook (run once by M1)
docs/                    findings, plots, and notes for the report
preprocessing.py         normalize_landmarks() — single source of truth
train.py                 MLP + RF training script
infer.py                 predict(landmarks) -> (letter, confidence)
buffer.py                SentenceBuffer — letter-to-sentence buffering
demo.py                  live webcam demo
evaluation.ipynb         confusion matrix, per-class accuracy, error analysis
results.json             training metrics (MLP 98.45%, RF 98.80%)
pyproject.toml           uv project manifest and dependency list
uv.lock                  fully pinned lockfile (auto-generated, do not edit)
```

See [docs/structure.md](docs/structure.md) for a full annotated breakdown.

---

## Results

| Model | Test Accuracy | Train Time |
|---|---|---|
| MLP (63→128→64→28) | 98.45% | 45.7 s |
| Random Forest (200 trees) | 98.80% | 37.0 s |

Full per-class breakdown and classification report in [docs/model_results.md](docs/model_results.md).

---

## Known Limitations

- **J and Z** are motion letters in ASL. This model classifies static hand poses, so webcam accuracy for those two letters may differ from the held-out test numbers.
- **Right-hand only.** Left-hand landmarks are not mirrored; left-hand signers will see reduced accuracy.
- **Studio dataset.** Trained on controlled-lighting Kaggle images. Generalization to different lighting or skin tones is untested.

See [docs/model_results.md](docs/model_results.md) for the full limitations section.
