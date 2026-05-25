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

### ASL Teacher (interactive web app)
```bash
uv run uvicorn asl-teacher.app:app --host 0.0.0.0 --port 8000
```
Opens a browser-based ASL tutor. Shows a reference image of each letter, streams your webcam, and gives real-time feedback as you sign. Tracks progress across all 24 static letters (A–Y, no J/Z). Press `Space` to skip, `R` to restart.

> Requires `fastapi` and `uvicorn` — already included when you run `uv sync`.

### Train from scratch
```bash
uv run python train_mlp.py    # MLP + Random Forest
uv run python train_gcn.py    # Graph Neural Network
```
Trains on `data/splits.npz`. Saves results to `models/` and `results_*.json`. Takes ~2 min on CPU.

### Evaluation notebook
```bash
uv run jupyter notebook evaluation.ipynb
```
Generates confusion matrix, per-class accuracy chart, and model comparison table. Plots are saved to `docs/`.

---

## Repo structure

```
data/                      raw data artifacts (landmarks, splits, label map)
models/                    trained model checkpoints + MediaPipe task file
notebooks/                 data pipeline + Colab training notebooks
docs/                      findings, plots, and notes for the report
asl-teacher/               interactive ASL tutor (FastAPI + vanilla JS)

preprocessing.py           normalize_landmarks() — single source of truth
gcn_model.py               HandGCN architecture + joint-angle features
mlp_model.py               MLP architecture (extracted from train_mlp.py)
train_gcn.py               GCN training script
train_mlp.py               MLP + Random Forest training script
train.py                   legacy alias → train_gcn.py
infer_mlp.py               predict(landmarks) → (letter, confidence) via MLP
infer_gcn.py               predict(landmarks) → (letter, confidence) via GCN
infer.py                   legacy alias → infer_gcn.py
buffer.py                  SentenceBuffer — letter-to-sentence buffering
demo.py                    live webcam demo
evaluation.ipynb           confusion matrix, per-class accuracy, error analysis
test_buffer.py             unit tests for SentenceBuffer
pyproject.toml             uv project manifest and dependency list
uv.lock                    fully pinned lockfile (auto-generated, do not edit)
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
