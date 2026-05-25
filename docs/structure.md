# Repo Structure

## Annotated file tree

```
Sign-Language-Project/
│
├── data/                             ← data artifacts (committed)
│   ├── landmarks.npy                 (63,580 × 63) float32 — normalized landmark vectors
│   ├── labels.npy                    (63,580,) str — class name per sample
│   ├── splits.npz                    train/val/test data arrays (X_* and y_* keys)
│   ├── label_to_index.json           class name → int (28 classes, 0–27)
│   ├── landmark_metadata.csv         per-sample image path + label (used by image-model notebooks)
│   └── failed_images.txt             images where MediaPipe found no hand (23,420 entries)
│
├── models/                           ← model checkpoints (committed)
│   ├── model.pt                      MLP checkpoint — state_dict + input_dim + num_classes
│   ├── model_gcn.pt                  HandGCN checkpoint — state_dict + num_classes
│   └── hand_landmarker.task          MediaPipe Hand Landmarker float16 model
│
├── notebooks/                        ← training and pipeline notebooks
│   ├── asl_landmark_pipeline.ipynb   data download → MediaPipe extraction → splits (already run)
│   ├── train_image_models.ipynb      Colab: HOG+SVM and MobileNetV2 CNN (run on Colab GPU)
│   └── train_hog_svm.ipynb           Colab: HOG+SVM standalone (shareable with team members)
│
├── teacher/                          ← interactive ASL teacher web app
│   ├── app.py                        FastAPI server — webcam WebSocket + MLP inference
│   └── static/
│       ├── index.html                single-page UI (vanilla JS, no framework)
│       └── asl/                      24 reference images — A.jpg … Y.jpg (no J, no Z)
│
├── docs/                             ← findings and report assets
│   ├── data.md                       dataset facts, preprocessing math, split sizes
│   ├── model_results.md              model comparison table, per-class accuracy, limitations
│   ├── structure.md                  this file — repo layout and design decisions
│   ├── confusion_matrix.png          MLP 28×28 confusion matrix heatmap
│   └── per_class_accuracy.png        MLP per-class accuracy bar chart
│
├── preprocessing.py                  normalize_landmarks() — wrist-center + scale normalize
├── train_mlp.py                      train MLP + Random Forest; saves model.pt + results_mlp.json
├── train.py                          train HandGCN; saves model_gcn.pt + results_gcn.json
├── gcn_model.py                      HandGCN architecture + joint-angle feature engineering
├── infer_mlp.py                      predict(landmarks) → (letter, confidence) via MLP; lazy-loads
├── infer.py                          legacy inference module (superseded by infer_mlp.py)
├── buffer.py                         SentenceBuffer — per-frame predictions → committed sentence
├── demo.py                           webcam loop: MediaPipe → MLP inference → buffer → display
├── test_buffer.py                    unit tests for SentenceBuffer
│
├── evaluation.ipynb                  offline evaluation, plots, comparison table
├── results_mlp.json                  {"mlp": {...}, "rf": {...}} — MLP + RF results
├── results_gcn.json                  {"gcn": {...}} — HandGCN results
├── pyproject.toml                    uv project manifest — Python version + dependencies
├── uv.lock                           fully pinned lockfile (auto-generated; commit, do not edit)
└── README.md                         setup and usage instructions
```

---

## Data flow

### Demo / inference pipeline

```
Webcam frame
    │
    ▼
MediaPipe Hand Landmarker  (models/hand_landmarker.task)
    │  result.hand_landmarks[0]  →  21 NormalizedLandmark objects
    ▼
np.array([[lm.x, lm.y, lm.z] …])  →  shape (21, 3)
    │
    ▼
preprocessing.normalize_landmarks()
    │  1. Norm_Loc:  coords -= coords[0]          (wrist to origin)
    │  2. Norm_Zoom: coords /= max(‖landmark_i[:2]‖)  (scale-invariant)
    │  output: (63,) float32
    ▼
infer_mlp.predict()  →  (letter, confidence)
    │
    ▼
buffer.SentenceBuffer.update()
    │  hold 15 frames  → commit letter
    │  no hand 20 frames → word space
    │  'del' class     → backspace
    ▼
Sentence displayed on screen  (demo.py)
          — or —
WebSocket response to browser  (teacher/app.py)
```

### Image-model training pipeline (Colab)

```
Kaggle ASL Alphabet dataset (~87k images)
    │  notebooks/train_image_models.ipynb  or  train_hog_svm.ipynb
    ▼
landmark_metadata.csv  →  stratified 70/15/15 split (random_state=42)
    │  filters 'nothing' class (not in label_to_index.json)
    │  caps to MAX_SAMPLES_PER_CLASS = 500 for speed
    ▼
┌──────────────────────────────────┬────────────────────────────────────┐
│  HOG + LinearSVM                 │  MobileNetV2 CNN                   │
│  grayscale → 64×64 → HOG(1764d)  │  224×224 RGB → frozen backbone     │
│  StandardScaler + LinearSVC      │  → Linear(1280, 28) head, 5 epochs │
│  CalibratedClassifierCV(cv=3)    │  Adam lr=1e-3                       │
└──────────────────────────────────┴────────────────────────────────────┘
    ▼
results_hog_svm.json  /  results_cnn.json
model_hog_svm.joblib  /  model_cnn.pt
confusion_matrix_hog_svm.png  /  confusion_matrix_cnn.png
```

---

## Module responsibilities

| Module | Responsibility | Consumed by |
|---|---|---|
| `preprocessing.py` | `normalize_landmarks()` — single source of truth for feature normalization | `train_mlp.py`, `infer_mlp.py` |
| `train_mlp.py` | Train MLP + Random Forest on landmark splits; save checkpoint + metrics | run once |
| `train.py` | Train HandGCN on landmark splits; save checkpoint + metrics | run once |
| `gcn_model.py` | `HandGCN` architecture; joint-angle feature engineering; adjacency matrix | `train.py` |
| `infer_mlp.py` | Lazy-load MLP; `predict(landmarks) → (letter, confidence)` | `demo.py`, `teacher/app.py` |
| `buffer.py` | `SentenceBuffer` — temporal smoothing of per-frame predictions into words | `demo.py` |
| `demo.py` | Webcam loop, MediaPipe, overlay rendering, sentence display | end user |
| `teacher/app.py` | FastAPI server: WebSocket frame intake → MediaPipe → MLP → JSON response | browser UI |
| `test_buffer.py` | Unit tests for `SentenceBuffer` commit/space/backspace/anti-repeat logic | CI / manual |

---

## Running the apps

### Command-line demo
```bash
python demo.py
# press q to quit
```

### ASL Teacher web app
```bash
# 1. Put 24 reference images in teacher/static/asl/  (A.jpg … Y.jpg, no J or Z)
# 2. Start the server
python teacher/app.py
# 3. Open http://localhost:8000
```

The teacher app requires `fastapi` and `uvicorn[standard]`:
```bash
pip install fastapi "uvicorn[standard]"
```

---

## Refactoring notes

### File reorganisation
The repo was reorganised after initial development. Original layout had all files at the root.

| File(s) | From | To |
|---|---|---|
| `landmarks.npy`, `labels.npy`, `splits.npz`, `label_to_index.json`, `landmark_metadata.csv`, `failed_images.txt` | root | `data/` |
| `model.pt`, `hand_landmarker.task` | root | `models/` |
| `asl_landmark_pipeline.ipynb` | root | `notebooks/` |

### MLP / GCN split
Training was split into two scripts after GCN was added:

| Script | Trains | Output |
|---|---|---|
| `train_mlp.py` | MLP + Random Forest | `models/model.pt`, `results_mlp.json` |
| `train.py` | HandGCN | `models/model_gcn.pt`, `results_gcn.json` |

`infer_mlp.py` replaced `infer.py` as the live-inference module. `infer.py` is retained for reference but is no longer imported by any active module.

### Package management
Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```
pyproject.toml   declares requires-python and top-level dependencies
uv.lock          fully resolved, reproducible lockfile — commit this file
.venv/           local virtual environment — do not commit
```

```bash
uv sync                   # creates .venv and installs everything
uv run python demo.py     # run inside the venv without activating it
```
