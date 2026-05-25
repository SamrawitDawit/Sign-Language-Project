# ASL Alphabet Recognition — Project Report

## 1. Overview

This project implements a real-time American Sign Language (ASL) alphabet recognition system capable of classifying 28 static hand signs (A–Z + `del` + `space`) from webcam video. The system is built around three core components:

- **Data pipeline**: MediaPipe Hand Landmarker extracts 21 hand landmarks from the `grassknoted/asl-alphabet` Kaggle dataset (~87K images), which are normalized and split into train/val/test sets.
- **Classification models**: Three paradigms are compared — a classic MLP, a Random Forest ensemble, and a Graph Convolutional Network (HandGCN) operating on the hand skeleton graph.
- **Interactive applications**: A command-line webcam demo (`demo.py`) buffers letters into sentences, and a browser-based ASL Teacher web app (`asl-teacher/`) provides structured practice with real-time feedback, session stats, and adaptive retraining.

**Key results**:
| Model | Test Accuracy | Train Time |
|---|---|---|
| MLP (63→128→64→28) | 98.45% | 45.7 s |
| Random Forest (200 trees) | 98.80% | 37.0 s |
| HandGCN (4→32→64→28) | 85.31% | 104.6 s |

The MLP model is used in the live applications due to its strong accuracy and fast inference.

---

## 2. Dataset

### 2.1 Source

- **Dataset**: `grassknoted/asl-alphabet` (Kaggle)
- **Original size**: 87,000 images across 29 classes (A–Z, `del`, `space`, `nothing`)
- **Resolution**: 200×200 pixels, RGB, studio lighting conditions
- **Class distribution**: 3,000 images per class

### 2.2 Preprocessing

Each image is processed through the MediaPipe Hand Landmarker to extract 21 hand landmarks (x, y, z coordinates). Images where no hand is detected (26.9%) are excluded.

Two normalization steps are applied to each sample:

**Norm_Loc** (wrist-centering):
```
coords = coords - coords[0]   # subtract wrist landmark (index 0)
```

**Norm_Zoom** (scale invariance):
```
scale = max(||coords[:, :2]||)    # max L2 norm of (x, y) across landmarks
if scale > 0: coords /= scale
```

This produces a 63-dimensional feature vector (21 × 3) per sample, invariant to hand position and distance from camera.

### 2.3 Final Dataset Statistics

| Metric | Value |
|---|---|
| Successful extractions | 63,580 (73.1%) |
| Classes | 28 (A–Z + `del` + `space`, excluding `nothing`) |
| Training set | 44,505 (70%) |
| Validation set | 9,537 (15%) |
| Test set | 9,537 (15%) |
| Feature dimension | 63 (21 × 3) |

The stratified split uses a random seed of 42. All data artifacts are stored in `data/splits.npz`.

---

## 3. Model Architectures

### 3.1 MLP (Multilayer Perceptron)

The baseline model is a 3-layer fully connected network:

```
Input (63) → Linear(63, 128) → ReLU → Dropout(0.3)
          → Linear(128, 64)  → ReLU → Dropout(0.3)
          → Linear(64, 28)   → Logits
```

- **Parameters**: ~10K
- **Training**: Adam (lr=1e-3), 50 epochs, batch_size=128
- **Loss**: CrossEntropyLoss

### 3.2 Random Forest

A classical ensemble baseline using 200 decision trees:

- **Architecture**: `RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)`
- **Input**: Same 63-dim normalized landmark vector
- **Training**: Scikit-learn implementation, default parameters

### 3.3 HandGCN (Graph Convolutional Network)

A 2-layer GCN that operates on the hand skeleton graph topology:

**Adjacency matrix**: Symmetric normalized `D^{-1/2}(A+I)D^{-1/2}` over 22 edges (the MediaPipe hand landmark connections). Self-loops are added.

**Node features**: Each of the 21 nodes receives 4 features — (x, y, z) normalized coordinates plus the joint angle at that landmark (computed from the angle between the two incident edges). Landmarks at fingertips and wrist (which lack a defined angle) receive a value of 0.

```
Input (21, 4)
  → GCNLayer(4→32, LeakyReLU(0.2), Dropout(0.3))
  → GCNLayer(32→64, LeakyReLU(0.2), Dropout(0.3))
  → Mean-pool over 21 nodes
  → Linear(64, 28)
  → Logits
```

- **Parameters**: ~4K
- **Training**: Adam (lr=3e-4, weight_decay=1e-4), 50 epochs, batch_size=64
- **Loss**: CrossEntropyLoss

Compared to the MLP, the GCN underperforms by 13.14 percentage points. This is attributable to its smaller capacity (4K vs 10K parameters) and the absence of residual connections. Reference implementations (Sarkar et al., 2024) achieve 99.14% with a 142K-parameter GCN using residual skip connections.

---

## 4. Results

### 4.1 Overall Performance

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Train Time |
|---|---|---|---|---|---|
| MLP | 98.45% | 0.9826 | 0.9827 | 0.9826 | 45.7 s |
| Random Forest | 98.80% | — | — | — | 37.0 s |
| HandGCN | 85.31% | — | — | — | 104.6 s |

### 4.2 Per-Class Accuracy (MLP)

| Letter | Accuracy | Letter | Accuracy |
|---|---|---|---|
| A | 99.08% | N | **90.58%** |
| B | 100.00% | O | 97.64% |
| C | 100.00% | P | 98.69% |
| D | 98.91% | Q | 99.06% |
| E | 98.84% | R | 97.64% |
| F | 99.07% | S | 98.17% |
| G | 99.73% | T | 98.02% |
| H | 98.31% | U | 98.40% |
| I | 97.76% | V | 98.69% |
| J | 99.22% | W | 98.64% |
| K | 99.01% | X | **95.99%** |
| L | 99.21% | Y | 99.22% |
| M | **93.33%** | Z | 100.00% |
| | | del | 100.00% |
| | | space | 98.25% |

### 4.3 Error Analysis

The weakest classes are **M (93.33%)** and **N (90.58%)** — both involve fingers curled over the thumb with a subtle difference (M uses three fingers, N uses two). This is a well-known confusion in static ASL handshape recognition.

**X (95.99%)** is also below average; its hooked index finger can resemble relaxed B or partially extended D depending on hand orientation.

**J and Z** score 99%+ on the test set but are motion letters in real ASL — the model learns the static handshape rather than the trajectory, so webcam accuracy for real signers performing the full motion may be lower.

A confusion matrix (see `docs/confusion_matrix.png`) shows that off-diagonal mass is concentrated in a small number of visually similar pairs rather than spread uniformly.

---

## 5. Applications

### 5.1 Webcam Demo (`demo.py`)

A real-time OpenCV application that:
1. Streams webcam video and runs MediaPipe Hand Landmarker per frame
2. Passes normalized landmarks to the MLP for inference
3. Uses `SentenceBuffer` to convert per-frame predictions into a running sentence
4. Renders hand landmarks, pending letter, commit progress bar, and sentence on screen

**Buffer grammar**:
- Same letter held for ~0.5s (10 frames) → committed to current word
- `del` held → backspace last character
- `space` held → insert word space
- No hand for ~1s → insert word space
- No hand for ~4s → reset entire sentence

### 5.2 ASL Teacher (`asl-teacher/`)

A browser-based interactive tutor built with FastAPI + vanilla JavaScript:

- **Architecture**: WebSocket stream from browser camera → server-side MediaPipe extraction → MLP inference → JSON response with landmarks and classification
- **Teaching flow**: 24 letters (A–Y, excluding motion letters J/Z), one at a time, with reference image display and real-time feedback
- **Progress tracking**: Per-letter commit progress bar, overall session progress bar, live attempts/wrong counters
- **Timer mode** (toggleable): 10-second countdown per letter; auto-advances on timeout
- **Session summary**: End-of-session modal with color-coded table showing attempts, wrong guesses, and result per letter
- **Weak-letter retraining**: The 3 hardest letters (most wrong guesses) are automatically queued for a second practice round
- **Keyboard shortcuts**: `Space` to skip, `R` to restart, `T` to toggle timer

---

## 6. Known Limitations

1. **Static-frame only**: J and Z are motion-based letters in real ASL. This model classifies static hand poses, so webcam accuracy for those two letters may differ from test-set numbers when signers perform the full motion.

2. **Studio dataset bias**: The Kaggle dataset was collected in controlled lighting with uniform backgrounds. Generalization to varied lighting, skin tones, camera angles, or hand sizes is untested.

3. **No signer independence**: All train/test samples come from the same source distribution. There is no held-out signer to measure cross-subject generalization.

4. **Right-hand only**: Left-hand landmarks are not mirrored. Left-handed signers will see reduced accuracy unless the webcam input is mirrored before inference.

5. **26.9% extraction failure rate**: Nearly a quarter of source images produced no MediaPipe detection. This may bias the dataset toward easier hand configurations.

6. **Model capacity gap**: The GCN (4K params) significantly underperforms the MLP (10K params). A deeper GCN with residual connections would likely close this gap.

---

## 7. Project Structure

```
data/              — precomputed landmarks, splits, label mapping
models/            — trained MLP and GCN checkpoints, MediaPipe model
notebooks/         — data pipeline and Colab training notebooks
asl-teacher/       — interactive browser-based tutor
docs/              — report, results, plots
preprocessing.py   — landmark normalization (single source of truth)
mlp_model.py       — MLP architecture
gcn_model.py       — HandGCN architecture + adjacency matrix
train_mlp.py       — train MLP + Random Forest
train_gcn.py       — train HandGCN
infer_mlp.py       — real-time MLP inference
infer_gcn.py       — real-time GCN inference
buffer.py          — SentenceBuffer for demo sentence assembly
demo.py            — webcam application
evaluation.ipynb   — offline evaluation and plots
```

**Data flow**:
```
Webcam → MediaPipe → normalize_landmarks() → model.predict() → (letter, confidence)
                                                                    ↓
                                                          SentenceBuffer / WebSocket
                                                                    ↓
                                                          Sentence on screen / Browser UI
```

---

## 8. Future Work

- **Motion-based recognition**: Extend the model to handle dynamic signs (J, Z, and full-word gestures) using an LSTM or Transformer over a sequence of landmark frames.
- **Left-hand support**: Mirror left-hand landmarks to match right-hand training distribution.
- **Signer-independent evaluation**: Collect a held-out dataset from different signers to measure true generalization.
- **Model improvements**: Implement residual GCN connections and increase capacity to match the MLP's performance.
- **Mobile deployment**: Convert the MLP to TensorFlow Lite or ONNX for on-device inference.
- **Gamification**: Add scoring, streaks, and achievement badges to the ASL Teacher app.
- **Speech output**: Add text-to-speech for the completed sentence in both demo and teacher modes.

---

*Report generated May 2026. Full source code available at the project repository.*
