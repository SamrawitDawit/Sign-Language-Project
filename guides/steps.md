# Sign Language Recognition — Complete Project Brief for Claude Code

## How to use this file

This is a single self-contained brief covering all remaining work on the project. Read it top to bottom once before doing anything. Each task below has its own scope, files, steps, and pitfalls. Tasks are ordered so they can be tackled sequentially or in parallel branches.

**Stopping points are marked explicitly.** Respect them. When a step says "stop and report," do that — do not continue to the next step on your own.

---

## Table of contents

1. [Project context](#1-project-context)
2. [Current repo state](#2-current-repo-state)
3. [Final architecture goal](#3-final-architecture-goal)
4. [Universal hard rules](#4-universal-hard-rules)
5. [Execution order](#5-execution-order)
6. [Task 1 — Sentence buffer (`buffer.py`)](#task-1--sentence-buffer-bufferpy)
7. [Task 2 — Replace MLP with GCN as Model B](#task-2--replace-mlp-with-gcn-as-model-b)
8. [Task 3 — Add HOG + SVM as Model A (classical CV)](#task-3--add-hog--svm-as-model-a-classical-cv)
9. [Task 4 — Add MobileNetV2 CNN as Model C](#task-4--add-mobilenetv2-cnn-as-model-c)
10. [Task 5 — Comparison script and integration](#task-5--comparison-script-and-integration)
11. [Pitfalls summary](#11-pitfalls-summary)
12. [Suggested first prompts for Claude Code](#suggested-first-prompts-for-claude-code)

---

## 1. Project context

- **Course project:** Computer Vision practical application. 5-person team. Due Monday.
- **Application:** ASL alphabet recognition from webcam, with letter buffering into displayed words/sentences.
- **What the brief asks for:** "A good body of knowledge in Computer Vision." Course explicitly lists Classical CV, CNNs, and Transformer-based architectures as valid paradigms.
- **NOT in scope:** Continuous sign language translation, multi-sign sentence parsing, signer-independent generalization. The "sentence" output is letter-buffering, not true translation. Be honest about this in the report.

---

## 2. Current repo state

GitHub: `https://github.com/SamrawitDawit/Sign-Language-Project`

**Already done:**
- `asl_landmark_pipeline.ipynb` — MediaPipe extraction notebook (M1)
- `landmarks.npy` — feature matrix (M1)
- `labels.npy` — integer labels (M1)
- `splits.npz` — train/val/test indices (M1)
- `label_to_index.json` — class name → integer map (M1)
- `landmark_metadata.csv` — per-sample metadata including image paths (M1)
- `hand_landmarker.task` — MediaPipe model file (M1)
- `failed_images.txt` — samples where extraction failed (M1)
- `train.py` and `infer.py` — MLP training and inference (M2, working, ~95–99% test accuracy)
- `model.pt` — trained MLP checkpoint (M2)
- `results.json` — MLP metrics (M2)

**Still to do (this brief covers all of these):**
- `buffer.py` — sentence buffering logic (Task 1)
- Replace MLP with GCN as primary Model B (Task 2)
- `train_hog_svm.py`, `infer_hog_svm.py` — classical CV pipeline (Task 3)
- `train_cnn.py`, `infer_cnn.py` — MobileNetV2 transfer learning (Task 4)
- `compare_models.py` — three-paradigm results table (Task 5)
- `demo.py` — webcam loop (M3, may or may not be done)
- Report and README (M5)

---

## 3. Final architecture goal

A three-paradigm comparison representing distinct families of computer vision:

| # | Paradigm | Model | Files |
|---|---|---|---|
| A | Classical CV | HOG features + Linear SVM | `train_hog_svm.py`, `infer_hog_svm.py` |
| B | Graph deep learning on hand skeleton | MediaPipe landmarks + GCN | `train.py`, `infer.py`, `gcn_model.py` |
| C | Convolutional deep learning on raw pixels | MobileNetV2 (transfer learning) | `train_cnn.py`, `infer_cnn.py` |

The MLP is preserved as `train_mlp.py` / `infer_mlp.py` (a backup baseline, mentioned briefly in the report).

The live demo uses Model B (GCN) via `from infer import predict`. The other models are for the report's comparison table only.

---

## 4. Universal hard rules

These apply to every task below.

1. **No PyTorch Geometric, no PyG, no DGL.** From-scratch GCN only. CUDA mismatches with PyG can eat 2 hours.
2. **No RBF SVM.** Use `LinearSVC`. RBF on 80k samples takes hours.
3. **Same train/val/test split** (`splits.npz`) across every model. Apples-to-apples.
4. **Same preprocessing flag** (whatever `APPLY_NORM` value the MLP was trained with) for any landmark-based model. Verify before training the GCN.
5. **Preserve existing working files** by renaming, not deleting. The MLP code becomes `train_mlp.py` / `infer_mlp.py`.
6. **All result files share the same JSON structure** so they can merge into one comparison table.
7. **Don't include classical CV or CNN in the live demo.** Only Model B (GCN) is used at inference time. The others exist for the report.
8. **When a step says "stop and report," stop.** Do not continue autonomously past a checkpoint.

---

## 5. Execution order

Recommended order, with rough time estimates:

1. **Task 1 — Buffer** (~30 min). Pure logic, no dependencies, can be done in parallel with anything else. Do this first.
2. **Task 2 — GCN replaces MLP** (~1 hour). Touches existing files. Do this before adding new models so the codebase is stable.
3. **Task 3 — HOG + SVM** (~1 hour, mostly waiting for feature extraction). Kick off and do other work while it runs.
4. **Task 4 — CNN** (~1.5 hours including training). Run alongside Task 3 if you have the compute.
5. **Task 5 — Comparison + integration** (~30 min). Pulls everything together.

If you're truly time-constrained, drop Task 4 (CNN) and ship with two paradigms (Classical CV + Graph DL). That's still a defensible project.

---

## Task 1 — Sentence buffer (`buffer.py`)

**Goal:** Take per-frame letter predictions and turn them into displayed words and sentences via temporal logic.

### Interaction grammar (lock these rules upfront, document in module docstring)

- **Commit a letter:** same letter at confidence ≥ threshold for N consecutive frames (default 10 ≈ 0.33 sec at 30fps). Letter is added to the current word.
- **Space (end word):** no hand detected for M consecutive frames (default 15 ≈ 0.5 sec). Inserts a space and ends the current word.
- **Sentence end (optional):** sustained no-hand for K consecutive frames (default 60 ≈ 2 sec). Commits sentence, resets.
- **Anti-repeat:** after a commit, the same letter must re-trigger (release + re-sign) before it can commit again. Prevents "HELLOOO" from one held gesture.

### Files to create

- `buffer.py` — the `SentenceBuffer` class
- `test_buffer.py` — mocked-input tests, runnable without webcam or model

### API (M3 will consume this — do not deviate)

```python
class SentenceBuffer:
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        commit_frames: int = 10,
        space_frames: int = 15,
        sentence_end_frames: int = 60,
    ): ...

    def update(self, letter: str | None, confidence: float = 0.0) -> None:
        """Call once per frame. `letter=None` means no hand detected."""

    def get_state(self) -> dict:
        """
        Returns:
        {
            "pending_letter": "H",       # currently held but not yet committed
            "pending_progress": 0.6,     # 0.0 to 1.0, for progress bar
            "current_word": "HEL",
            "sentence": "HELLO WORLD",
            "is_no_hand": False,
        }
        """

    def reset(self) -> None: ...
```

### Steps

1. Implement `SentenceBuffer` per the API above.
2. Write `test_buffer.py` with mocked sequences:
   - Spell "HI" → check `current_word == "HI"`
   - Spell "HI WORLD" with pause → check `sentence` contains the space
   - Held letter for 30 frames → check word is "H", not "HHH" (anti-repeat works)
3. Run `python test_buffer.py`. Fix the buffer logic — never the tests — until all pass.

### Pitfalls

- Don't auto-correct or spellcheck. Keep output raw.
- Confidence below threshold should be treated as no prediction — don't increment the same-letter counter.
- Initial state when first calls are `None` should not trigger anything. Empty state by default.
- All frame counts must be constructor params so M3 can tune at integration time without editing logic.

### Done when

`test_buffer.py` passes for all sequences. M3 can `from buffer import SentenceBuffer` and use it in the webcam loop with no further changes.

---

## Task 2 — Replace MLP with GCN as Model B

**Goal:** Replace the landmark + MLP pipeline with a from-scratch 2-layer Graph Convolutional Network that treats the 21 MediaPipe landmarks as nodes in a graph whose edges follow the hand's skeletal topology.

**Why replace:** The MLP isn't really "CV" — it's a dense network on tabular data. GCN demonstrates graph neural networks and geometric structure, which strengthens the CV-knowledge story.

**Safety net:** The MLP code is renamed, not deleted. If the GCN underperforms by >3 points, fall back to the MLP by swapping one import.

### Files to create / modify

- Rename: `train.py` → `train_mlp.py`, `infer.py` → `infer_mlp.py`
- Move/rename: `results.json` → `results_mlp.json`
- Create: `gcn_model.py` (the `HandGCN` class + adjacency builder)
- Create: `train.py` (GCN training, replaces old MLP version)
- Create: `infer.py` (GCN inference, preserves `predict()` signature)
- New artifacts: `model_gcn.pt`, `results_gcn.json`

### Hand topology

MediaPipe Hands has a fixed 21-node skeleton. Encode once:

```python
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),                       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                       # index
    (5, 9), (9, 10), (10, 11), (11, 12),                  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),                # ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),      # pinky
]
```

### Normalized adjacency

Build once at model init, register as a buffer so it follows `.to(device)`:

```python
def build_normalized_adjacency(num_nodes: int = 21) -> torch.Tensor:
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in HAND_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += torch.eye(num_nodes)                       # self-loops
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D                                # (21, 21)
```

### Model

2-layer GCN with mean pooling. ~10–20K params, trains in 2–3 min on CPU.

```python
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, A_norm):
        return A_norm @ self.linear(x)              # propagate then transform


class HandGCN(nn.Module):
    def __init__(self, num_classes, h1=32, h2=64, dropout=0.3):
        super().__init__()
        self.register_buffer("A_norm", build_normalized_adjacency(21))
        self.gcn1 = GCNLayer(3, h1)
        self.gcn2 = GCNLayer(h1, h2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h2, num_classes)

    def forward(self, x):                           # x: (B, 21, 3)
        h = torch.relu(self.gcn1(x, self.A_norm))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(h, self.A_norm))
        h = h.mean(dim=1)                           # mean pool: (B, 21, h2) → (B, h2)
        h = self.dropout(h)
        return self.classifier(h)
```

### Steps

1. **STOP. Report current shape of `landmarks.npy` and the `APPLY_NORM` value used by the MLP. Do not proceed until user confirms.**
2. `git mv train.py train_mlp.py`, `git mv infer.py infer_mlp.py`, `mv results.json results_mlp.json`. Update any internal imports inside the renamed files.
3. Create `gcn_model.py` with `HandGCN`, `GCNLayer`, `build_normalized_adjacency`, `HAND_EDGES`. Add a `__main__` block that runs a forward pass on a `(2, 21, 3)` dummy tensor and prints output shape. Run it to sanity-check.
4. Create new `train.py`: loads data, reshapes to `(N, 21, 3)`, applies same `APPLY_NORM` as MLP, trains `HandGCN` with Adam(lr=1e-3), batch=128, 50 epochs, dropout=0.3. Saves `model_gcn.pt` and `results_gcn.json`.
5. Create new `infer.py`: same `predict(landmarks) → (letter, confidence)` signature. Reshape input to `(1, 21, 3)`, apply same normalization, return prediction.
6. Sanity-test: load a few test-split samples, run through new `infer.predict()`, verify predictions are consistent with training-time evaluation.
7. **STOP and report results.** Specifically: GCN test accuracy vs MLP test accuracy. If GCN underperforms by >3 points, do not silently swap — flag it.
8. Run `demo.py` if it exists. Verify it works without modification (import contract preserved).

### Pitfalls

- **Adjacency on wrong device.** Use `register_buffer`, not a plain attribute, so `.to(device)` moves it.
- **Wrong input shape.** Reshape `(N, 63)` → `(N, 21, 3)` before training and before each inference call. Get this wrong → silent accuracy collapse.
- **Normalization flag drift.** `train.py` and `infer.py` must match the MLP's `APPLY_NORM` exactly. Re-verify after renaming.
- **Don't add fancy GCN tricks** (residual connections, attention, deeper layers). Plain 2-layer is enough and the literature paper using "successive residual GCN" gets SOTA gains that don't matter for this timeline.
- **Don't train for 200 epochs.** 50 is plenty.
- **If GCN underperforms**, fall back to MLP by swapping one import in `demo.py`: `from infer_mlp import predict`. Don't waste time tuning.

### Done when

- `model_gcn.pt`, `results_gcn.json`, `gcn_model.py`, new `train.py`, new `infer.py` exist
- GCN test accuracy ≥ 95% (ideally ≥ MLP's number)
- `demo.py` runs unchanged

---

## Task 3 — Add HOG + SVM as Model A (classical CV)

**Goal:** A purely classical computer vision baseline using Histogram of Oriented Gradients features fed to a Linear SVM. Demonstrates classical CV knowledge for the report.

### Files to create

- `train_hog_svm.py` — extract HOG features, train Linear SVM, save model + metrics
- `infer_hog_svm.py` — exposes `predict_hog_svm(image: np.ndarray) → (letter, confidence)`
- New artifacts: `hog_features.npy`, `model_hog_svm.joblib`, `results_hog_svm.json`

### Approach

- **Feature extraction:** `skimage.feature.hog` with `orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'`. Convert to grayscale, resize to **64×64** (NOT 224×224 — 3× slower extraction with no accuracy gain on ASL).
- **Classifier:** `Pipeline([StandardScaler(), LinearSVC(C=1.0, max_iter=5000)])`. Optionally wrap in `CalibratedClassifierCV(cv=3)` for probability outputs (needed for confidence scores).
- **Expected accuracy:** 85–92% on the test split. Will lose to GCN and CNN — that's the point. Demonstrates the value-add of modern approaches.
- **Training time:** 5–15 min for HOG extraction (the bottleneck). SVM fit itself is <1 min.

### Steps

1. **Verify image dataset access.** Read `landmark_metadata.csv`, locate the column with image paths, verify a few exist on disk. **Stop and ask the user if images aren't accessible.**
2. Estimate HOG extraction time for the full training set at 64×64. Report the number. **Stop and wait for user approval before running extraction.**
3. Write extraction loop using `tqdm` for progress. Save to `hog_features.npy` so re-runs don't recompute.
4. Train: same `splits.npz` indices. Fit `Pipeline(StandardScaler + LinearSVC)` on train. Optionally wrap with `CalibratedClassifierCV` for probabilities.
5. Evaluate on test split: accuracy, classification report, confusion matrix. Save `results_hog_svm.json` matching the schema of the other result files.
6. Save model: `joblib.dump(pipeline, "model_hog_svm.joblib")`.
7. Write `infer_hog_svm.py` with `predict_hog_svm()`: same image → grayscale → resize → HOG → scaler → SVM → return `(letter, confidence)`.

### Pitfalls

- **Don't use `SVC` with default RBF kernel.** Use `LinearSVC`. RBF on 80k samples = hours.
- **HOG at 224×224 is overkill and slow.** 64×64 is fine.
- **Don't recompute HOG features on every script run.** Cache to `hog_features.npy` and load if present.
- **`LinearSVC` has no `predict_proba`.** Either wrap in `CalibratedClassifierCV` or use `decision_function()` output as an unnormalized confidence (and document this honestly in the inference module).
- **Image paths in `landmark_metadata.csv` may be absolute paths from M1's machine.** Verify they work on the current machine; rewrite to relative paths if needed.
- **Do not include this model in the webcam demo.** Report comparison only.

### Done when

- `python train_hog_svm.py` produces `model_hog_svm.joblib`, `hog_features.npy`, `results_hog_svm.json`
- HOG + SVM test accuracy reported
- `infer_hog_svm.py` exposes a working `predict_hog_svm()` function

---

## Task 4 — Add MobileNetV2 CNN as Model C

**Goal:** Transfer learning with a pretrained MobileNetV2 on raw ASL alphabet images. Demonstrates convolutional deep learning for the report.

### Files to create

- `train_cnn.py` — CNN training script
- `infer_cnn.py` — exposes `predict_cnn(image: np.ndarray) → (letter, confidence)`
- New artifacts: `model_cnn.pt`, `results_cnn.json`

### Approach

- **Model:** `torchvision.models.mobilenet_v2(weights="DEFAULT")`. Replace `model.classifier[1]` with `nn.Linear(1280, num_classes)`. Full fine-tuning at low LR (no layer freezing).
- **Input pipeline:** Load images from paths in `landmark_metadata.csv`. Resize to 224×224, normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Training augmentation:** Random horizontal flip, small rotation (±15°), color jitter. Train split only.
- **Hyperparameters:** Adam(lr=1e-4), CrossEntropyLoss, batch=32, **5–10 epochs only** (transfer learning converges fast).
- **Expected accuracy:** 96–99% on test split. May match or slightly exceed GCN.
- **Training time:** ~10–30 min on CPU. Faster on Colab T4.

### Steps

1. **Verify image dataset access** (same check as Task 3). Stop and ask if not accessible.
2. Build a custom `torch.utils.data.Dataset` that loads images from the metadata paths and applies the splits from `splits.npz`.
3. Apply ImageNet normalization. Apply training augmentations to the train split only.
4. Load pretrained MobileNetV2, replace classifier head.
5. Train 5–10 epochs. Save best checkpoint by val accuracy as `model_cnn.pt`.
6. Evaluate on test split, save `results_cnn.json`.
7. Write `infer_cnn.py` with `predict_cnn()` applying the same resize + normalization pipeline.

### Pitfalls

- **Wrong normalization stats** → silent accuracy collapse. Use exact ImageNet values.
- **OpenCV gives BGR, MobileNetV2 expects RGB.** Convert explicitly: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.
- **Don't train for 50 epochs.** Transfer learning at lr=1e-4 converges in 5–10.
- **If CPU training is painfully slow**, drop input size to 128×128 or 96×96. MobileNetV2 still works.
- **Don't include this model in the webcam demo.** Report comparison only.
- **Different filename** for the checkpoint: `model_cnn.pt`, not `model.pt`.

### Done when

- `python train_cnn.py` produces `model_cnn.pt` and `results_cnn.json`
- CNN test accuracy reported
- `infer_cnn.py` exposes a working `predict_cnn()` function

---

## Task 5 — Comparison script and integration

**Goal:** A single script that produces the three-paradigm comparison table for the report.

### Files to create

- `compare_models.py` — loads all result JSON files, prints and optionally saves a side-by-side comparison

### Output format

```
Paradigm                        Model               Test Acc   Params/Features   Train Time   Inference (ms/frame)
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Classical CV                    HOG + LinearSVM     88.4%      ~1764 features    ~12 min      ~5
Graph deep learning             MediaPipe + GCN     96.8%      ~15K params       ~3 min       ~3
Convolutional deep learning     MobileNetV2         98.2%      ~3.5M params      ~22 min      ~30
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
(Baseline reference)            MediaPipe + MLP     96.2%      ~10K params       ~2 min       ~2
```

Numbers above are illustrative — fill from actual `results_*.json` files.

### Steps

1. Load `results_hog_svm.json`, `results_gcn.json`, `results_cnn.json`, `results_mlp.json`.
2. Extract test accuracy from each and any config metadata.
3. Measure inference latency: call each model's `predict*()` 100 times on a held-out sample, compute mean ms/frame.
4. Print a formatted table to stdout. Also save as `comparison.md` for direct inclusion in the report.
5. Generate a confusion matrix figure for each model (matplotlib heatmap) and save as PNG. These are figures for the report.

### Done when

- Running `python compare_models.py` prints the three-paradigm comparison table
- `comparison.md` is saved for the report
- Confusion matrix PNGs exist for at least Models A, B, and C

---

## 11. Pitfalls summary

Combined, in priority order:

1. **Preprocessing mismatch between train and inference** is the single most common silent failure across all model types. Same normalization, same input shape, same color space, same image size. Always.
2. **Don't install PyTorch Geometric.** From-scratch GCN works fine for a 21-node graph.
3. **Don't use RBF SVM.** `LinearSVC` only.
4. **Don't include classical CV or CNN models in the live webcam demo.** They exist for the report's comparison table. Live demo uses only Model B (GCN), via the new `infer.py`.
5. **Image dataset path issues** can derail Tasks 3 and 4 — always verify access first, before writing training code.
6. **`landmark_metadata.csv` paths may be absolute from M1's machine.** Check and rewrite if needed.
7. **Same train/val/test split across every model.** Use `splits.npz` consistently.
8. **Don't chase hyperparameters past the targets.** GCN ≥ 95%, CNN ≥ 96%, HOG+SVM ≥ 85%. Stop there.
9. **Renaming files (Task 2) is destructive.** `demo.py` is briefly broken between renaming `infer.py` and creating the new GCN one. Stopping point at Task 2 step 1 is critical.
10. **If GCN underperforms by >3 points vs MLP**, fall back to MLP for the demo by swapping one import. Don't waste an hour tuning.

---

## Suggested first prompts for Claude Code

You can run these tasks in one long session or in separate sessions. Here are entry points for each.

### Starting Task 1 (Buffer)
> Read `CLAUDE.md` sections 1–5 and Task 1. Then write `buffer.py` and `test_buffer.py` per the spec. Run the tests and report results. Do not modify any other files.

### Starting Task 2 (GCN replaces MLP)
> Read `CLAUDE.md` sections 1–5 and Task 2. Then:
> 1. Report the current shape of `landmarks.npy`.
> 2. Report the `APPLY_NORM` value used in the existing `train.py`.
> 3. STOP. Do not rename files or write any code. I will confirm before you proceed.

### Starting Task 3 (HOG + SVM)
> Read `CLAUDE.md` sections 1–5 and Task 3. Then:
> 1. Locate the image path column in `landmark_metadata.csv` and verify a few paths exist on disk.
> 2. Estimate HOG extraction time for the full training set at 64×64.
> 3. STOP and report. Do not start extraction until I approve.

### Starting Task 4 (CNN)
> Read `CLAUDE.md` sections 1–5 and Task 4. Then:
> 1. Confirm image dataset is accessible (skip if already verified in Task 3).
> 2. Report dataset structure: number of images, folder layout, label names.
> 3. Propose a minimal `train_cnn.py` plan, including hyperparameters and augmentations. Do not write code yet.

### Starting Task 5 (Comparison)
> Read `CLAUDE.md` sections 1–5 and Task 5. Write `compare_models.py` per the spec. It should load all available `results_*.json` files in the repo, print a comparison table to stdout, and save `comparison.md`. Generate confusion matrix PNGs for each model.
