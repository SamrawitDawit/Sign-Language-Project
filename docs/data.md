# Dataset & Preprocessing Notes

## Source

- **Dataset:** `grassknoted/asl-alphabet` (Kaggle)
- **Pipeline notebook:** `notebooks/asl_landmark_pipeline.ipynb`
- **Landmark model:** MediaPipe Hand Landmarker (`models/hand_landmarker.task`, float16)

## Class set

28 classes: letters A–Z (26) + `del` + `space`.  
J and Z are **motion letters** in real ASL (J traces a hook, Z traces a Z). The Kaggle dataset contains static images of those handshapes, so the classifier learns the static start/end pose, not the motion. This is a known limitation.

## Extraction

| | |
|---|---|
| Images attempted | 87,000 |
| Landmarks extracted | 63,580 |
| Failed (no hand detected) | 23,420 (26.9%) |

Failures are logged in `data/failed_images.txt`. The high failure rate is expected — some Kaggle images have low contrast or unusual framing. Only images with a detected hand are included in the dataset.

MediaPipe was configured with:
- `num_hands = 1` — only the first detected hand is used
- `min_hand_detection_confidence = 0.5`
- No left/right mirroring — handedness is not normalized

## Feature representation

Each sample is a **(63,)** float32 vector: 21 MediaPipe hand landmarks × 3 coordinates (x, y, z), flattened row-major.

## Preprocessing: `normalize_landmarks()`

Defined in `preprocessing.py` (single source of truth; imported by both `train.py` and `infer.py`).

Two normalization steps applied to every sample before saving:

1. **Norm_Loc** — wrist-center: subtract landmark 0 (wrist) from all 21 landmarks so the hand position is origin-invariant.
2. **Norm_Zoom** — scale normalization: divide all coordinates by `max(||landmark_i[:2]||)` (maximum L2 norm of the (x, y) components across all landmarks). Makes the representation scale-invariant to hand distance from camera.

Raw MediaPipe coordinates are relative to image dimensions (0–1). After normalization, coordinates are unitless and camera-distance-independent.

## Splits

Stratified 70 / 15 / 15 split (random seed 42). Stored as actual data arrays in `splits.npz` (not index arrays).

| Split | Samples | % |
|---|---|---|
| Train | 44,505 | 70% |
| Val | 9,537 | 15% |
| Test | 9,537 | 15% |

Keys in `data/splits.npz`: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`.  
Labels are integer-indexed (0–27); mapping is in `data/label_to_index.json`.
