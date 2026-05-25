import numpy as np


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """Wrist-center and scale-normalize (21, 3) MediaPipe landmarks, returns (63,).

    Matches the preprocessing applied during training data extraction in
    asl_landmark_pipeline.ipynb. Must be used identically in both train.py and
    infer.py to avoid train/inference mismatch.

    Norm_Loc: subtract wrist (landmark 0)
    Norm_Zoom: divide by max L2 norm of (x, y) across all landmarks
    """
    coords = coords.astype(np.float32).reshape(21, 3).copy()
    coords -= coords[0]
    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if scale > 0:
        coords /= scale
    return coords.reshape(-1)
