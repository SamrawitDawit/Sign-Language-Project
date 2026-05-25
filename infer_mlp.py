import json
from pathlib import Path

import numpy as np
import torch

from mlp_model import MLP
from preprocessing import normalize_landmarks

_model: MLP | None = None
_index_to_label: dict[int, str] | None = None


def _load():
    global _model, _index_to_label
    if _model is not None:
        return

    root = Path(__file__).parent
    with open(root / "data" / "label_to_index.json") as f:
        label_to_index: dict[str, int] = json.load(f)
    _index_to_label = {v: k for k, v in label_to_index.items()}

    ckpt = torch.load(root / "models" / "model.pt", map_location="cpu", weights_only=True)
    _model = MLP(ckpt["input_dim"], ckpt["num_classes"])
    _model.load_state_dict(ckpt["state_dict"])
    _model.eval()


def predict(landmarks: np.ndarray) -> tuple[str, float]:
    """
    landmarks: (21, 3) raw MediaPipe output for ONE hand, or (63,) already normalized
    returns: (letter, confidence in [0, 1])

    Applies the same Norm_Loc + Norm_Zoom normalization used during training
    when given raw (21, 3) input.
    """
    _load()

    if landmarks.shape == (21, 3) or (landmarks.ndim == 2 and landmarks.shape == (21, 3)):
        vec = normalize_landmarks(landmarks)
    else:
        vec = landmarks.astype(np.float32).reshape(-1)

    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    idx = int(probs.argmax())
    return _index_to_label[idx], float(probs[idx])
