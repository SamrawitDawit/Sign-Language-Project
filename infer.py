import json
from pathlib import Path

import numpy as np
import torch

from preprocessing import normalize_landmarks
from gcn_model import HandGCN, extract_node_features

_model: HandGCN | None = None
_index_to_label: dict[int, str] | None = None


def _load() -> None:
    global _model, _index_to_label
    if _model is not None:
        return

    root = Path(__file__).parent
    with open(root / "data" / "label_to_index.json") as f:
        label_to_index: dict[str, int] = json.load(f)
    _index_to_label = {v: k for k, v in label_to_index.items()}

    ckpt = torch.load(root / "models" / "model_gcn.pt", map_location="cpu", weights_only=True)
    _model = HandGCN(num_classes=ckpt["num_classes"], in_dim=ckpt["in_dim"])
    _model.load_state_dict(ckpt["state_dict"])
    _model.eval()


def predict(landmarks: np.ndarray) -> tuple[str, float]:
    """
    landmarks: (21, 3) raw MediaPipe coords for ONE hand, or (63,) already normalized
    returns: (letter, confidence in [0, 1])

    Applies Norm_Loc + Norm_Zoom normalization when given raw (21, 3) input,
    then computes joint angles to produce (21, 4) node features for the GCN.
    """
    _load()

    if landmarks.shape == (21, 3):
        coords = normalize_landmarks(landmarks).reshape(21, 3)
    else:
        coords = landmarks.astype(np.float32).reshape(21, 3)

    features = extract_node_features(coords)                        # (21, 4)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)   # (1, 21, 4)

    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    idx = int(probs.argmax())
    return _index_to_label[idx], float(probs[idx])
