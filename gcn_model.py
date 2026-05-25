import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hand skeleton topology (MediaPipe 21 landmarks)
# ---------------------------------------------------------------------------

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),                        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                        # index
    (5, 9), (9, 10), (10, 11), (11, 12),                   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),                 # ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),       # pinky
]

# (prev, curr, next) triplets — angle computed at curr node
# Landmarks 0, 4, 8, 12, 16, 20 have no defined angle (wrist / fingertips)
ANGLE_TRIPLETS: dict[int, tuple[int, int, int]] = {
    1: (0, 1, 2),   2: (1, 2, 3),   3: (2, 3, 4),         # thumb
    5: (0, 5, 6),   6: (5, 6, 7),   7: (6, 7, 8),         # index
    9: (0, 9, 10),  10: (9, 10, 11), 11: (10, 11, 12),    # middle
    13: (0, 13, 14), 14: (13, 14, 15), 15: (14, 15, 16),  # ring
    17: (0, 17, 18), 18: (17, 18, 19), 19: (18, 19, 20),  # pinky
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_joint_angles(coords: np.ndarray) -> np.ndarray:
    """
    coords: (21, 3) normalized landmarks
    returns: (21,) float32 — angle in radians at each node, 0.0 where undefined
    """
    angles = np.zeros(21, dtype=np.float32)
    for curr, (prev, _, nxt) in ANGLE_TRIPLETS.items():
        va = coords[prev] - coords[curr]
        vb = coords[nxt] - coords[curr]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 1e-6 and nb > 1e-6:
            angles[curr] = np.arccos(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return angles


def compute_joint_angles_batch(X: np.ndarray) -> np.ndarray:
    """
    X: (N, 21, 3) normalized landmarks
    returns: (N, 21) float32 angles — vectorized over the batch dimension
    """
    N = X.shape[0]
    angles = np.zeros((N, 21), dtype=np.float32)
    for curr, (prev, _, nxt) in ANGLE_TRIPLETS.items():
        va = X[:, prev, :] - X[:, curr, :]           # (N, 3)
        vb = X[:, nxt,  :] - X[:, curr, :]           # (N, 3)
        na = np.linalg.norm(va, axis=1)              # (N,)
        nb = np.linalg.norm(vb, axis=1)
        valid = (na > 1e-6) & (nb > 1e-6)
        cos_a = np.einsum("ni,ni->n", va, vb) / np.where(valid, na * nb, 1.0)
        angles[valid, curr] = np.arccos(np.clip(cos_a[valid], -1.0, 1.0))
    return angles


def extract_node_features(coords: np.ndarray) -> np.ndarray:
    """
    coords: (21, 3) normalized landmarks
    returns: (21, 4) — appends joint angle as 4th feature per node
    """
    angles = compute_joint_angles(coords)
    return np.concatenate([coords, angles[:, None]], axis=1).astype(np.float32)


def extract_node_features_batch(X: np.ndarray) -> np.ndarray:
    """
    X: (N, 21, 3) normalized landmarks
    returns: (N, 21, 4)
    """
    angles = compute_joint_angles_batch(X)          # (N, 21)
    return np.concatenate([X, angles[:, :, None]], axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

def build_normalized_adjacency(num_nodes: int = 21) -> torch.Tensor:
    """D^{-1/2} (A + I) D^{-1/2} symmetric normalization."""
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in HAND_EDGES:
        A[i, j] = A[j, i] = 1.0
    A += torch.eye(num_nodes)                        # self-loops
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D                                 # (21, 21)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, 21, in_dim) — A_norm broadcasts over batch
        return A_norm @ self.linear(x)               # (B, 21, out_dim)


class HandGCN(nn.Module):
    """
    2-layer GCN on the MediaPipe hand skeleton.
    Input: (B, 21, 4)  — (x, y, z, joint_angle) per node
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int = 4,
        h1: int = 32,
        h2: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.register_buffer("A_norm", build_normalized_adjacency(21))
        self.gcn1 = GCNLayer(in_dim, h1)
        self.gcn2 = GCNLayer(h1, h2)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 21, 4)
        h = self.act(self.gcn1(x, self.A_norm))      # (B, 21, h1)
        h = self.dropout(h)
        h = self.act(self.gcn2(h, self.A_norm))      # (B, 21, h2)
        h = h.mean(dim=1)                             # (B, h2)  — mean pool over nodes
        h = self.dropout(h)
        return self.classifier(h)                     # (B, num_classes)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = torch.randn(2, 21, 4)
    model = HandGCN(num_classes=28)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    # angle feature check
    coords = np.random.rand(21, 3).astype("float32")
    feats = extract_node_features(coords)
    print(f"Node features shape: {feats.shape}  (expect (21, 4))")

    batch = np.random.rand(4, 21, 3).astype("float32")
    batch_feats = extract_node_features_batch(batch)
    print(f"Batch features shape: {batch_feats.shape}  (expect (4, 21, 4))")
    print("Sanity check passed.")
