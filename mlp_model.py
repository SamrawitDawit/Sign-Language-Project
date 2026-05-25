import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    3-layer MLP for ASL landmark classification.
    Input: (B, 63) normalized landmark vector
    Output: (B, num_classes) logits
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
