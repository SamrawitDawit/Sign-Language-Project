import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from gcn_model import HandGCN, extract_node_features_batch


def load_splits(splits_path: str = "data/splits.npz") -> dict:
    sp = np.load(splits_path)
    out = {}
    for split in ("train", "val", "test"):
        X = sp[f"X_{split}"].reshape(-1, 21, 3)        # (N, 21, 3)
        X = extract_node_features_batch(X)              # (N, 21, 4)
        out[split] = (X, sp[f"y_{split}"])
    return out


def train(
    sets: dict,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
) -> tuple[HandGCN, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.long).to(device),
        )

    X_tr, y_tr = tensors(*sets["train"])
    X_v,  y_v  = tensors(*sets["val"])
    X_te, y_te = tensors(*sets["test"])

    loader   = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model    = HandGCN(num_classes=num_classes).to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn  = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_acc = accuracy_score(
                    y_v.cpu().numpy(),
                    model(X_v).argmax(1).cpu().numpy(),
                )
            print(f"  epoch {epoch:3d} | val_acc {val_acc:.4f}")

    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        test_preds = model(X_te).argmax(1).cpu().numpy()
    test_acc = float(accuracy_score(y_te.cpu().numpy(), test_preds))

    return model, test_acc, elapsed


def main():
    with open("data/label_to_index.json") as f:
        num_classes = len(json.load(f))

    print("Preparing node features (x, y, z + joint angle)...")
    sets = load_splits()
    tr_shape = sets["train"][0].shape
    print(f"  train {tr_shape}  val {sets['val'][0].shape}  test {sets['test'][0].shape}")

    print(f"\nTraining HandGCN ({num_classes} classes, in_dim=4)...")
    model, test_acc, elapsed = train(sets, num_classes)
    print(f"GCN test accuracy: {test_acc:.4f}  ({elapsed:.1f}s)")

    torch.save(
        {"state_dict": model.state_dict(), "num_classes": num_classes, "in_dim": 4},
        "models/model_gcn.pt",
    )
    print("Saved models/model_gcn.pt")

    with open("results_gcn.json", "w") as f:
        json.dump({"gcn": {"test_accuracy": test_acc, "train_time_s": round(elapsed, 1)}}, f, indent=2)
    print("Saved results_gcn.json")

    print(f"\nGCN test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
