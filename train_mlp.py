
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset



class MLP(nn.Module):
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


def load_splits(splits_path: str = "data/splits.npz"):
    sp = np.load(splits_path)
    return (
        sp["X_train"], sp["y_train"],
        sp["X_val"],   sp["y_val"],
        sp["X_test"],  sp["y_test"],
    )


def train_mlp(
    X_train, y_train, X_val, y_val,
    input_dim: int, num_classes: int,
    epochs: int = 50, batch_size: int = 128, lr: float = 1e-3,
) -> tuple[MLP, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    to_t = lambda x, dt: torch.tensor(x, dtype=dt).to(device)
    X_tr = to_t(X_train, torch.float32)
    y_tr = to_t(y_train, torch.long)
    X_v  = to_t(X_val,   torch.float32)
    y_v  = to_t(y_val,   torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = MLP(input_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

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
                    y_val, model(X_v).argmax(1).cpu().numpy()
                )
            print(f"  epoch {epoch:3d} | val_acc {val_acc:.4f}")

    return model, time.time() - t0


def evaluate(model: MLP, X_test, y_test) -> float:
    device = next(model.parameters()).device
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    return float(accuracy_score(y_test, preds))


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    with open("data/label_to_index.json") as f:
        label_to_index = json.load(f)
    num_classes = len(label_to_index)
    input_dim = X_train.shape[1]

    print(f"Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Classes: {num_classes}, input_dim: {input_dim}")

    # --- MLP ---
    print("\nTraining MLP (63→128→64→num_classes)...")
    model, mlp_time = train_mlp(X_train, y_train, X_val, y_val, input_dim, num_classes)
    mlp_acc = evaluate(model, X_test, y_test)
    print(f"MLP test accuracy: {mlp_acc:.4f}  ({mlp_time:.1f}s)")

    torch.save(
        {"state_dict": model.state_dict(), "input_dim": input_dim, "num_classes": num_classes},
        "models/model.pt",
    )
    print("Saved models/model.pt")

    # --- Random Forest ---
    print("\nTraining Random Forest (200 trees)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_acc = float(accuracy_score(y_test, rf.predict(X_test)))
    print(f"RF  test accuracy: {rf_acc:.4f}  ({rf_time:.1f}s)")

    # --- Save results ---
    results = {
        "mlp": {"test_accuracy": mlp_acc, "train_time_s": round(mlp_time, 1)},
        "rf":  {"test_accuracy": rf_acc,  "train_time_s": round(rf_time, 1)},
    }
    with open("results_mlp.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_mlp.json")
    print(f"\n{'Model':<8} {'Test Acc':>10} {'Train Time':>12}")
    print("-" * 32)
    print(f"{'MLP':<8} {mlp_acc:>10.4f} {mlp_time:>10.1f}s")
    print(f"{'RF':<8} {rf_acc:>10.4f} {rf_time:>10.1f}s")


if __name__ == "__main__":
    main()
