"""
Two PyTorch implementations of multi-output linear regression:
  1. From-scratch  — manual weight tensors, manual gradient update, no optimizer
  2. nn.Module     — torch.nn.Linear + torch.optim.Adam

Same data pipeline as model_sklearn.py so results are directly comparable.
Input  : 7 radar pings flattened (49 features)
Output : x_end, y_end, t_remaining
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA  = "/Users/yanbernard/MDMI/data"
SEED  = 42
LR    = 0.05
EPOCHS = 3000

torch.manual_seed(SEED)

# ── 1. Data (identical to sklearn script) ────────────────────────────────────
radar = pd.read_parquet(f"{DATA}/radar_marks_clean_3d.parquet")
full  = pd.read_parquet(f"{DATA}/full_trajectories_3d.parquet").sort_values(["trajectory_id", "t"])

landing = (
    full.groupby("trajectory_id")[["x", "y", "t"]]
    .last()
    .rename(columns={"x": "x_end", "y": "y_end", "t": "t_land"})
)
last_ping_t = (
    radar.sort_values(["trajectory_id", "t"])
    .groupby("trajectory_id")["t"].last()
    .rename("t_last_ping")
)
targets = landing.join(last_ping_t)
targets["t_remaining"] = targets["t_land"] - targets["t_last_ping"]

FEATURES = ["x", "y", "z", "vx", "vy", "vz", "v"]
radar_s  = radar.sort_values(["trajectory_id", "t"]).copy()
radar_s["ping_idx"] = radar_s.groupby("trajectory_id").cumcount()
wide = radar_s.pivot(index="trajectory_id", columns="ping_idx", values=FEATURES)
wide.columns = [f"{feat}_p{i}" for feat, i in wide.columns]
wide = wide.sort_index()
targets = targets.loc[wide.index]

X = wide.values
y = targets[["x_end", "y_end", "t_remaining"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Scale both X and y — critical for gradient descent convergence
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)

# Tensors
X_tr = torch.tensor(X_train_s, dtype=torch.float32)
y_tr = torch.tensor(y_train_s, dtype=torch.float32)
X_te = torch.tensor(X_test_s,  dtype=torch.float32)

n_feat, n_out = X_tr.shape[1], y_tr.shape[1]

# ── helpers ───────────────────────────────────────────────────────────────────
def evaluate(y_pred_raw, label):
    """y_pred_raw is in scaled space; inverse-transform then compute metrics."""
    y_pred = scaler_y.inverse_transform(y_pred_raw)
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    rmse2d = np.sqrt(np.mean(np.sum((y_pred[:, :2] - y_test[:, :2]) ** 2, axis=1)))
    print(f"\n=== {label} ===")
    print(f"  RMSE x_end      : {rmse(y_pred[:,0], y_test[:,0]):>10.1f} m")
    print(f"  RMSE y_end      : {rmse(y_pred[:,1], y_test[:,1]):>10.1f} m")
    print(f"  RMSE 2D landing : {rmse2d:>10.1f} m")
    print(f"  RMSE t_remaining: {rmse(y_pred[:,2], y_test[:,2]):>10.1f} s")
    print(f"  R²   x_end      : {r2_score(y_test[:,0], y_pred[:,0]):>10.4f}")
    print(f"  R²   y_end      : {r2_score(y_test[:,1], y_pred[:,1]):>10.4f}")
    print(f"  R²   t_remaining: {r2_score(y_test[:,2], y_pred[:,2]):>10.4f}")
    return y_pred

# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — FROM SCRATCH
# Manual weight matrix, manual gradient accumulation, manual update step.
# No nn.Module, no optimizer object.
# ═══════════════════════════════════════════════════════════════════════════════
print("Training from-scratch model ...")

W = torch.zeros(n_feat, n_out, requires_grad=True)
b = torch.zeros(n_out,         requires_grad=True)

losses_scratch = []

for epoch in range(EPOCHS):
    # Forward
    y_hat = X_tr @ W + b                          # (N, 3)
    loss  = ((y_hat - y_tr) ** 2).mean()

    # Backward
    loss.backward()

    # Manual parameter update (SGD step, no optimizer object)
    with torch.no_grad():
        W -= LR * W.grad
        b -= LR * b.grad

    # Zero gradients manually
    W.grad.zero_()
    b.grad.zero_()

    losses_scratch.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"  epoch {epoch+1:>4}/{EPOCHS}  loss={loss.item():.6f}")

# Predict on test set
with torch.no_grad():
    y_pred_scratch_s = (X_te @ W + b).numpy()

y_pred_scratch = evaluate(y_pred_scratch_s, "PYTORCH FROM-SCRATCH")

# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — nn.Module + torch.optim
# ═══════════════════════════════════════════════════════════════════════════════
print("\nTraining nn.Module model ...")

class LinearRegressor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

model     = LinearRegressor(n_feat, n_out)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

losses_nn = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_hat = model(X_tr)
    loss  = criterion(y_hat, y_tr)
    loss.backward()
    optimizer.step()

    losses_nn.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"  epoch {epoch+1:>4}/{EPOCHS}  loss={loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred_nn_s = model(X_te).numpy()

y_pred_nn = evaluate(y_pred_nn_s, "PYTORCH nn.Module + Adam")

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(losses_scratch, label="From-scratch (SGD)", alpha=0.8)
ax.plot(losses_nn,      label="nn.Module (Adam)",   alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss (scaled space)")
ax.set_title("Training loss — PyTorch from-scratch vs nn.Module")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/yanbernard/MDMI/eda/pytorch_training_curves.png", dpi=150)
plt.show()
print("\nTraining curves saved to eda/pytorch_training_curves.png")
