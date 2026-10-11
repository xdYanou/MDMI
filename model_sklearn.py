"""
Sklearn linear regression:
  Input  : 7 radar pings (flattened, 49 features)
  Output : x_end, y_end, t_remaining  (landing point + time left after last ping)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA = "/Users/yanbernard/MDMI/data"
G    = 9.81   # m/s²

# ── 1. Load ──────────────────────────────────────────────────────────────────
radar = pd.read_parquet(f"{DATA}/radar_marks_clean_3d.parquet")
full  = pd.read_parquet(f"{DATA}/full_trajectories_3d.parquet").sort_values(["trajectory_id", "t"])

# ── 2. Targets ───────────────────────────────────────────────────────────────
# Landing point from full trajectories (last row, z ≈ 0)
landing = (
    full.groupby("trajectory_id")[["x", "y", "t"]]
    .last()
    .rename(columns={"x": "x_end", "y": "y_end", "t": "t_land"})
)

# Last radar ping time per trajectory
last_ping_t = (
    radar.sort_values(["trajectory_id", "t"])
    .groupby("trajectory_id")["t"]
    .last()
    .rename("t_last_ping")
)

targets = landing.join(last_ping_t)
targets["t_remaining"] = targets["t_land"] - targets["t_last_ping"]

# ── 3. Feature matrix: flatten 7 pings ───────────────────────────────────────
FEATURES = ["x", "y", "z", "vx", "vy", "vz", "v"]

radar_sorted = radar.sort_values(["trajectory_id", "t"])
radar_sorted = radar_sorted.copy()
radar_sorted["ping_idx"] = radar_sorted.groupby("trajectory_id").cumcount()

wide = radar_sorted.pivot(index="trajectory_id", columns="ping_idx", values=FEATURES)
wide.columns = [f"{feat}_p{i}" for feat, i in wide.columns]
wide = wide.sort_index()

targets = targets.loc[wide.index]

X = wide.values
y = targets[["x_end", "y_end", "t_remaining"]].values

print(f"Feature matrix : {X.shape}")
print(f"Targets        : {y.shape}  (x_end, y_end, t_remaining)")
print(f"t_remaining    : mean={y[:,2].mean():.1f}s  min={y[:,2].min():.1f}s  max={y[:,2].max():.1f}s")

# ── 4. Split, scale, fit ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.array(wide.index), test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

# ── 5. Metrics ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_s)

rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
print("\n=== SKLEARN LINEAR REGRESSION ===")
print(f"  RMSE x_end      : {rmse(y_pred[:,0], y_test[:,0]):>10.1f} m")
print(f"  RMSE y_end      : {rmse(y_pred[:,1], y_test[:,1]):>10.1f} m")
print(f"  RMSE 2D landing : {np.sqrt(np.mean(np.sum((y_pred[:,:2]-y_test[:,:2])**2,axis=1))):>10.1f} m")
print(f"  RMSE t_remain   : {rmse(y_pred[:,2], y_test[:,2]):>10.1f} s")
print(f"  R²   x_end      : {r2_score(y_test[:,0], y_pred[:,0]):>10.4f}")
print(f"  R²   y_end      : {r2_score(y_test[:,1], y_pred[:,1]):>10.4f}")
print(f"  R²   t_remaining: {r2_score(y_test[:,2], y_pred[:,2]):>10.4f}")

# ── 6. Reconstruct predicted trajectory for one sample ────────────────────────
def make_pred_trajectory(last_ping, x_end_pred, y_end_pred, t_rem_pred, n=200):
    """
    From the last radar ping, build a predicted path to landing.
      - x, y : linearly interpolated to predicted landing
      - z     : kinematic parabola using last observed vz
    """
    x_l, y_l, z_l = last_ping["x"], last_ping["y"], last_ping["z"]
    vz_l = last_ping["vz"]

    t_rem_pred = max(t_rem_pred, 1.0)          # guard against negative prediction
    tau = np.linspace(0, t_rem_pred, n)

    x_p = x_l + (x_end_pred - x_l) * tau / t_rem_pred
    y_p = y_l + (y_end_pred - y_l) * tau / t_rem_pred
    z_p = np.maximum(0, z_l + vz_l * tau - 0.5 * G * tau**2)

    return tau, x_p, y_p, z_p

# Pick 3 test trajectories to plot
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for col, sample_i in enumerate([0, 3, 7]):
    tid   = idx_test[sample_i]
    xp, yp, tr = y_pred[sample_i]
    xt, yt, tt = y_test[sample_i]

    # Full true trajectory
    true_traj = full[full["trajectory_id"] == tid]

    # Radar pings
    pings = radar_sorted[radar_sorted["trajectory_id"] == tid].sort_values("t")
    last  = pings.iloc[-1]

    # Predicted path
    tau, x_p, y_p, z_p = make_pred_trajectory(last, xp, yp, tr)

    ax = axes[col]
    # Ground track (top-down x vs y)
    ax.plot(true_traj["x"]/1000, true_traj["y"]/1000,
            color="lightgray", lw=1.5, label="True trajectory")
    ax.plot(pings["x"]/1000, pings["y"]/1000,
            "kx", ms=7, mew=2, label="Radar pings")
    ax.plot(x_p/1000, y_p/1000,
            color="tab:orange", lw=2, ls="--", label="Predicted path")

    # True landing
    ax.scatter(xt/1000, yt/1000, color="green", s=80, zorder=5, label=f"True landing")
    # Predicted landing
    ax.scatter(xp/1000, yp/1000, color="tab:red", s=80, marker="*", zorder=5,
               label=f"Predicted landing")

    err_2d = np.sqrt((xp-xt)**2 + (yp-yt)**2) / 1000
    is_rocket = full[full["trajectory_id"]==tid]["is_rocket"].iloc[0]
    label = "Rocket" if is_rocket else "Ballistic"
    ax.set_title(f"{tid} ({label})\n2D err={err_2d:.1f} km | Δt={abs(tr-tt):.1f}s")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

plt.suptitle("Sklearn Linear Regression — Predicted vs True trajectory (ground track)", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/yanbernard/MDMI/eda/sklearn_predictions.png", dpi=150)
plt.show()
print("\nPlot saved to eda/sklearn_predictions.png")
