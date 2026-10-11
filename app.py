"""
app.py — Gradio dashboard.  Loads pre-trained artifacts from models/.
NEVER retrains anything.  Run train.py first.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gradio as gr

MODELS  = "models"
DATA    = "data"
G       = 9.81

# ── Load artifacts ────────────────────────────────────────────────────────────
scaler_X       = joblib.load(f"{MODELS}/scaler_X.pkl")
scaler_y       = joblib.load(f"{MODELS}/scaler_y.pkl")
sk_model       = joblib.load(f"{MODELS}/sklearn_ridge.pkl")
baseline_mean  = np.load(f"{MODELS}/baseline_mean.npy")

scratch_W      = torch.tensor(np.load(f"{MODELS}/scratch_W.npy"), dtype=torch.float32)
scratch_b      = torch.tensor(np.load(f"{MODELS}/scratch_b.npy"), dtype=torch.float32)

with open(f"{MODELS}/nn_arch.json") as f:
    arch = json.load(f)

class LinearRegressor(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
    def forward(self, x):
        return self.linear(x)

nn_model = LinearRegressor(arch["in_features"], arch["out_features"])
nn_model.load_state_dict(torch.load(f"{MODELS}/nn_model_state.pt", weights_only=True))
nn_model.eval()

split    = np.load(f"{MODELS}/test_split.npz", allow_pickle=True)
X_te_s   = split["X_te_s"]
y_te     = split["y_te"]
ids_te   = split["ids_te"]
y_pred_baseline = split["y_pred_baseline"]
y_pred_sk       = split["y_pred_sk"]
y_pred_scratch  = split["y_pred_scratch"]
y_pred_nn       = split["y_pred_nn"]

curves   = np.load(f"{MODELS}/train_curves.npz")
losses_scratch  = curves["losses_scratch"]
losses_nn       = curves["losses_nn"]

with open(f"{MODELS}/metrics.json") as f:
    all_metrics = json.load(f)

features_df  = pd.read_parquet(f"{MODELS}/features.parquet")
targets_df   = pd.read_parquet(f"{MODELS}/targets.parquet")
display_traj = pd.read_parquet(f"{DATA}/display_trajectories.parquet")
radar_all    = pd.read_parquet(f"{DATA}/radar_marks_clean_3d.parquet")
radar_s      = radar_all.sort_values(["trajectory_id", "t"]).copy()
radar_s["ping_idx"] = radar_s.groupby("trajectory_id").cumcount()

# ── Visual constants ──────────────────────────────────────────────────────────
BG      = "#0a0f1a"
GRID_C  = "#1a3a2a"
RING_C  = "#0d2d1a"
TRUE_C  = "#2a4a3a"
PING_C  = "#00ff88"
LAST_C  = "#ffffff"
LAND_C  = "#ff3333"
MODEL_STYLES = {
    "Baseline":         ("#888888", ":"),
    "Sklearn Ridge":    ("#00cfff", "-"),
    "Scratch (SGD)":    ("#ffaa00", "--"),
    "nn.Module (Adam)": ("#cc44ff", "-."),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _glow(ax, xd, yd, color, lw=1.5, ls="-", alpha=0.9, zorder=3):
    for w, a in [(lw*6, 0.06), (lw*3, 0.15), (lw, alpha)]:
        ax.plot(xd, yd, color=color, lw=w, ls=ls, alpha=a,
                zorder=zorder, solid_capstyle="round")

def _range_rings(ax, cx, cy, max_r, n=5):
    step = max(1, round(max_r / n / 5) * 5)
    for r in range(step, int(max_r) + step, step):
        ax.add_patch(plt.Circle((cx, cy), r, color=RING_C, fill=False,
                                lw=0.6, ls="--", zorder=0))
        ax.text(cx, cy + r, f"{r} km", color="#1a5a3a",
                fontsize=6, ha="center", va="bottom", zorder=0)

def make_pred_path(last_ping, x_end, y_end, t_rem, n=300):
    t_rem = max(float(t_rem), 1.0)
    tau   = np.linspace(0, t_rem, n)
    xp = float(last_ping["x"]) + (x_end - float(last_ping["x"])) * tau / t_rem
    yp = float(last_ping["y"]) + (y_end - float(last_ping["y"])) * tau / t_rem
    zp = np.maximum(0, float(last_ping["z"]) + float(last_ping["vz"]) * tau - 0.5*G*tau**2)
    return xp, yp, zp

def predict_one(tid):
    """Return predictions for one trajectory from all models."""
    feat_row = features_df.loc[tid].values.astype(np.float32).reshape(1, -1)
    xs = scaler_X.transform(feat_row)

    p_sk  = sk_model.predict(xs)[0]

    xt = torch.tensor(xs, dtype=torch.float32)
    with torch.no_grad():
        p_sc_s = (xt @ scratch_W + scratch_b).numpy()
        p_sc   = scaler_y.inverse_transform(p_sc_s)[0]
        p_nn_s = nn_model(xt).numpy()
        p_nn   = scaler_y.inverse_transform(p_nn_s)[0]

    p_base = baseline_mean.copy()
    return p_base, p_sk, p_sc, p_nn

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Trajectory Inspector
# ═══════════════════════════════════════════════════════════════════════════════
ALL_IDS = sorted(features_df.index.tolist())

def plot_trajectory(tid):
    true_traj = display_traj[display_traj["trajectory_id"] == tid].sort_values("t")
    pings     = radar_s[radar_s["trajectory_id"] == tid].sort_values("t")
    last      = pings.iloc[-1]
    true_end  = targets_df.loc[tid, ["x_end", "y_end"]].values

    p_base, p_sk, p_sc, p_nn = predict_one(tid)
    is_rocket = bool(targets_df.loc[tid, "is_rocket"])
    obj_label = "ROCKET" if is_rocket else "BALLISTIC"

    preds = {
        "Sklearn Ridge":    (p_sk,  *MODEL_STYLES["Sklearn Ridge"]),
        "Scratch (SGD)":    (p_sc,  *MODEL_STYLES["Scratch (SGD)"]),
        "nn.Module (Adam)": (p_nn,  *MODEL_STYLES["nn.Module (Adam)"]),
    }

    tx, ty, tz = true_traj["x"]/1000, true_traj["y"]/1000, true_traj["z"]/1000
    px_, py_, pz_ = pings["x"]/1000, pings["y"]/1000, pings["z"]/1000

    fig, (ax_gt, ax_sv) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG,
                                        gridspec_kw={"wspace": 0.35})

    # ── Ground track ──────────────────────────────────────────────────────────
    ax = ax_gt;  ax.set_facecolor(BG)
    lx0, ly0 = float(tx.iloc[0]), float(ty.iloc[0])
    all_pts   = [(p[0]/1000, p[1]/1000) for p in [p_sk, p_sc, p_nn]]
    max_r = max(np.hypot(tx-lx0, ty-ly0).max(),
                max(np.hypot(x-lx0, y-ly0) for x,y in all_pts)) + 5
    _range_rings(ax, lx0, ly0, max_r)
    for xv in np.linspace(float(tx.min())-5, float(tx.max())+5, 7):
        ax.axvline(xv, color=GRID_C, lw=0.4, zorder=0)
    for yv in np.linspace(float(ty.min())-5, float(ty.max())+5, 7):
        ax.axhline(yv, color=GRID_C, lw=0.4, zorder=0)

    ax.plot(tx, ty, color=TRUE_C, lw=1.2, zorder=1)
    for xi, yi in zip(px_.iloc[:-1], py_.iloc[:-1]):
        ax.scatter(xi, yi, color=PING_C, s=45, zorder=6)
        ax.scatter(xi, yi, color=PING_C, s=160, facecolors="none",
                   edgecolors=PING_C, lw=0.8, alpha=0.4, zorder=5)
    ax.scatter(float(px_.iloc[-1]), float(py_.iloc[-1]),
               color=LAST_C, s=100, zorder=8)
    ax.scatter(float(px_.iloc[-1]), float(py_.iloc[-1]),
               color=LAST_C, s=350, facecolors="none",
               edgecolors=LAST_C, lw=1.2, alpha=0.5, zorder=7)

    vx_n = float(last["vx"]); vy_n = float(last["vy"])
    spd  = np.hypot(vx_n, vy_n) + 1e-9
    al   = max_r * 0.08
    ax.annotate("", xy=(float(px_.iloc[-1]) + al*vx_n/spd,
                         float(py_.iloc[-1]) + al*vy_n/spd),
                xytext=(float(px_.iloc[-1]), float(py_.iloc[-1])),
                arrowprops=dict(arrowstyle="->", color=LAST_C, lw=1.5))

    for mname, (pred, col, ls) in preds.items():
        mpx, mpy, _ = make_pred_path(last, pred[0], pred[1], pred[2])
        _glow(ax, mpx/1000, mpy/1000, col, lw=1.8, ls=ls, zorder=3)
        ax.scatter(pred[0]/1000, pred[1]/1000, color=col, s=90, marker="*", zorder=9)
        for rs in [120, 240]:
            ax.scatter(pred[0]/1000, pred[1]/1000, s=rs,
                       facecolors="none", edgecolors=col,
                       lw=0.7, alpha=0.5, zorder=8)

    ax.scatter(true_end[0]/1000, true_end[1]/1000,
               color=LAND_C, s=120, marker="D", zorder=10)
    ax.axvline(true_end[0]/1000, color=LAND_C, lw=0.5, alpha=0.4, zorder=2)
    ax.axhline(true_end[1]/1000, color=LAND_C, lw=0.5, alpha=0.4, zorder=2)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (km)", color="#88aaaa", fontsize=9)
    ax.set_ylabel("North (km)", color="#88aaaa", fontsize=9)
    ax.tick_params(colors="#446655", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a3a2a")
    ax.set_title("GROUND TRACK  [PPI]", color="#00ff88",
                 fontsize=10, fontweight="bold", pad=8)

    spd_ms = float(last["v"]); alt_m = float(last["z"])
    brg    = float(np.degrees(np.arctan2(vy_n, vx_n))) % 360
    hud = (f"TRACK ID : {tid}\nCLASS    : {obj_label}\n"
           f"SPEED    : {spd_ms:.0f} m/s\nALT      : {alt_m:.0f} m\n"
           f"BEARING  : {brg:.1f}°\nPINGS    : {len(pings)}")
    ax.text(0.02, 0.98, hud, transform=ax.transAxes, fontsize=7.5,
            fontfamily="monospace", color="#00ff88", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#050f0a",
                      edgecolor="#00aa44", alpha=0.85))

    # ── Side view ─────────────────────────────────────────────────────────────
    ax = ax_sv;  ax.set_facecolor(BG)
    gx = np.array([float(tx.min())-5, float(tx.max())+5])
    ax.fill_between(gx, [-0.5,-0.5], [0,0], color="#1a2a10", zorder=0)
    ax.plot(gx, [0,0], color="#2a5a20", lw=1.5, zorder=1)
    for alt_km in [10, 20]:
        if alt_km < float(tz.max()) + 2:
            ax.axhline(alt_km, color="#111a33", lw=0.6, ls=":", zorder=0)
            ax.text(float(tx.min())-4, alt_km+0.2, f"{alt_km} km",
                    color="#222844", fontsize=6, zorder=0)
    for yv in np.arange(0, float(tz.max())+5, 5):
        ax.axhline(yv, color=GRID_C, lw=0.3, zorder=0)
    for xv in np.linspace(float(tx.min()), float(tx.max()), 8):
        ax.axvline(xv, color=GRID_C, lw=0.3, zorder=0)

    ax.plot(tx, tz, color=TRUE_C, lw=1.2, zorder=1)
    for xi, zi in zip(px_.iloc[:-1], pz_.iloc[:-1]):
        ax.scatter(xi, zi, color=PING_C, s=45, zorder=6)
        ax.scatter(xi, zi, color=PING_C, s=160, facecolors="none",
                   edgecolors=PING_C, lw=0.8, alpha=0.4, zorder=5)
        ax.plot([xi,xi], [0,zi], color=PING_C, lw=0.5, alpha=0.3, ls=":", zorder=2)
    ax.scatter(float(px_.iloc[-1]), float(pz_.iloc[-1]),
               color=LAST_C, s=100, zorder=8)
    ax.scatter(float(px_.iloc[-1]), float(pz_.iloc[-1]),
               color=LAST_C, s=350, facecolors="none",
               edgecolors=LAST_C, lw=1.2, alpha=0.5, zorder=7)

    for mname, (pred, col, ls) in preds.items():
        mpx, mpy, mpz = make_pred_path(last, pred[0], pred[1], pred[2])
        _glow(ax, mpx/1000, mpz/1000, col, lw=1.8, ls=ls, zorder=3)
        apex_i = np.argmax(mpz)
        ax.scatter(mpx[apex_i]/1000, mpz[apex_i]/1000,
                   color=col, s=60, marker="^", zorder=8, alpha=0.7)
        ax.scatter(pred[0]/1000, 0, color=col, s=90, marker="*", zorder=9)
        ax.plot([pred[0]/1000, pred[0]/1000], [0, mpz[apex_i]/1000],
                color=col, lw=0.4, ls=":", alpha=0.3, zorder=2)

    ax.scatter(true_end[0]/1000, 0, color=LAND_C, s=120, marker="D", zorder=10)
    ax.axvline(true_end[0]/1000, color=LAND_C, lw=0.5, alpha=0.4, zorder=2)

    ax.set_xlabel("East (km)", color="#88aaaa", fontsize=9)
    ax.set_ylabel("Altitude (km)", color="#88aaaa", fontsize=9)
    ax.tick_params(colors="#446655", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a3a2a")
    ax.set_title("ALTITUDE PROFILE  [SIDE VIEW]", color="#00ff88",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_ylim(bottom=-0.5)

    # Shared legend + layout
    legend_elements = [
        plt.Line2D([0],[0], color=TRUE_C, lw=1.5, label="True trajectory"),
        plt.Line2D([0],[0], color=PING_C, lw=0, marker="o", ms=6, label="Radar pings"),
        plt.Line2D([0],[0], color=LAST_C, lw=0, marker="o", ms=7, label="Last ping"),
        plt.Line2D([0],[0], color=LAND_C, lw=0, marker="D", ms=7, label="True impact"),
    ]
    for mname, (col, ls) in MODEL_STYLES.items():
        if mname != "Baseline":
            legend_elements.append(plt.Line2D([0],[0], color=col, lw=2, ls=ls, label=mname))

    fig.patch.set_facecolor(BG)
    fig.suptitle(f"RADAR TRACK  ·  {tid}  ·  {obj_label}",
                 color="#00ff88", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.14, left=0.07, right=0.97)
    fig.legend(handles=legend_elements, loc="lower center", ncol=7,
               fontsize=8, facecolor="#050f0a", edgecolor="#1a3a2a",
               labelcolor="#aaccaa", framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    errors = ""
    for mname, (pred, col, ls) in preds.items():
        e2d = np.sqrt((pred[0]-true_end[0])**2 + (pred[1]-true_end[1])**2) / 1000
        et  = abs(pred[2] - float(targets_df.loc[tid, "t_remaining"]))
        errors += f"{mname:<22}  2D err={e2d:.1f} km   Δt={et:.1f}s\n"

    return fig, errors.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison  (metrics table + pred vs actual scatter)
# ═══════════════════════════════════════════════════════════════════════════════
def show_comparison_table():
    rows = []
    for m in all_metrics:
        rows.append({
            "Model":            m["name"],
            "RMSE 2D (m)":      round(m["rmse_2d"]),
            "RMSE x (m)":       round(m["rmse_x"]),
            "RMSE y (m)":       round(m["rmse_y"]),
            "RMSE Δt (s)":      round(m["rmse_t"], 1),
            "R² x":             round(m["r2_x"], 4),
            "R² y":             round(m["r2_y"], 4),
            "R² Δt":            round(m["r2_t"], 4),
        })
    return pd.DataFrame(rows)

def plot_pred_vs_actual():
    """Scatter: predicted 2D range vs actual 2D range for all test trajectories."""
    true_range = np.sqrt(y_te[:, 0]**2 + y_te[:, 1]**2) / 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#111111")
    pairs = [
        ("Sklearn Ridge",    y_pred_sk,      "#00cfff"),
        ("Scratch (SGD)",    y_pred_scratch, "#ffaa00"),
        ("nn.Module (Adam)", y_pred_nn,      "#cc44ff"),
    ]
    for ax, (name, y_pred, col) in zip(axes, pairs):
        pred_range = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2) / 1000
        ax.set_facecolor("#0a0f1a")
        ax.scatter(true_range, pred_range, color=col, alpha=0.25, s=8, zorder=3)
        lim = [min(true_range.min(), pred_range.min()) - 1,
               max(true_range.max(), pred_range.max()) + 1]
        ax.plot(lim, lim, color="#ff3333", lw=1.5, ls="--", zorder=4, label="Perfect")
        r2 = float(np.corrcoef(true_range, pred_range)[0, 1]**2)
        ax.set_title(f"{name}\nR²={r2:.4f}", color="#00ff88", fontsize=10)
        ax.set_xlabel("True range (km)", color="#88aaaa", fontsize=9)
        ax.set_ylabel("Predicted range (km)", color="#88aaaa", fontsize=9)
        ax.tick_params(colors="#446655", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a2a")
        ax.grid(alpha=0.2, color="#1a3a2a")

    fig.patch.set_facecolor("#111111")
    fig.suptitle("Predicted vs Actual 2D Range — Test Set (n=1000)",
                 color="#00ff88", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature & Target Distributions
# ═══════════════════════════════════════════════════════════════════════════════
def plot_distributions():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="#111111")
    axes = axes.flatten()

    # Target distributions
    target_cols = ["x_end", "y_end", "t_remaining"]
    target_lbls = ["Landing x (m)", "Landing y (m)", "Time remaining (s)"]
    colors_t    = ["#00cfff", "#cc44ff", "#ffaa00"]
    for i, (col, lbl, col_c) in enumerate(zip(target_cols, target_lbls, colors_t)):
        ax = axes[i]
        ax.set_facecolor("#0a0f1a")
        vals = targets_df[col].values
        ax.hist(vals, bins=40, color=col_c, alpha=0.8, edgecolor="none")
        ax.set_title(lbl, color="#00ff88", fontsize=9)
        ax.set_xlabel("Value", color="#88aaaa", fontsize=8)
        ax.set_ylabel("Count", color="#88aaaa", fontsize=8)
        ax.tick_params(colors="#446655", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a2a")

    # Engineered feature distributions
    eng_feats = ["h_speed_last", "h_range_obs", "est_apogee_km", "v_slope", "bearing_rad"]
    eng_lbls  = ["Horiz. speed last ping (m/s)", "Horiz. range observed (m)",
                 "Est. apogee (km)", "Speed slope (m/s per ping)", "Bearing (rad)"]
    colors_e  = ["#00ff88", "#ff8844", "#44aaff", "#ffcc00", "#ff44aa"]
    for i, (feat, lbl, col_c) in enumerate(zip(eng_feats, eng_lbls, colors_e)):
        ax = axes[i + 3]
        ax.set_facecolor("#0a0f1a")
        if feat in features_df.columns:
            vals = features_df[feat].values
            ax.hist(vals, bins=40, color=col_c, alpha=0.8, edgecolor="none")
        ax.set_title(lbl, color="#00ff88", fontsize=9)
        ax.set_xlabel("Value", color="#88aaaa", fontsize=8)
        ax.set_ylabel("Count", color="#88aaaa", fontsize=8)
        ax.tick_params(colors="#446655", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a2a")

    fig.patch.set_facecolor("#111111")
    fig.suptitle("Feature & Target Distributions", color="#00ff88",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Training Curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_curves():
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0a0f1a")
    ax.set_facecolor("#0a0f1a")
    _glow(ax, range(len(losses_scratch)), losses_scratch,
          "#ffaa00", lw=1.5, label=None)
    _glow(ax, range(len(losses_nn)), losses_nn,
          "#cc44ff", lw=1.5, label=None)
    ax.plot([], [], color="#ffaa00", lw=2, label="From-scratch (SGD)")
    ax.plot([], [], color="#cc44ff", lw=2, label="nn.Module (Adam)")
    ax.set_xlabel("Epoch", color="#88aaaa", fontsize=9)
    ax.set_ylabel("MSE loss (scaled space)", color="#88aaaa", fontsize=9)
    ax.set_title("Training Loss Convergence", color="#00ff88",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="#446655", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a3a2a")
    ax.grid(alpha=0.2, color="#1a3a2a")
    ax.legend(facecolor="#050f0a", edgecolor="#1a3a2a", labelcolor="#aaccaa", fontsize=9)
    fig.patch.set_facecolor("#0a0f1a")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Gradio layout
# ═══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Missile Trajectory Prediction") as demo:
    gr.Markdown(
        """
        # Missile Trajectory Prediction
        **Business problem:** A ground radar detects an incoming object and records **7 pings** (~36 s window).
        Using only those pings, predict the **landing point (x, y)** and **time to impact** before it leaves sensor range.

        Three implementations of the same **multi-output linear regression** (Ridge, α=1.0):
        `sklearn` · `PyTorch from-scratch (SGD)` · `PyTorch nn.Module (Adam)`

        *Models are pre-trained — this app never retrains anything.*
        """
    )

    with gr.Tab("Trajectory Inspector"):
        gr.Markdown("Pick any trajectory. All three models predict the remaining flight path from the last radar ping.")
        tid_dd  = gr.Dropdown(choices=ALL_IDS, value=ALL_IDS[0], label="Trajectory ID")
        btn     = gr.Button("Predict", variant="primary")
        plot_out  = gr.Plot(label="Radar track")
        error_out = gr.Textbox(label="Per-model landing errors", lines=4)
        btn.click(fn=plot_trajectory, inputs=[tid_dd], outputs=[plot_out, error_out])

    with gr.Tab("Model Comparison"):
        gr.Markdown("### Metrics — test set (n=1,000 trajectories)")
        gr.DataFrame(value=show_comparison_table())
        gr.Markdown(
            """
            **Why the gap between implementations?**
            `sklearn Ridge` solves the normal equations analytically (exact minimum).
            `nn.Module + Adam` uses adaptive gradient descent — converges close but not exact at 3,000 epochs.
            `Scratch + SGD` uses vanilla gradient descent — slowest convergence, largest gap.
            Given unlimited epochs all three reach the same solution; the gap demonstrates the
            convergence cost of gradient-based vs closed-form optimisation.

            **Baseline:** predicting the training-set mean for every trajectory gives RMSE 2D ≈ 34 km
            and R² ≈ 0 — all three learned models improve by ~8×.
            """
        )
        gr.Markdown("### Predicted vs Actual 2D Range")
        scatter_btn = gr.Button("Show scatter plots")
        scatter_plot = gr.Plot()
        scatter_btn.click(fn=plot_pred_vs_actual, inputs=[], outputs=scatter_plot)

    with gr.Tab("Distributions"):
        gr.Markdown("Target and engineered-feature distributions across all 4,999 trajectories.")
        dist_btn = gr.Button("Show distributions")
        dist_plot = gr.Plot()
        dist_btn.click(fn=plot_distributions, inputs=[], outputs=dist_plot)

    with gr.Tab("Training Curves"):
        gr.Markdown("Loss curves for both PyTorch implementations (3,000 epochs).")
        curves_btn  = gr.Button("Show curves")
        curves_plot = gr.Plot()
        curves_btn.click(fn=plot_curves, inputs=[], outputs=curves_plot)

demo.launch()
