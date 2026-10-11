"""
Apply a random horizontal bearing angle per trajectory so that motion
is no longer locked to the x-axis.  The same angle is applied to all
three datasets so radar pings and full-trajectory rows stay consistent.

Physics:
  - Let dx(t) = x(t) - x_launch  (horizontal distance travelled along original axis)
  - x_new(t) = x_launch + dx(t) * cos(θ)
  - y_new(t) = y_launch + dx(t) * sin(θ)
  - vx_new   = vx_orig  * cos(θ)
  - vy_new   = vx_orig  * sin(θ)
  - v stays the same: sqrt(vx_new²+vy_new²+vz²) = sqrt(vx²+vz²) = v  ✓
  - z, vz, t, is_rocket  untouched
"""

import numpy as np
import pandas as pd

DATA = "/Users/yanbernard/MDMI/data"

# ── load ────────────────────────────────────────────────────────────────────
full  = pd.read_parquet(f"{DATA}/full_trajectories.parquet")
clean = pd.read_parquet(f"{DATA}/radar_marks_clean.parquet")
noisy = pd.read_parquet(f"{DATA}/radar_marks_noisy.parquet")

# ── one bearing per trajectory (reproducible) ────────────────────────────────
rng   = np.random.default_rng(42)
tids  = full["trajectory_id"].unique()
theta = rng.uniform(0, 2 * np.pi, size=len(tids))
bearing = pd.Series(theta, index=tids)          # trajectory_id → angle

# ── launch position from full trajectories (t=0 row) ────────────────────────
launch = (
    full.sort_values("t")
    .groupby("trajectory_id")[["x", "y"]]
    .first()
    .rename(columns={"x": "x_launch", "y": "y_launch"})
)

# ── transformation ───────────────────────────────────────────────────────────
def apply_bearing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    tid        = df["trajectory_id"]
    cos_t      = np.cos(bearing[tid].values)
    sin_t      = np.sin(bearing[tid].values)
    x_launch   = launch.loc[tid, "x_launch"].values
    y_launch   = launch.loc[tid, "y_launch"].values

    dx         = df["x"].values - x_launch        # ground-track displacement
    vx_orig    = df["vx"].values

    df["x"]    = x_launch + dx * cos_t
    df["y"]    = y_launch + dx * sin_t
    df["vx"]   = vx_orig  * cos_t
    df["vy"]   = vx_orig  * sin_t
    # v is recomputed to be exactly consistent (handles float rounding)
    df["v"]    = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)

    return df

full_3d  = apply_bearing(full)
clean_3d = apply_bearing(clean)
noisy_3d = apply_bearing(noisy)

# ── sanity checks ─────────────────────────────────────────────────────────────
print("=== SANITY CHECKS ===")

# 1. v should be identical before and after (within float tolerance)
v_diff = (full_3d["v"] - full["v"]).abs()
print(f"[v unchanged]  max |Δv| = {v_diff.max():.6f} m/s  (should be ~0)")

# 2. z unchanged
z_diff = (full_3d["z"] - full["z"]).abs()
print(f"[z unchanged]  max |Δz| = {z_diff.max():.6f} m    (should be 0)")

# 3. vy now varies
print(f"[vy varies]    vy std in 3d full = {full_3d['vy'].std():.1f} m/s  (was 0)")

# 4. Cross-dataset consistency: impact point in full_3d == impact in clean_3d?
#    (last ping in clean should match the same row in full_3d at that t)
sample_tid = "T_000042"
f_row = full_3d[full_3d["trajectory_id"] == sample_tid].sort_values("t").iloc[-1]
c_row = clean_3d[clean_3d["trajectory_id"] == sample_tid].sort_values("t").iloc[-1]

# Both should have same bearing applied from same x_launch
f_launch = launch.loc[sample_tid]
print(f"\n[Cross-dataset check for {sample_tid}]")
print(f"  full_3d  last point: x={f_row['x']:.1f}  y={f_row['y']:.1f}  z={f_row['z']:.1f}")
print(f"  clean_3d last ping:  x={c_row['x']:.1f}  y={c_row['y']:.1f}  z={c_row['z']:.1f}")
print(f"  bearing θ = {np.degrees(bearing[sample_tid]):.1f}°")
print(f"  x_launch={f_launch['x_launch']:.1f}  y_launch={f_launch['y_launch']:.1f}")

# 5. Range sanity (in km)
for name, df in [("full_3d", full_3d), ("clean_3d", clean_3d)]:
    g = df.groupby("trajectory_id")
    end   = g[["x", "y"]].last()
    start = g[["x", "y"]].first()
    ranges = np.hypot(end["x"] - start["x"], end["y"] - start["y"]) / 1000
    print(f"\n[Range km {name}]  mean={ranges.mean():.1f}  min={ranges.min():.1f}  max={ranges.max():.1f}")

# 6. Verify bearing angles produce correct x/y split
sample_row = full_3d[full_3d["trajectory_id"] == sample_tid].iloc[10]
orig_row   = full [full ["trajectory_id"] == sample_tid].iloc[10]
theta_s    = bearing[sample_tid]
dx_s       = orig_row["x"] - launch.loc[sample_tid, "x_launch"]
expected_x = launch.loc[sample_tid, "x_launch"] + dx_s * np.cos(theta_s)
expected_y = launch.loc[sample_tid, "y_launch"] + dx_s * np.sin(theta_s)
print(f"\n[Rotation check]  expected x={expected_x:.3f}  got x={sample_row['x']:.3f}")
print(f"                  expected y={expected_y:.3f}  got y={sample_row['y']:.3f}")

# ── save ─────────────────────────────────────────────────────────────────────
full_3d .to_parquet(f"{DATA}/full_trajectories_3d.parquet",   index=False)
clean_3d.to_parquet(f"{DATA}/radar_marks_clean_3d.parquet",   index=False)
noisy_3d.to_parquet(f"{DATA}/radar_marks_noisy_3d.parquet",   index=False)

print("\n=== FILES SAVED ===")
for fname in ["full_trajectories_3d", "radar_marks_clean_3d", "radar_marks_noisy_3d"]:
    df = pd.read_parquet(f"{DATA}/{fname}.parquet")
    print(f"  {fname}.parquet  shape={df.shape}  vy_std={df['vy'].std():.1f}")
