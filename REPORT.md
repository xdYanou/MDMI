# Assignment 1 — Report

## Dataset

**Source:** Synthetic missile trajectory simulation dataset generated for this assignment.

**Files used for training:**
| File | Size | Rows | Description |
|---|---|---|---|
| `radar_marks_clean_3d.parquet` | 2.2 MB | 34,993 | 7 radar pings per trajectory (clean) |
| `radar_marks_noisy_3d.parquet` | 2.5 MB | 34,993 | Same pings with Gaussian noise (~90 m σ) |

**Full trajectory files (`full_trajectories_3d.parquet`, 210 MB) are not committed** — they are used only to extract landing-point labels (5-column targets file, < 1 MB, committed as `models/targets.parquet`).

**Effective training set:** 4,999 trajectories → 4,999 rows × 55 features after pivoting.
Two classes: ballistic shells (`is_rocket=False`, ~50%) and powered rockets (`is_rocket=True`, ~50%).

**3D bearing transformation:** The raw dataset had all motion along the x-axis (`vy=0` always). A random horizontal bearing angle θ was applied per trajectory, splitting ground-track motion into genuine x and y components while leaving z, vz, and total speed unchanged. See `data/add_bearing.py` for implementation and sanity checks.

---

## Business Framing

**Scenario:** A ground-based air-defence radar detects an incoming object. It obtains exactly **7 position/velocity pings over a ~36-second window**, then loses track. The system must immediately answer two questions:

1. **Where will it land?** `(x_end, y_end)` in metres — to position interceptors or issue evacuation orders.
2. **How much time remains?** `t_remaining` in seconds — to determine intercept feasibility.

**Why this drives pipeline decisions:**

| Decision | Reasoning |
|---|---|
| **Target: regression, not classification** | The operationally critical output is a continuous landing coordinate, not a label. |
| **Random train/test split is valid** | Trajectories are physically independent — no trajectory's outcome depends on another's. There is no temporal ordering between trajectories (they are parallel simulations). A time-based split would be wrong here. |
| **Metric: RMSE 2D (metres)** | A 3-km miss is operationally different from a 15-km miss. MSE/RMSE in metric units is interpretable to a domain expert. R² is reported for method comparison only. |
| **Baseline: training-mean predictor** | The simplest possible model. Any ML model that fails to significantly beat it has learned nothing. |
| **No threshold slider** | This is regression, not classification. Threshold decisions (e.g. intercept / no-intercept) would require a downstream cost model not specified in the problem. |

---

## Feature Engineering

Starting point: 7 radar pings × 7 variables (`x, y, z, vx, vy, vz, v`) = **49 raw features** per trajectory (pivoted wide).

### Engineered features (6 added → 55 total)

| Feature | Formula | Physical rationale |
|---|---|---|
| `h_speed_last` | `√(vx_p6² + vy_p6²)` | Horizontal speed at last ping directly determines how far the object can travel before impact |
| `h_range_obs` | `√((x_p6−x_p0)² + (y_p6−y_p0)²)` | Distance covered during radar window — proxy for total range |
| `est_apogee_km` | `z_p6 + vz_p6²/(2g)` if ascending, else `z_p6` | Kinematic estimate of max altitude; ballistics peak early, rockets later |
| `v_slope` | Linear fit slope of `v` over 7 pings | Rockets accelerate (positive slope), ballistics decelerate (negative) |
| `bearing_rad` | `arctan2(vy_p6, vx_p6)` | Direction of travel — combined with last position, determines landing quadrant |

**What didn't work / was dropped:**
- Time features (`t_first`, `t_last`) were tried but added minimal R² lift because ping spacing is fixed (6 s) and first-ping time is already embedded in the position features.
- Interaction terms (e.g. `vx × vz`) were considered but not added — the linear model already achieves R² ≈ 0.986 without them, and adding polynomial features would undermine the "simple model" pedagogical goal.
- The `v` column is mathematically redundant (`v = √(vx²+vy²+vz²)`) but retained in the raw features because it provides a clean combined-speed signal without requiring the model to learn the Pythagorean relationship.

### Preprocessing
- `StandardScaler` fitted on training set only, applied to test set (no leakage).
- Targets (`x_end`, `y_end`, `t_remaining`) are also scaled for PyTorch training stability, then inverse-transformed for evaluation.

---

## Split Strategy

```
train_test_split(X, y, test_size=0.2, random_state=42)
→ Train: 3,999 trajectories   Test: 1,000 trajectories
```

**Important:** the raw radar file has 7 rows per trajectory (one per ping). The 7 pings must be consumed in order — ping 0 through ping 6 — and must never be split across train and test. The pipeline handles this by pivoting all 7 pings into a single row per trajectory **before** any split:

```
radar_marks (34,993 rows, 7 per trajectory)
    → pivot on ping_idx  →  feature matrix X (4,999 rows × 55 features)
    → train_test_split on X  →  train (3,999 rows) / test (1,000 rows)
```

By the time `train_test_split` is called, the unit is one complete trajectory. The sequential order of pings is preserved in the feature names (`x_p0 → x_p6`, `vz_p0 → vz_p6`, etc.). A trajectory's pings are never separated across the split.

**Why random (not time-based) split:** the trajectories are independent parallel simulations — no trajectory's outcome depends on another's, and there is no meaningful time ordering between trajectory IDs. A time-based split would not prevent leakage here; it would simply discard training data without benefit.

---

## Three-Method Comparison

All three methods implement the same model: **multi-output Ridge regression** (L2 regularisation, α=1.0) predicting `(x_end, y_end, t_remaining)`.

### Metrics on test set (n=1,000)

| Model | RMSE 2D (m) | RMSE Δt (s) | R² x | R² y | R² Δt |
|---|---|---|---|---|---|
| **Baseline (mean)** | 34,318 | — | −0.001 | 0.000 | −0.002 |
| **sklearn Ridge** | 4,102 | 2.4 s | 0.9862 | 0.9852 | 0.9843 |
| **PyTorch Scratch (SGD)** | 4,582 | 2.6 s | 0.9828 | 0.9815 | 0.9830 |
| **PyTorch nn.Module (Adam)** | 4,141 | 2.4 s | 0.9860 | 0.9848 | 0.9844 |

### Why they differ

All three implement the same linear model — the gap is **optimisation, not model capacity**.

- **sklearn Ridge** solves the normal equations analytically: `W* = (XᵀX + αI)⁻¹ Xᵀy`. This is the exact global minimum and the reference solution.
- **nn.Module + Adam** uses adaptive per-parameter learning rates. It converges faster than plain SGD but does not reach the exact OLS solution after a fixed number of epochs (3,000 here). Gap: ~40 m RMSE 2D vs sklearn.
- **From-scratch SGD** uses a fixed learning rate (0.05) with no momentum or adaptation. It converges most slowly and shows the largest gap (~480 m). The loss curve in Tab 4 shows it is still decreasing at epoch 3,000.
- **Given unlimited epochs**, all three converge to the same weights. The gap is the pedagogical point: closed-form vs gradient-based optimisation.

---

## Limitations & Honest Failures

1. **Synthetic data:** The dataset was generated, not collected from a real radar system. Physical parameters (drag, thrust profile, atmospheric conditions) are simplified. The high R² (0.986) partly reflects this regularity — a real radar dataset would be messier.

2. **Linear model ceiling:** With R² ≈ 0.986, most of the remaining error (RMSE ≈ 4 km) likely comes from rockets in their powered phase, where non-linear thrust effects make the trajectory unpredictable from a linear model. A non-linear model (MLP, Kalman filter) would reduce this but is outside the scope of this assignment.

3. **Radar window timing:** The radar window can start anywhere from t=0 to t=128 s. A trajectory observed early (ascending phase) is much harder to predict than one observed late (descending toward impact). The model does not explicitly condition on this — a feature like `fraction_of_flight_observed` might improve rockets specifically.

4. **From-scratch convergence:** The SGD implementation has not fully converged at 3,000 epochs. Running to 10,000+ epochs would close most of the gap with sklearn. This is documented rather than hidden.

5. **y_end is nearly trivially predictable** for trajectories with large lateral offsets (y_start is already far from 0), because the model can simply learn `y_end ≈ y_start`. The truly hard prediction is `x_end` (the down-range component).
