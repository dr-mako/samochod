import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================
# ŚCIEŻKI
# ============================================================
BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-17-5")

TRAJ_PATH = BASE / "synced_with_traj.csv"
OBS_PATH  = BASE / "out_lane_aruco_bev" / "summary_clean.csv"
CONFIG_PATH = BASE / "config_slam.json"

# ============================================================
# LOAD CONFIG
# ============================================================
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

yaw_gain       = cfg["yaw_gain"]
scale_effective = cfg["scale_effective"]
curvature_bias = cfg["curvature_bias"]
lateral_slip   = cfg["lateral_slip"]

print("\n📦 CONFIG LOADED:")
print(json.dumps(cfg, indent=4))

# ============================================================
# PARAMETRY BEV
# ============================================================
Gx = 412.0
h  = 420.0

Dx = 0.11125
Dy = 0.009
yaw_offset = np.deg2rad(2.2125)

# ============================================================
# PARAMETRY SLAM
# ============================================================
SIGMA_XY  = 0.05
SIGMA_PHI = 0.02

W_ODOM = 1.0
W_LAND = 1.0

STRIDE = 10
MAX_NFEV = 100
LAMBDA_POSE = 0.02

# ============================================================
# PREPROCESSING
# ============================================================
traj = pd.read_csv(TRAJ_PATH)
obs  = pd.read_csv(OBS_PATH)

traj = traj.sort_values("frame_id").reset_index(drop=True)
traj = traj.replace([np.inf, -np.inf], np.nan)
traj = traj.dropna(subset=["x", "y", "phi_rad"])
traj = traj.iloc[::STRIDE].reset_index(drop=True)

obs = obs.replace([np.inf, -np.inf], np.nan)
obs["frame_id"] = obs["frame"].str.extract(r"(\d+)").astype(int)

obs = obs.dropna(subset=["smoothed_id"])
obs["smoothed_id"] = obs["smoothed_id"].astype(int)

obs["marker_bev_x"] = pd.to_numeric(obs["marker_bev_x"], errors="coerce")
obs["marker_bev_y"] = pd.to_numeric(obs["marker_bev_y"], errors="coerce")

obs = obs.dropna(subset=["marker_bev_x", "marker_bev_y"])
obs = obs[obs["frame_id"] % STRIDE == 0]
obs = obs[obs["is_outlier"] == False]

# ============================================================
# MAPOWANIA
# ============================================================
frame_to_idx = {fid: i for i, fid in enumerate(traj["frame_id"])}

lm_ids = np.sort(obs["smoothed_id"].unique())
lm_index = {lid: i for i, lid in enumerate(lm_ids)}

poses0 = traj[["x", "y", "phi_rad"]].to_numpy()

N = len(traj)
M = len(lm_ids)

print(f"\nN poses = {N}, M landmarks = {M}")

# ============================================================
# FUNKCJE
# ============================================================
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def bev_pix_to_metric(x_pix, y_pix):
    v = -(x_pix - Gx) * scale_effective
    u = -(y_pix - h)  * scale_effective
    return np.array([u, v])

# ============================================================
# STAN POCZĄTKOWY
# ============================================================
land0 = np.zeros((M, 2))

x0 = np.hstack([
    poses0.reshape(-1),
    land0.reshape(-1)
])

pose_anchor = poses0[0].copy()

# ============================================================
# RESIDUALS
# ============================================================
def residuals(x):

    poses = x[:3*N].reshape(N, 3)
    lands = x[3*N:].reshape(M, 2)

    res = []

    # ---------------- ODOM ----------------
    for i in range(N - 1):

        dp = poses[i+1] - poses[i]
        dp[2] = wrap_pi(dp[2])

        dp0 = poses0[i+1] - poses0[i]
        dp0[2] = wrap_pi(dp0[2])

        dp0[2] *= yaw_gain
        dp0[2] *= curvature_bias

        err = dp - dp0
        err[2] = wrap_pi(err[2])

        res.extend(W_ODOM * np.array([
            err[0] / SIGMA_XY,
            err[1] / SIGMA_XY,
            err[2] / SIGMA_PHI
        ]))

    # ---------------- LANDMARK OBS ----------------
    for _, row in obs.iterrows():

        fid = int(row["frame_id"])
        lid = row["smoothed_id"]

        if fid not in frame_to_idx:
            continue

        i = frame_to_idx[fid]
        j = lm_index[lid]

        z_bev = bev_pix_to_metric(row["marker_bev_x"],
                                  row["marker_bev_y"])

        z_corr = z_bev + np.array([Dx, Dy])

        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        R_yaw = np.array([[c, -s], [s, c]])
        z_veh = R_yaw @ z_corr

        z_veh[1] += lateral_slip * z_veh[0]

        th = poses[i, 2]
        c, s = np.cos(th), np.sin(th)

        pred = poses[i, 0:2] + np.array([
            c*z_veh[0] - s*z_veh[1],
            s*z_veh[0] + c*z_veh[1]
        ])

        err = pred - lands[j]

        res.extend(W_LAND * np.array([
            err[0] / SIGMA_XY,
            err[1] / SIGMA_XY
        ]))

    # ---------------- ANCHOR ----------------
    res.extend([
        (poses[0,0] - pose_anchor[0]) / SIGMA_XY,
        (poses[0,1] - pose_anchor[1]) / SIGMA_XY,
        wrap_pi(poses[0,2] - pose_anchor[2]) / SIGMA_PHI
    ])

    # ---------------- STIFFNESS ----------------
    diff = poses[:,0:2] - poses0[:,0:2]
    res.extend((LAMBDA_POSE * diff / SIGMA_XY).reshape(-1))

    return np.array(res)

# ============================================================
# SOLVER
# ============================================================
print("\nStart RUNTIME SLAM (bias frozen)...")

result = least_squares(
    residuals,
    x0,
    method="trf",
    loss="huber",
    verbose=2,
    max_nfev=MAX_NFEV
)

# ============================================================
# WYNIKI
# ============================================================
poses_opt = result.x[:3*N].reshape(N, 3)
lands_opt = result.x[3*N:].reshape(M, 2)


# ============================================================
# WYNIKI
# ============================================================
poses_opt = result.x[:3*N].reshape(N, 3)
lands_opt = result.x[3*N:].reshape(M, 2)

# ============================================================
# SAVE OUTPUTS
# ============================================================
np.save(BASE / "poses_runtime.npy", poses_opt)
np.save(BASE / "lands_runtime.npy", lands_opt)

print("✅ Saved:")
print("   poses_runtime.npy")
print("   lands_runtime.npy")

# ============================================================
# RAPORT
# ============================================================
deformation = np.linalg.norm(
    poses_opt[:,0:2] - poses0[:,0:2],
    axis=1
)

print("\n--- RAPORT RUNTIME ---")
print("Final cost:", np.sum(residuals(result.x)**2))
print("Solver:", result.message)
print("Mean deformation [m]:", deformation.mean())
print("Max deformation  [m]:", deformation.max())

lm_spread = np.std(lands_opt, axis=0)
print("Landmark spread σ [m]:", lm_spread)

# ============================================================
# WYKRES
# ============================================================
plt.figure(figsize=(8,8))
plt.plot(poses0[:,0], poses0[:,1], "--k", alpha=0.5, label="Odom")
plt.plot(poses_opt[:,0], poses_opt[:,1], "-r", label="SLAM runtime")
plt.scatter(lands_opt[:,0], lands_opt[:,1],
            c="blue", s=80, label="LM")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM Runtime (Bias Frozen)")
plt.show()

# ============================================================
# GT LANDMARKS (diagnostyka ONLY)
# ============================================================
gt_raw = {
    0: [0, 0], 1: [68, 0], 2: [121.5, 0], 3: [154, -32],
    4: [154, -86], 5: [121.5, -117], 6: [95, -86],
    7: [68, -59], 8: [41, -117], 9: [0, -117],
    11: [-22, -86], 10: [-22, -32]
}
SCALE = 0.01
gt_df = pd.DataFrame.from_dict(gt_raw, orient='index',
                               columns=['x', 'y']) * SCALE

gt_used = gt_df.loc[lm_ids]

A = lands_opt.copy()                 # SLAM
B = gt_used[["x", "y"]].values       # GT

# ============================================================
# ALIGNMENT → LM1 = (0,0)
# ============================================================
REF_ID = 1

if REF_ID not in lm_ids:
    print(f"⚠ REF_ID {REF_ID} not present → centroid fallback")
    A_shift = A - A.mean(axis=0)
    B_shift = B - B.mean(axis=0)
else:
    print(f"\n🎯 Alignment using landmark {REF_ID}")
    j_ref = lm_index[REF_ID]

    A_shift = A - A[j_ref]
    B_shift = B - B[j_ref]

# ============================================================
# ROTATION → align to OX via SVD
# ============================================================
U, _, Vt = np.linalg.svd(A_shift.T @ B_shift)
R = U @ Vt

# wymuszenie proper rotation
if np.linalg.det(R) < 0:
    Vt[-1,:] *= -1
    R = U @ Vt

A_aligned = (R @ A_shift.T).T
B_aligned = B_shift

# ============================================================
# ERRORS
# ============================================================
errors = np.linalg.norm(A_aligned - B_aligned, axis=1)

print("\n--- LANDMARK ERRORS (LM1 aligned) ---")
for lid, e in zip(lm_ids, errors):
    print(f"ID {lid:2d} → {e:.4f} m")

print("RMSE [m]:", np.sqrt(np.mean(errors**2)))
print("Mean [m]:", errors.mean())
print("Max  [m]:", errors.max())

# ============================================================
# WYKRES
# ============================================================
plt.figure(figsize=(8,8))

plt.scatter(B_aligned[:,0], B_aligned[:,1],
            c="red", marker="+", s=200, label="GT")

plt.scatter(A_aligned[:,0], A_aligned[:,1],
            c="blue", s=80, label="SLAM")

# wektory błędu
for i in range(len(A_aligned)):
    plt.plot([A_aligned[i,0], B_aligned[i,0]],
             [A_aligned[i,1], B_aligned[i,1]],
             "gray", linewidth=1)

# etykiety
for i, lid in enumerate(lm_ids):
    plt.text(B_aligned[i,0], B_aligned[i,1], f" {lid}", color="red")
    plt.text(A_aligned[i,0], A_aligned[i,1], f" {lid}", color="blue")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM vs Ground Truth (LM1 aligned)")
plt.show()


# ============================================================
# WSPÓLNY WYKRES – SUBPLOTS POZIOMO
# ============================================================
plt.figure(figsize=(16, 8))

# ----------------------------
# Subplot 1 – trajektorie SLAM
# ----------------------------
plt.subplot(1, 2, 1)

plt.plot(poses0[:,0], poses0[:,1], "--k", alpha=0.5, label="Odom")
plt.plot(poses_opt[:,0], poses_opt[:,1], "-r", label="SLAM runtime")
plt.scatter(lands_opt[:,0], lands_opt[:,1],
            c="blue", s=80, label="LM")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM Runtime (Bias Frozen)")

# ----------------------------
# Subplot 2 – porównanie LM
# ----------------------------
plt.subplot(1, 2, 2)

plt.scatter(B_aligned[:,0], B_aligned[:,1],
            c="red", marker="+", s=200, label="GT")

plt.scatter(A_aligned[:,0], A_aligned[:,1],
            c="blue", s=80, label="SLAM")

# wektory błędu
for i in range(len(A_aligned)):
    plt.plot([A_aligned[i,0], B_aligned[i,0]],
             [A_aligned[i,1], B_aligned[i,1]],
             "gray", linewidth=1)

# etykiety
for i, lid in enumerate(lm_ids):
    plt.text(B_aligned[i,0], B_aligned[i,1], f" {lid}", color="red")
    plt.text(A_aligned[i,0], A_aligned[i,1], f" {lid}", color="blue")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM vs Ground Truth (LM1 aligned)")

plt.tight_layout()
plt.show()