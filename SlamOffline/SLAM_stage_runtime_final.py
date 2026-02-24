import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ŚCIEŻKI
# ============================================================
BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-1")

TRAJ_PATH = BASE / "synced_with_traj.csv"
OBS_PATH  = BASE / "out_lane_aruco_bev" / "summary_clean.csv"

# ============================================================
# PARAMETRY BEV
# ============================================================
Gx = 412.0
h  = 420.0
scale = 1 / 1000.0  # px → m

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

# opcjonalna "stiffness"
LAMBDA_POSE = 0.02

# ============================================================
# 1️⃣ PREPROCESSING
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
# 2️⃣ MAPOWANIA
# ============================================================
frame_to_idx = {fid: i for i, fid in enumerate(traj["frame_id"])}

lm_ids = np.sort(obs["smoothed_id"].unique())
lm_index = {lid: i for i, lid in enumerate(lm_ids)}

N = len(traj)
M = len(lm_ids)

print(f"N poses = {N}, M landmarks = {M}")

# ============================================================
# FUNKCJE
# ============================================================
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def bev_pix_to_metric(x_pix, y_pix):
    v = -(x_pix - Gx) * scale
    u = -(y_pix - h)  * scale
    return np.array([u, v])

# ============================================================
# 3️⃣ STAN POCZĄTKOWY
# ============================================================
poses0 = traj[["x", "y", "phi_rad"]].to_numpy()
land0  = np.zeros((M, 2))

x0 = np.hstack([poses0.reshape(-1), land0.reshape(-1)])
pose_anchor = poses0[0].copy()

# ============================================================
# 4️⃣ RESIDUALS
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
        if lid not in lm_index:
            continue

        i = frame_to_idx[fid]
        j = lm_index[lid]

        z_bev = bev_pix_to_metric(row["marker_bev_x"],
                                  row["marker_bev_y"])

        z_corr = z_bev + np.array([Dx, Dy])

        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        R_yaw = np.array([[c, -s], [s, c]])
        z_veh = R_yaw @ z_corr

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
    if LAMBDA_POSE > 0:
        diff = poses[:,0:2] - poses0[:,0:2]
        res.extend((LAMBDA_POSE * diff / SIGMA_XY).reshape(-1))

    return np.array(res)

# ============================================================
# DIAGNOSTYKA STARTOWA
# ============================================================
r0 = residuals(x0)

print("\n--- DIAGNOSTYKA START ---")
print("Residual finite:", np.all(np.isfinite(r0)))
print("Initial cost:", np.sum(r0**2))

# ============================================================
# 5️⃣ SOLVER
# ============================================================
print("\nStart SLAM runtime FINAL...")

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
# 📊 RAPORT
# ============================================================
deformation = np.linalg.norm(
    poses_opt[:,0:2] - poses0[:,0:2],
    axis=1
)

print("\n--- RAPORT SLAM ---")
print("Final cost:", np.sum(residuals(result.x)**2))
print("Solver:", result.message)
print("Mean deformation [m]:", deformation.mean())
print("Max deformation  [m]:", deformation.max())

lm_spread = np.std(lands_opt, axis=0)
print("Landmark spread σ [m]:", lm_spread)

# ============================================================
# 📈 WIZUALIZACJA
# ============================================================
plt.figure(figsize=(8,8))
plt.plot(poses0[:,0], poses0[:,1], "--k", alpha=0.5, label="Odom")
plt.plot(poses_opt[:,0], poses_opt[:,1], "-r", label="SLAM")
plt.scatter(lands_opt[:,0], lands_opt[:,1],
            c="blue", s=80, label="LM")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM Runtime FINAL")
plt.show()