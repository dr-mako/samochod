import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================
# ŚCIEŻKI
# ============================================================
BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-2")

TRAJ_PATH   = BASE / "synced_with_traj.csv"
OBS_PATH    = BASE / "out_lane_aruco_bev" / "summary_clean.csv"
CONFIG_PATH = BASE / "config_slam.json"
GT_PATH     = BASE / "out_lane_aruco_bev" / "ArUcoPos.csv"

# ============================================================
# LOAD CONFIG
# ============================================================
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

yaw_gain        = cfg["yaw_gain"]
scale_effective = cfg["scale_effective"]
curvature_bias  = cfg["curvature_bias"]
lateral_slip    = cfg["lateral_slip"]

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

STRIDE    = 11
MAX_NFEV  = 40
LAMBDA_POSE = 0.02

# ============================================================
# PREPROCESSING
# ============================================================
traj = pd.read_csv(TRAJ_PATH)
obs  = pd.read_csv(OBS_PATH)

# ---------------- TRAJ ----------------
traj = traj.sort_values("frame_id").reset_index(drop=True)
traj = traj.replace([np.inf, -np.inf], np.nan)
traj = traj.dropna(subset=["x", "y", "phi_rad"])
traj = traj.iloc[::STRIDE].reset_index(drop=True)

# ---------------- OBS ----------------
obs = obs.replace([np.inf, -np.inf], np.nan)
obs["frame_id"] = obs["frame"].str.extract(r"(\d+)").astype(int)

# AUTO OFFSET
frame_offset = traj["frame_id"].min() - obs["frame_id"].min()
obs["frame_id"] += frame_offset

print("\nApplied frame_offset =", frame_offset)

obs = obs.dropna(subset=["smoothed_id"])
obs["smoothed_id"] = obs["smoothed_id"].astype(int)

obs["marker_bev_x"] = pd.to_numeric(obs["marker_bev_x"], errors="coerce")
obs["marker_bev_y"] = pd.to_numeric(obs["marker_bev_y"], errors="coerce")

obs = obs.dropna(subset=["marker_bev_x", "marker_bev_y"])
obs = obs[obs["is_outlier"] == False]

# STRIDE-SAFE FILTER
valid_frames = set(traj["frame_id"])
obs = obs[obs["frame_id"].isin(valid_frames)]

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

    if scale_effective <= 0:
        return np.ones(10) * 1e6

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
# NORMALIZACJA PCA
# ============================================================
print("\n🔧 Normalizing map using PCA...")

centroid = lands_opt.mean(axis=0)

lands_centered = lands_opt - centroid
poses_centered = poses_opt.copy()
poses_centered[:, 0:2] -= centroid

U, S, Vt = np.linalg.svd(lands_centered, full_matrices=False)
R_pca = Vt.T

if np.linalg.det(R_pca) < 0:
    Vt[-1, :] *= -1
    R_pca = Vt.T

lands_norm = (R_pca.T @ lands_centered.T).T
poses_norm = poses_centered.copy()
poses_norm[:, 0:2] = (R_pca.T @ poses_centered[:, 0:2].T).T

yaw_rot = np.arctan2(R_pca[1,0], R_pca[0,0])
poses_norm[:, 2] = wrap_pi(poses_centered[:, 2] - yaw_rot)

print("✅ Map normalized")

# ============================================================
# SAVE
# ============================================================
np.save(BASE / "poses_runtime.npy", poses_opt)
np.save(BASE / "lands_runtime.npy", lands_opt)
np.save(BASE / "poses_norm.npy", poses_norm)
np.save(BASE / "lands_norm.npy", lands_norm)

print("\n✅ Saved:")
print("   poses_runtime.npy")
print("   lands_runtime.npy")
print("   poses_norm.npy")
print("   lands_norm.npy")

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

# ============================================================
# WYKRES TRAJEKTORII
# ============================================================
plt.figure(figsize=(8,8))
plt.plot(poses0[:,0], poses0[:,1], "--k", alpha=0.5, label="Odom")
plt.plot(poses_opt[:,0], poses_opt[:,1], "-r", label="SLAM runtime")
plt.scatter(lands_opt[:,0], lands_opt[:,1],
            c="blue", s=80, label="LM")
plt.axis("equal")
plt.grid(True)
plt.legend()

# ============================================================
# PLOT PCA MAP
# ============================================================
plt.figure(figsize=(8,8))
plt.plot(poses_norm[:,0], poses_norm[:,1], "-r", label="SLAM runtime (norm)")
plt.scatter(lands_norm[:,0], lands_norm[:,1],
            c="blue", s=80, label="LM (norm)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM Runtime (PCA normalized)")
#plt.show()

# ============================================================
# GT LANDMARKS → PCA FRAME
# ============================================================
print("\n🎯 Comparing SLAM vs GT in PCA frame...")

gt_df = pd.read_csv(GT_PATH)

if "smoothed_id" in gt_df.columns:
    gt_df = gt_df.set_index("smoothed_id")
elif "id" in gt_df.columns:
    gt_df = gt_df.set_index("id")
else:
    raise ValueError("GT CSV must contain 'id' or 'smoothed_id'")

# cm → m
gt_df[["x", "y"]] /= 100.0

gt_used = gt_df.loc[lm_ids]
B = gt_used[["x", "y"]].values

# --- centrowanie GT tym SAMYM centroidem ---
B_centered = B - centroid

# --- obrót GT tą SAMĄ macierzą PCA ---
B_norm = (R_pca.T @ B_centered.T).T

A_norm = lands_norm.copy()

# ============================================================
# RIGID ALIGNMENT W PCA
# ============================================================
REF_ID = lm_ids[0]
j_ref = lm_index[REF_ID]

A_shift = A_norm - A_norm[j_ref]
B_shift = B_norm - B_norm[j_ref]

U, _, Vt = np.linalg.svd(A_shift.T @ B_shift)
R_align = U @ Vt

if np.linalg.det(R_align) < 0:
    Vt[-1,:] *= -1
    R_align = U @ Vt

A_aligned = (R_align @ A_shift.T).T
B_aligned = B_shift

errors = np.linalg.norm(A_aligned - B_aligned, axis=1)

print("\n--- LANDMARK ERRORS (PCA frame) ---")
for lid, e in zip(lm_ids, errors):
    print(f"ID {lid:2d} → {e:.4f} m")

print("RMSE [m]:", np.sqrt(np.mean(errors**2)))
print("Mean [m]:", errors.mean())
print("Max  [m]:", errors.max())

# ============================================================
# GT LANDMARKS vs SLAM — centroid (0,0) + PCA axes aligned
# ============================================================
print("\n🎯 Comparing SLAM vs GT in aligned PCA frames (centroid @ 0,0)...")

gt_df = pd.read_csv(GT_PATH)

if "smoothed_id" in gt_df.columns:
    gt_df = gt_df.set_index("smoothed_id")
elif "id" in gt_df.columns:
    gt_df = gt_df.set_index("id")
else:
    raise ValueError("GT CSV must contain 'id' or 'smoothed_id'")

# cm → m
gt_df[["x", "y"]] /= 100.0

# Upewnij się, że porównujesz tylko wspólne ID
gt_ids = gt_df.index.to_numpy()
lm_set = set(lm_ids.tolist())
common_ids = np.array([i for i in gt_df.index.to_numpy() if i in lm_set], dtype=int)
if len(common_ids) < len(lm_ids):
    missing = set(lm_ids) - set(common_ids)
    print("⚠️ Missing GT ids:", sorted(list(missing)))

# A: SLAM landmarki w kolejności common_ids
A = np.vstack([lands_opt[lm_index[lid]] for lid in common_ids]).astype(float)

# B: GT landmarki w tej samej kolejności
B = gt_df.loc[common_ids, ["x", "y"]].values.astype(float)


def pca_frame(X: np.ndarray):
    """
    Zwraca:
      c: centroid (2,)
      R: macierz (2x2), taka że (X-c) @ R daje współrzędne w osiach PCA
      Xp: (X-c) @ R
    """
    c = X.mean(axis=0)
    Xc = X - c

    # PCA przez SVD (jak u Ciebie)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    R = Vt.T  # 2x2

    # prawoskrętny układ
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1

    Xp = Xc @ R
    return c, R, Xp


cA, RA, A_pca = pca_frame(A)
cB, RB, B_pca = pca_frame(B)

# B jest w swojej ramie PCA. Teraz dobieramy permutację osi i znaki,
# żeby osie PCA "patrzyły" tak samo jak w A (PCA ma niejednoznaczność znaku).
perms = [
    np.eye(2),
    np.array([[0.0, 1.0], [1.0, 0.0]]),  # swap osi
]
signs = [
    np.diag([sx, sy]).astype(float)
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
]

best_rmse = np.inf
best_C = None

for P in perms:
    for S in signs:
        C = P @ S
        B_try = B_pca @ C
        rmse = np.sqrt(np.mean(np.sum((A_pca - B_try) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_C = C

A_aligned = A_pca
B_aligned = B_pca @ best_C

# Kontrola: oba centroidy powinny być ~0,0
print("Centroid A_aligned:", A_aligned.mean(axis=0))
print("Centroid B_aligned:", B_aligned.mean(axis=0))
print("Chosen RMSE after PCA-axis alignment:", best_rmse)

errors = np.linalg.norm(A_aligned - B_aligned, axis=1)

print("\n--- LANDMARK ERRORS (aligned PCA, centroid @ 0) ---")
for lid, e in zip(common_ids, errors):
    print(f"ID {lid:2d} → {e:.4f} m")

print("RMSE [m]:", np.sqrt(np.mean(errors**2)))
print("Mean [m]:", errors.mean())
print("Max  [m]:", errors.max())

# ============================================================
# WYKRES SLAM vs GT (aligned PCA)
# ============================================================
plt.figure(figsize=(8, 8))

plt.scatter(
    B_aligned[:, 0],
    B_aligned[:, 1],
    c="red",
    marker="+",
    s=200,
    label="GT (PCA aligned, centroid@0)",
)

plt.scatter(
    A_aligned[:, 0],
    A_aligned[:, 1],
    c="blue",
    s=80,
    label="SLAM (PCA aligned, centroid@0)",
)

# wektory błędu
for i in range(len(A_aligned)):
    plt.plot(
        [A_aligned[i, 0], B_aligned[i, 0]],
        [A_aligned[i, 1], B_aligned[i, 1]],
        "gray",
        linewidth=1,
    )

# etykiety
for i, lid in enumerate(common_ids):
    plt.text(B_aligned[i, 0], B_aligned[i, 1], f" {lid}", color="red")
    plt.text(A_aligned[i, 0], A_aligned[i, 1], f" {lid}", color="blue")

plt.axhline(0, color="k", linewidth=0.5)
plt.axvline(0, color="k", linewidth=0.5)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM vs Ground Truth (centroid aligned + PCA axes aligned)")
#plt.show()

# ============================================================
# KLASYCZNY ALIGNMENT: Kabsch / Procrustes
# ============================================================
print("\n🧭 Kabsch/Procrustes alignment (classic map evaluation)...")

# A: SLAM (dopasowane po ID), B: GT (dopasowane po ID)
# Zakładam, że masz już:
# - gt_df (w metrach)
# - common_ids
# - lm_index, lands_opt
A = np.vstack([lands_opt[lm_index[lid]] for lid in common_ids]).astype(float)
B = gt_df.loc[common_ids, ["x", "y"]].values.astype(float)


def kabsch_2d(A: np.ndarray, B: np.ndarray, allow_scale: bool = False):
    """
    Dopasowuje B do A: minimalizuje ||A - (s * B R + t)||.
    Zwraca:
      s (float), R (2x2), t (2,)
      B_fit: B po dopasowaniu do ramy A
    """
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    Ac = A - cA
    Bc = B - cB

    H = Bc.T @ Ac
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    if allow_scale:
        denom = np.sum(Bc**2)
        if denom < 1e-12:
            s = 1.0
        else:
            s = np.sum(S) / denom
    else:
        s = 1.0

    t = cA - s * (cB @ R)
    B_fit = s * (B @ R) + t
    return s, R, t, B_fit


# Rigid (bez skali)
s_r, R_r, t_r, B_rigid = kabsch_2d(A, B, allow_scale=False)
err_r = np.linalg.norm(A - B_rigid, axis=1)

print("\n--- KABSCH (rigid, no scale) ---")
print("s:", s_r)
print("t:", t_r)
print("RMSE [m]:", np.sqrt(np.mean(err_r**2)))
print("Mean [m]:", err_r.mean())
print("Max  [m]:", err_r.max())

# Similarity Procrustes (ze skalą) — czasem przydatne, gdy skala w BEV/SLAM pływa
s_s, R_s, t_s, B_sim = kabsch_2d(A, B, allow_scale=True)
err_s = np.linalg.norm(A - B_sim, axis=1)

print("\n--- PROCRUSTES (similarity, with scale) ---")
print("s:", s_s)
print("t:", t_s)
print("RMSE [m]:", np.sqrt(np.mean(err_s**2)))
print("Mean [m]:", err_s.mean())
print("Max  [m]:", err_s.max())

# Wykres: rigid (najczęściej najbardziej 'uczciwy' do oceny)
plt.figure(figsize=(8, 8))

plt.scatter(
    A[:, 0],
    A[:, 1],
    c="blue",
    s=80,
    label="SLAM LM",
)

plt.scatter(
    B_rigid[:, 0],
    B_rigid[:, 1],
    c="red",
    marker="+",
    s=200,
    label="GT fitted (Kabsch rigid)",
)

for i in range(len(A)):
    plt.plot(
        [A[i, 0], B_rigid[i, 0]],
        [A[i, 1], B_rigid[i, 1]],
        "gray",
        linewidth=1,
    )

for i, lid in enumerate(common_ids):
    plt.text(A[i, 0], A[i, 1], f" {lid}", color="blue")
    plt.text(B_rigid[i, 0], B_rigid[i, 1], f" {lid}", color="red")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM vs GT (Kabsch rigid alignment)")
plt.show()