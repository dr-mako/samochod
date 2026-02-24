import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ŚCIEŻKI
# ============================================================
BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-1")

SLAM_LAND_PATH = BASE / "lands_runtime.npy"
GT_PATH     = BASE / "out_lane_aruco_bev" /"ArUcoPos.csv"

# ============================================================
# LOAD SLAM LANDMARKS
# ============================================================
lands_opt = np.load(SLAM_LAND_PATH)

# ============================================================
# LOAD GT LANDMARKS (CSV)
# ============================================================
gt_df = pd.read_csv(GT_PATH, sep=None, engine="python", decimal=".")

if "smoothed_id" in gt_df.columns:
    gt_df = gt_df.set_index("smoothed_id")
elif "id" in gt_df.columns:
    gt_df = gt_df.set_index("id")
else:
    raise ValueError("GT CSV must contain 'id' or 'smoothed_id' column")

# cm → m
gt_df[["x", "y"]] /= 100.0

print("\n--- GT CSV ---")
print(gt_df.head())

# ============================================================
# LOAD lm_ids FROM SLAM (BEST OPTION)
# ============================================================
LM_IDS_PATH = BASE / "lm_ids_runtime.npy"

if LM_IDS_PATH.exists():
    lm_ids = np.load(LM_IDS_PATH)
    print("\n✅ Loaded lm_ids from runtime")
else:
    print("\n⚠ lm_ids_runtime.npy not found → reconstructing from GT")
    lm_ids = np.array(sorted(gt_df.index))[:len(lands_opt)]

# ============================================================
# COMMON LANDMARKS ONLY
# ============================================================
common_ids = np.intersect1d(lm_ids, gt_df.index)

if len(common_ids) == 0:
    raise ValueError("❌ No common landmark IDs between SLAM and GT")

print(f"\nCommon landmarks: {len(common_ids)}")

# indeksy landmarków w SLAM
slam_idx = [np.where(lm_ids == lid)[0][0] for lid in common_ids]

A = lands_opt[slam_idx]                        # SLAM
B = gt_df.loc[common_ids][["x", "y"]].values   # GT

# ============================================================
# AFFINE ALIGNMENT
# ============================================================
A_aug = np.hstack([A, np.ones((len(A), 1))])

X, *_ = np.linalg.lstsq(A_aug, B, rcond=None)

M = X[:2, :].T
t = X[2, :]

A_affine = (M @ A.T).T + t

print("\n💥 Affine alignment SLAM → GT")
print("Estimated affine M:")
print(M)
print("Estimated translation t:")
print(t)

# ============================================================
# ERRORS
# ============================================================
errors = np.linalg.norm(A_affine - B, axis=1)

print("\n--- LANDMARK ERRORS (Affine aligned) ---")
for lid, e in zip(common_ids, errors):
    print(f"ID {lid:2d} → {e:.4f} m")

print("\nRMSE [m]:", np.sqrt(np.mean(errors**2)))
print("Mean [m]:", errors.mean())
print("Max  [m]:", errors.max())

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(8,8))

plt.scatter(B[:,0], B[:,1],
            c="red", marker="+", s=200, label="GT")

plt.scatter(A_affine[:,0], A_affine[:,1],
            c="blue", s=80, label="SLAM (affine)")

# error vectors
for i in range(len(A)):
    plt.plot([A_affine[i,0], B[i,0]],
             [A_affine[i,1], B[i,1]],
             "gray", linewidth=1)

# labels
for i, lid in enumerate(common_ids):
    plt.text(B[i,0], B[i,1], f" {lid}", color="red")
    plt.text(A_affine[i,0], A_affine[i,1], f" {lid}", color="blue")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM vs Ground Truth (Affine aligned)")
plt.show()

# ============================================================
# SAVE OPTIONAL
# ============================================================
SAVE = True

if SAVE:
    np.save(BASE / "lands_affine_aligned.npy", A_affine)
    print("\n✅ Saved → lands_affine_aligned.npy")