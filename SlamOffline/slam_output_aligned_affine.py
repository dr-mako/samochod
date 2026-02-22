import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ŚCIEŻKI
# ============================================================
BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-17-5")

SLAM_LAND_PATH = BASE / "lands_runtime.npy"   # ← zapisany lands_opt
# jeśli nie masz .npy → można wczytać z csv

# ============================================================
# Wczytanie landmarków SLAM
# ============================================================
lands_opt = np.load(SLAM_LAND_PATH)

# Jeśli zapisujesz jako CSV:
# lands_opt = pd.read_csv(BASE/"lands_runtime.csv")[["x","y"]].values

# ============================================================
# GT LANDMARKS
# ============================================================
gt_raw = {
    0: [0, 0], 1: [68, 0], 2: [121.5, 0], 3: [154, -32],
    4: [154, -86], 5: [121.5, -117], 6: [95, -86],
    7: [68, -59], 8: [41, -117], 9: [0, -117],
    11: [-22, -86], 10: [-22, -32]
}

SCALE = 0.01
gt_df = pd.DataFrame.from_dict(gt_raw, orient="index",
                               columns=["x", "y"]) * SCALE

# ============================================================
# ID landmarków (ważne!)
# ============================================================
# MUSI odpowiadać kolejności SLAM
lm_ids = np.array(sorted(gt_df.index))[:len(lands_opt)]

B = gt_df.loc[lm_ids][["x", "y"]].values
A = lands_opt.copy()

print(f"N landmarks used = {len(A)}")

# ============================================================
# AFFINE ALIGNMENT
# ============================================================
# A → SLAM, B → GT

A_aug = np.hstack([A, np.ones((len(A), 1))])

# solve A_aug * X = B
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
for lid, e in zip(lm_ids, errors):
    print(f"ID {lid:2d} → {e:.4f} m")

print("\nRMSE [m]:", np.sqrt(np.mean(errors**2)))
print("Mean [m]:", errors.mean())
print("Max  [m]:", errors.max())

# ============================================================
# WYKRES
# ============================================================
plt.figure(figsize=(8,8))

plt.scatter(B[:,0], B[:,1],
            c="red", marker="+", s=200, label="GT")

plt.scatter(A_affine[:,0], A_affine[:,1],
            c="blue", s=80, label="SLAM (affine)")

# wektory błędu
for i in range(len(A)):
    plt.plot([A_affine[i,0], B[i,0]],
             [A_affine[i,1], B[i,1]],
             "gray", linewidth=1)

# etykiety
for i, lid in enumerate(lm_ids):
    plt.text(B[i,0], B[i,1], f" {lid}", color="red")
    plt.text(A_affine[i,0], A_affine[i,1], f" {lid}", color="blue")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("SLAM vs Ground Truth (Affine aligned)")
plt.show()

# ============================================================
# OPCJONALNY ZAPIS
# ============================================================
SAVE = True

if SAVE:
    np.save(BASE / "lands_affine_aligned.npy", A_affine)
    print("\n✅ Saved → lands_affine_aligned.npy")