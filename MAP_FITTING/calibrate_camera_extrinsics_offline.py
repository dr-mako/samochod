import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ============================================================
# ŚCIEŻKI
# ============================================================

BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-2")

TRAJ_PATH = BASE / "synced_with_traj.csv"
LAND_PATH = BASE / "out_lane_aruco_bev" / "summary_clean.csv"

# ============================================================
# PARAMETRY BEV
# ============================================================

IMAGE_H_PX = 420
U0_PX = 412
M_PER_PX = 1.0 / 1000.0

# ============================================================
# ZAKRES TESTU
# ============================================================

Dx_values = np.linspace(0.1, 0.5, 81)
Dy_values = np.linspace(-0.75, 0.5, 81)
yaw_values = np.linspace(-2*np.pi/180,
                         2*np.pi/180,
                         81)

MIN_OBS = 10

# ============================================================
# WCZYTANIE
# ============================================================

traj = pd.read_csv(TRAJ_PATH)
land = pd.read_csv(LAND_PATH)

land["frame_id"] = (
    land["frame"]
    .str.extract(r"(\d+)")
    .astype(int)
)

df = pd.merge(
    traj[["frame_id", "x", "y", "phi_rad"]],
    land[["frame_id", "smoothed_id",
          "marker_bev_x", "marker_bev_y"]],
    on="frame_id",
    how="inner",
).dropna(subset=["marker_bev_x", "marker_bev_y"])

# ============================================================
# ✅ WYCIĘCIE DRUGIEGO PRZEJAZDU NA PODSTAWIE PRZERWY CZASOWEJ
# ============================================================

df = df.sort_values("frame_id")

max_gap = 50   # liczba klatek uznana za "dużą przerwę"

last_seen = {}
rows = []

for _, row in df.iterrows():

    lid = int(row["smoothed_id"])
    frame = int(row["frame_id"])

    if lid in last_seen:
        gap = frame - last_seen[lid]

        if gap > max_gap:
            print("Drugi przejazd wykryty dla ID:", lid)
            break

    last_seen[lid] = frame
    rows.append(row)

df = pd.DataFrame(rows)

print("Obserwacji w pierwszym przejeździe:", len(df))

# ============================================================
# KONWERSJA NA NUMPY
# ============================================================

xC = df["x"].values
yC = df["y"].values
phi = df["phi_rad"].values
ids_raw = df["smoothed_id"].values.astype(int)

unique_ids, ids = np.unique(ids_raw, return_inverse=True)
K = len(unique_ids)

u_loc = (IMAGE_H_PX - df["marker_bev_y"].values) * M_PER_PX
v_loc = (U0_PX - df["marker_bev_x"].values) * M_PER_PX

# ============================================================
# FUNKCJA CELU (RMS do centroidu)
# ============================================================

def compute_score(Dx, Dy, yaw):

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    xG = xC + Dx*cos_phi - Dy*sin_phi
    yG = yC + Dx*sin_phi + Dy*cos_phi

    c = np.cos(-(phi + yaw))
    s = np.sin(-(phi + yaw))

    x_w = xG + (u_loc*c + v_loc*s)
    y_w = yG + (-u_loc*s + v_loc*c)

    count = np.bincount(ids, minlength=K)
    sum_x = np.bincount(ids, weights=x_w, minlength=K)
    sum_y = np.bincount(ids, weights=y_w, minlength=K)

    valid = count >= MIN_OBS
    if not np.any(valid):
        return np.inf

    cx = sum_x / count
    cy = sum_y / count

    dx = x_w - cx[ids]
    dy = y_w - cy[ids]

    dist2 = dx**2 + dy**2

    sum_dist2 = np.bincount(ids, weights=dist2, minlength=K)

    rms_per_id = np.sqrt(sum_dist2[valid] / count[valid])

    return np.mean(rms_per_id)

# ============================================================
# OPTYMALIZACJA
# ============================================================

best_score = np.inf
best_params = None

total_iter = len(Dx_values)*len(Dy_values)*len(yaw_values)

with tqdm(total=total_iter) as pbar:
    for Dx in Dx_values:
        for Dy in Dy_values:
            for yaw in yaw_values:

                score = compute_score(Dx, Dy, yaw)

                if score < best_score:
                    best_score = score
                    best_params = (Dx, Dy, yaw)

                pbar.update(1)

Dx_opt, Dy_opt, yaw_opt = best_params

print("\n============================")
print("NAJLEPSZY D_x =", Dx_opt, "m")
print("NAJLEPSZY D_y =", Dy_opt, "m")
print("NAJLEPSZY yaw_offset =",
      yaw_opt * 180/np.pi, "deg")
print("ŚREDNI RMS =", best_score)
print("============================")