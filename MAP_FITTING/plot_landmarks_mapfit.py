
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ŚCIEŻKI
# ============================================================

BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-2")

TRAJ_PATH = BASE / "synced_with_traj.csv"
LAND_PATH = BASE / "out_lane_aruco_bev" / "summary_clean.csv"

OUT_CSV = BASE / "landmarks_mapfit.csv"

# ============================================================
# PARAMETRY (Z ESTYMACJI)
# ============================================================
# 2026-02-11-3 to to:
#D_X_FROM_C_M = 0.11187500000000002     # wzdłuż osi pojazdu
#D_Y_FROM_C_M = 0.0085     
#YAW_OFFSET_RAD = 7.5 * np.pi / 180.0
# 2026-02-15:
#D_X_FROM_C_M = 0.11562500000000002
#D_Y_FROM_C_M = -0.0015000000000000013
#YAW_OFFSET_RAD = -0.18749999999999947 * np.pi / 180.0
# 2026-02-17-1:
#D_X_FROM_C_M = 0.11562500000000002
#D_Y_FROM_C_M = 0.016000000000000004
#YAW_OFFSET_RAD = -1.8124999999999998 * np.pi / 180.0
# 2026-02-17-2:
#D_X_FROM_C_M = 0.10625000000000001
#D_Y_FROM_C_M = 0.0075
#YAW_OFFSET_RAD = -0.7499999999999994 * np.pi / 180.0

# 2026-02-17-3:
D_X_FROM_C_M = 0.14
D_Y_FROM_C_M = 0.01575
YAW_OFFSET_RAD = -0.49999999999999994 * np.pi / 180.0

# 2026-02-23-1:
D_X_FROM_C_M = 0.175
D_Y_FROM_C_M = 0.02
YAW_OFFSET_RAD = 2.0 * np.pi / 180.0

# 2026-02-23-2:
D_X_FROM_C_M = 0.2150
D_Y_FROM_C_M = 0.21875
YAW_OFFSET_RAD = -0.45 * np.pi / 180.0

# przed extrinsics 
#(z modelu L/2 + R) L=260, R = 37.25 , d = 0.1673 m
D_X_FROM_C_M = 0.1673
D_Y_FROM_C_M = 0.0 
YAW_OFFSET_RAD = 0 * np.pi / 180.0


IMAGE_H_PX = 420
U0_PX = 412
M_PER_PX = 1.0 / 1000.0

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
)

df = df.dropna(subset=["marker_bev_x", "marker_bev_y"])

print("Obserwacji landmarków:", len(df))

# ============================================================
# TRANSFORMACJA DO GLOBAL
# ============================================================

global_rows = []

for _, row in df.iterrows():

    xC = row["x"]
    yC = row["y"]
    phi = row["phi_rad"]

    # ========================================================
    # NOWE: pełne przesunięcie 2D kamery
    # ========================================================

    xG = (
        xC
        + D_X_FROM_C_M * np.cos(phi)
        - D_Y_FROM_C_M * np.sin(phi)
    )

    yG = (
        yC
        + D_X_FROM_C_M * np.sin(phi)
        + D_Y_FROM_C_M * np.cos(phi)
    )

    # yaw kamery
    phi_total = phi + YAW_OFFSET_RAD

    # lokalne BEV (px → m)
    u_loc = (IMAGE_H_PX - row["marker_bev_y"]) * M_PER_PX
    v_loc = (U0_PX - row["marker_bev_x"]) * M_PER_PX

    # rotacja
    c = np.cos(-phi_total)
    s = np.sin(-phi_total)

    x_w = xG + (u_loc * c + v_loc * s)
    y_w = yG + (-u_loc * s + v_loc * c)

    global_rows.append({
        "frame_id": row["frame_id"],
        "id": int(row["smoothed_id"]),
        "x_global": x_w,
        "y_global": y_w
    })

global_df = pd.DataFrame(global_rows)

# ============================================================
# ZAPIS
# ============================================================

global_df.to_csv(OUT_CSV, index=False)
print("Zapisano:", OUT_CSV)

# ============================================================
# WYKRES
# ============================================================

plt.figure(figsize=(9, 8))
plt.plot(traj["x"], traj["y"], "-k", linewidth=1.5)

for uid in sorted(global_df["id"].unique()):
    mask = global_df["id"] == uid
    plt.scatter(
        global_df.loc[mask, "x_global"],
        global_df.loc[mask, "y_global"],
        s=20,
        label=f"ID {uid}"
    )

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trajektoria + globalne pozycje landmarków")
plt.grid(True)
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
plt.subplots_adjust(right=0.75)
plt.show()


