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
LAND_PATH = BASE / "landmarks_mapfit.csv"

# ============================================================
# PARAMETRY STARTOWE (x0) I GRANICE
# ============================================================
# [dx, dy, dyaw]
x0 = np.array([0.1156, 0.0160, -1.81 * np.pi / 180.0])
lb = np.array([0.0, -0.10, -10 * np.pi / 180]) # dolna granica
ub = np.array([0.30,  0.10,  10 * np.pi / 180]) # górna granica

# ============================================================
# FUNKCJE
# ============================================================

def transform_C_to_G_vectorized(x_c, y_c, yaw_c, dx, dy, dyaw):
    """
    Pełna transformacja 2D (identyczna z Twoim skryptem do generowania mapy)
    Wersja zoptymalizowana pod NumPy.
    """
    # 1. Przesunięcie pozycji bazowej (środek kamery)
    x_base = x_c + dx * np.cos(yaw_c) - dy * np.sin(yaw_c)
    y_base = y_c + dx * np.sin(yaw_c) + dy * np.cos(yaw_c)
    
    # Uwaga: W optymalizacji szukamy poprawki do pozycji kamery 
    # względem trajektorii. W Twoim skrypcie rysującym 
    # landmarki są one już "wypalone" w x_global, y_global.
    return x_base, y_base

def residuals(params, x_c, y_c, yaw_c, x_glob, y_glob):
    dx, dy, dyaw = params
    # W tym modelu optymalizujemy głównie dx, dy robota 
    # Jeśli chcesz optymalizować też obrót samych punktów, 
    # należałoby uwzględnić dyaw w transformacji lokalnej.
    x_pred, y_pred = transform_C_to_G_vectorized(x_c, y_c, yaw_c, dx, dy, dyaw)
    
    res = np.empty(x_pred.size * 2)
    res[0::2] = x_pred - x_glob
    res[1::2] = y_pred - y_glob
    return res

def get_loop_closure_frame(df_land, gap_threshold=1000, min_obs=5):
    potential_closures = []
    for uid in df_land["id"].unique():
        sub = df_land[df_land["id"] == uid].sort_values("frame_id")
        frames = sub["frame_id"].values
        if len(frames) < 2: continue
        diffs = np.diff(frames)
        if diffs.max() > gap_threshold:
            idx = np.argmax(diffs)
            if len(sub.iloc[:idx+1]) >= min_obs and len(sub.iloc[idx+1:]) >= min_obs:
                potential_closures.append(sub.iloc[idx+1]["frame_id"])
    return min(potential_closures) if potential_closures else None

# ============================================================
# WCZYTYWANIE I PRZYGOTOWANIE
# ============================================================
traj = pd.read_csv(TRAJ_PATH).set_index("frame_id")
land_raw = pd.read_csv(LAND_PATH)

closure_frame = get_loop_closure_frame(land_raw)
if closure_frame:
    print(f"Wykryto zamknięcie pętli na frame: {closure_frame}. Odcinam duble.")
    land = land_raw[land_raw["frame_id"] < closure_frame].copy()
else:
    land = land_raw.copy()

# Łączenie danych do optymalizacji
data = pd.merge(
    land, 
    traj[["x", "y", "phi_rad"]], 
    left_on="frame_id", 
    right_index=True, 
    how="inner"
).dropna()

# Dane do optymalizacji (NumPy)
xc, yc, yawc = data["x"].values, data["y"].values, data["phi_rad"].values
xg, yg = data["x_global"].values, data["y_global"].values

# ============================================================
# OPTYMALIZACJA
# ============================================================
print(f"Start optymalizacji na {len(data)} punktach...")
res = least_squares(
    residuals, x0, args=(xc, yc, yawc, xg, yg),
    bounds=(lb, ub), loss='soft_l1', f_scale=0.1, verbose=2
)

dx_opt, dy_opt, dyaw_opt = res.x
print(f"\nWynik: dx={dx_opt:.4f}, dy={dy_opt:.4f}, yaw_deg={np.degrees(dyaw_opt):.4f}")

# ============================================================
# WYKRES (Z KOLORAMI I LEGENDĄ)
# ============================================================
# Obliczamy pozycje PO optymalizacji dla wizualizacji
x_opt, y_opt = transform_C_to_G_vectorized(xc, yc, yawc, *res.x)

plt.figure(figsize=(12, 8))

# 1. Rysujemy trajektorię
plt.plot(traj["x"], traj["y"], "-k", linewidth=1, alpha=0.3, label="Trajektoria")

# 2. Rysujemy landmarki z podziałem na ID
unique_ids = sorted(data["id"].unique())

for uid in unique_ids:
    # Tworzymy maskę pozycji w tablicy 'data' (którą użyliśmy do xc, yc...)
    # Używamy .values, aby dostać czystą maskę NumPy
    mask = (data["id"] == uid).values 
    
    # Kolor zostanie przydzielony automatycznie z cyklu
    p = plt.scatter(
        xg[mask], 
        yg[mask], 
        s=40, marker='o', alpha=0.6, label=f"ID {int(uid)}"
    )
    
    # Pobieramy kolor ostatniego scattera, żeby 'x' miały ten sam kolor
    color = p.get_facecolor()
    
    # Rysujemy punkty po optymalizacji (krzyżyki)
    plt.scatter(
        x_opt[mask], 
        y_opt[mask], 
        s=20, marker='x', color=color, alpha=0.8
    )

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Dopasowanie: dx={dx_opt:.3f}m, dy={dy_opt:.3f}m, dyaw={np.degrees(dyaw_opt):.2f}°\n"
          f"(Kółka = Mapa, Krzyżyki = Po optymalizacji)")
plt.grid(True, linestyle='--', alpha=0.5)

# Legenda na zewnątrz
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize='small', ncol=2)
plt.subplots_adjust(right=0.75)

plt.tight_layout()
plt.show()

# ============================================================
# UPROSZCZONE PORÓWNANIE: TRAJEKTORIA + LANDMARKI + GT
# ============================================================

# 1. Definicja Ground Truth (z MATLABa)
gt_raw = {
    0: [0, 0], 1: [68, 0], 2: [121.5, 0], 3: [154, -32],
    4: [154, -86], 5: [121.5, -117], 6: [95, -86],
    7: [68, -59], 8: [41, -117], 9: [0, -117],
    11: [-22, -86], 10: [-22, -32]
}
SCALE = 0.01 
gt_df = pd.DataFrame.from_dict(gt_raw, orient='index', columns=['x', 'y']) * SCALE

# 2. Wyrównanie Ground Truth do pierwszego wykrytego punktu
first_obs = data.iloc[0]
first_id = int(first_obs["id"])
if first_id in gt_df.index:
    off_x = first_obs["x_global"] - gt_df.loc[first_id, 'x']
    off_y = first_obs["y_global"] - gt_df.loc[first_id, 'y']
    gt_df['x_shifted'] = gt_df['x'] + off_x
    gt_df['y_shifted'] = gt_df['y'] + off_y
else:
    gt_df['x_shifted'], gt_df['y_shifted'] = gt_df['x'], gt_df['y']

# 3. WYKRES
plt.figure(figsize=(10, 10))

# A. Trajektoria odometryczna ("zamrożona")
plt.plot(traj["x"], traj["y"], "-k", linewidth=1, alpha=0.3, label="Trajektoria (Odometria)")

# B. Wykryte landmarki (Surowe chmury punktów z logu)
for uid in sorted(data["id"].unique()):
    mask = (data["id"] == uid).values
    # Rysujemy chmurę punktów dla każdego ID
    plt.scatter(xg[mask], yg[mask], s=10, alpha=0.5, label=f"Wykryte ID {int(uid)}")

# C. Rzeczywiste położenia landmarków (Ground Truth)
plt.scatter(gt_df['x_shifted'], gt_df['y_shifted'], s=120, color='red', marker='+', 
            linewidth=2, label="Prawdziwe położenie (GT)")

# Dodanie etykiet ID przy punktach rzeczywistych
for idx, row in gt_df.iterrows():
    plt.text(row['x_shifted'], row['y_shifted'], f"  {int(idx)}", color='red', fontweight='bold')

plt.gca().set_aspect("equal")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Weryfikacja: Wykryte chmury punktów vs Rzeczywiste pozycje (GT)")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize='small')

plt.tight_layout()
plt.show()