import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# ŚCIEŻKI (zmień jeśli trzeba)
# ---------------------------------------------------------
BASE = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-17-3")
TRAJ_PATH = BASE / "synced_with_traj.csv"
LAND_PATH = BASE / "landmarks_mapfit.csv"   # <- TU wynik Twojego programu

# ---------------------------------------------------------
# PUNKT STARTOWY (możesz zmienić)
# ---------------------------------------------------------
x0 = np.array([0.1156, 0.0025, -2.62 * np.pi / 180.0])  # [dx, dy, dyaw]

# ---------------------------------------------------------
# FUNKCJA AUTOMATYCZNEGO WYKRYWANIA ZAMKNIĘCIA PĘTLI
# ---------------------------------------------------------
def get_loop_closure_frame(df_land, gap_threshold=1000, min_obs=5):
    """
    Analizuje wszystkie landmarki i zwraca frame_id pierwszego 
    wykrytego zamknięcia pętli (powrotu do znanego punktu).
    """
    potential_closures = []

    for uid in df_land["id"].unique():
        sub = df_land[df_land["id"] == uid].sort_values("frame_id")
        frames = sub["frame_id"].values

        if len(frames) < 2:
            continue

        diffs = np.diff(frames)
        if diffs.max() > gap_threshold:
            idx = np.argmax(diffs)
            seg1 = sub.iloc[:idx+1]
            seg2 = sub.iloc[idx+1:]

            # Sprawdzamy czy oba wystąpienia są wiarygodne
            if len(seg1) >= min_obs and len(seg2) >= min_obs:
                # Klatka, w której marker pojawia się ponownie
                closure_start = seg2["frame_id"].min()
                potential_closures.append(closure_start)

    # Zwracamy najwcześniejsze wykryte zamknięcie pętli ze wszystkich markerów
    return min(potential_closures) if potential_closures else None

# ---------------------------------------------------------
# WCZYTAJ DANE I ODETNIJ POWTÓRZENIA
# ---------------------------------------------------------
traj = pd.read_csv(TRAJ_PATH).set_index("frame_id")
land_raw = pd.read_csv(LAND_PATH)

# Automatyczne wykrycie końca pierwszego okrążenia
closure_frame = get_loop_closure_frame(land_raw, gap_threshold=1000, min_obs=5)

if closure_frame:
    print(f"--- AUTOMATYCZNE WYKRYWANIE PĘTLI ---")
    print(f"Wykryto zamknięcie pętli na frame_id: {closure_frame}")
    # Filtrujemy: zostawiamy tylko dane PRZED ponownym zobaczeniem któregokolwiek markera
    land = land_raw[land_raw["frame_id"] < closure_frame].copy()
    print(f"Odrzucono dane z drugiego przejazdu (zachowano {len(land)} z {len(land_raw)} wpisów).")
else:
    print("Nie wykryto zamknięcia pętli (brak powtórzonych markerów). Używam wszystkich danych.")
    land = land_raw.copy()

# Dalej program idzie bez zmian...

# ---------------------------------------------------------
# FUNKCJE
# ---------------------------------------------------------
def transform_C_to_G(x_c, y_c, yaw_c, dx, dy, dyaw):
    c = np.cos(yaw_c + dyaw)
    s = np.sin(yaw_c + dyaw)
    x_g = x_c + c * dx - s * dy
    y_g = y_c + s * dx + c * dy
    return x_g, y_g

def residuals(params):
    dx, dy, dyaw = params
    res = []
    used = 0

    for _, row in land.iterrows():
        # frame id
        try:
            frame_id = int(row["frame_id"])
        except Exception:
            continue
        if frame_id not in traj.index:
            continue

        # traj
        tr = traj.loc[frame_id]
        x_c = tr["x"]
        y_c = tr["y"]
        yaw_c = tr["phi_rad"]
        if not (np.isfinite(x_c) and np.isfinite(y_c) and np.isfinite(yaw_c)):
            continue

        # global marker
        x_glob = row["x_global"]
        y_glob = row["y_global"]
        if not (np.isfinite(x_glob) and np.isfinite(y_glob)):
            continue

        # predicted global marker
        x_pred, y_pred = transform_C_to_G(x_c, y_c, yaw_c, dx, dy, dyaw)

        # residual
        res.append(x_pred - x_glob)
        res.append(y_pred - y_glob)
        used += 1

    if used == 0:
        raise ValueError("Brak poprawnych punktów do optymalizacji.")
    r = np.array(res)
    if not np.all(np.isfinite(r)):
        raise ValueError("Residuale zawierają NaN/inf.")
    #print(f"Użyto punktów: {used}")
    return r



# ---------------------------------------------------------
# OPTYMALIZACJA
# ---------------------------------------------------------
print("Start least_squares...")
#result = least_squares(residuals, x0, verbose=2)
lb = np.array([ -0.30, -0.30, -15*np.pi/180 ])
ub = np.array([  0.30,  0.30,  15*np.pi/180 ])
#x_scale = np.array([0.1, 0.1, 0.01])  # metry, metry, rad
result = least_squares(
    residuals,
    x0,
    bounds=(lb, ub),      
    method='trf',          # trf - Trust Region Reflective – domyślny, ale trzeba go „rozkręcić”
                           # lm - Levenberga-Marquardta
    jac='3-point',         # '3-point' lub 'cs' dla lepszej dokładności, ale wolniej
    ftol=1e-12,            # domyślnie 1e-8
    xtol=1e-12,            # domyślnie 1e-8
    gtol=1e-12,            # domyślnie 1e-8
    x_scale='jac',         # automatyczna skala – lepsza kondycja Hessego
    tr_solver='lsmr',      # dla dużych, rzadkich Jacobich
    verbose=2,
    max_nfev=1000,         # bezpieczny limit kroków
)

# ---------------------------------------------------------
# WYNIKI
# ---------------------------------------------------------
dx_opt, dy_opt, dyaw_opt = result.x
print("\n=== WYNIK OPTYMALIZACJI ===")
print(f"dx   = {dx_opt:.6f} m")
print(f"dy   = {dy_opt:.6f} m")
print(f"yaw  = {dyaw_opt:.6f} rad  ({np.degrees(dyaw_opt):.3f} deg)")
print(f"cost = {result.cost:.6f}")

# ---------------------------------------------------------
# RYSOWANIE
# ---------------------------------------------------------
pairs = pd.merge(
    land,
    traj[["x", "y", "phi_rad"]],
    left_on="frame_id",
    right_index=True,
    how="inner",
)

# przed
x0_pred, y0_pred = transform_C_to_G(
    pairs["x"].to_numpy(),
    pairs["y"].to_numpy(),
    pairs["phi_rad"].to_numpy(),
    x0[0], x0[1], x0[2],
)
# po
x1_pred, y1_pred = transform_C_to_G(
    pairs["x"].to_numpy(),
    pairs["y"].to_numpy(),
    pairs["phi_rad"].to_numpy(),
    dx_opt, dy_opt, dyaw_opt,
)

plt.figure(figsize=(14, 6))

ax = plt.subplot(1, 2, 1)
plt.plot(traj["x"], traj["y"], "-k", linewidth=1, label="traj (C)")
plt.scatter(pairs["x_global"], pairs["y_global"], s=15, label="landmarki (mapa)")
plt.scatter(x0_pred, y0_pred, s=8, label="C->G (x0)", color="C1")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.legend()
plt.title("PRZED optymalizacją")

ax = plt.subplot(1, 2, 2)
plt.plot(traj["x"], traj["y"], "-k", linewidth=1, label="traj (C)")
plt.scatter(pairs["x_global"], pairs["y_global"], s=15, label="landmarki (mapa)")
plt.scatter(x1_pred, y1_pred, s=8, label="C->G (opt)", color="C2")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.legend()
plt.title("PO optymalizacji")

plt.tight_layout()
plt.show()