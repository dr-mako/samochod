# plot_and_traj.py 
# przeliczenie sterowań na trajektorię + niepewność i wykresy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== ŚCIEŻKI =====

in_dir_main = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-17-5")
in_dir_main = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-19")
SYNCED_RAW_PATH = in_dir_main / "synced_raw.csv"
OUT_WITH_TRAJ =  in_dir_main / "synced_with_traj.csv"
OUT_SIMPLE =  in_dir_main / "synced_simple.csv"
OUT_PATH = in_dir_main / "ath.csv"


# ===== PARAMETRY GEOMETRII =====
R = 37.25 / 1000.0  # [m]
L = 260.0 / 1000.0  # [m]

# ===== OFFSETY (w stopniach) =====
OFFSET_FWD = -2
OFFSET_REV = 0
OFFSET_MODE = "fwd"  # "auto" / "fwd" / "rev"
V_DEADBAND = 0.02  # [m/s]
DEFAULT_DIR = "fwd"  # "fwd" / "rev"

# Zakres czasu (opcjonalnie)
T_MIN = None
T_MAX = None

# ===== PARAMETRY NIEPEWNOŚCI (jak w Matlabie) =====
# Błędy sterowania (u=[v,delta])
SIGMA_V = 5.0 / 60.0 * 2.0 * np.pi * R  # [m/s] (5 obr/min)
SIGMA_DELTA = 2.0 * np.pi / 180.0  # [rad] (2 deg)
Q = np.diag([SIGMA_V**2, SIGMA_DELTA**2])

# Błędy pozycji i kursu (P0)
SIGMA_X0 = 0.05  # [m]
SIGMA_Y0 = 0.05  # [m]
SIGMA_PHI0 = 2.0 * np.pi / 180.0  # [rad]
P0 = np.diag([SIGMA_X0**2, SIGMA_Y0**2, SIGMA_PHI0**2])

# Rysowanie
DRAW_FROM_K = 20
DRAW_EVERY = 25
ELLIPSE_NSIGMA = 0.5


def choose_offset_auto(v_mps: np.ndarray) -> np.ndarray:
    if DEFAULT_DIR not in ("fwd", "rev"):
        raise ValueError("DEFAULT_DIR must be 'fwd' or 'rev'")

    offset = np.empty_like(v_mps, dtype=float)
    dir_state = DEFAULT_DIR

    for i, v in enumerate(v_mps):
        if np.isfinite(v):
            if v > V_DEADBAND:
                dir_state = "fwd"
            elif v < -V_DEADBAND:
                dir_state = "rev"
        offset[i] = OFFSET_FWD if dir_state == "fwd" else OFFSET_REV

    return offset


def ackermann_step(x: np.ndarray, u: np.ndarray, dt: float, L: float) -> np.ndarray:
    v = float(u[0])
    delta = float(u[1])
    phi = float(x[2])

    xn = np.empty_like(x, dtype=float)
    xn[0] = x[0] + dt * v * np.cos(phi)
    xn[1] = x[1] + dt * v * np.sin(phi)
    xn[2] = x[2] + 2.0 * dt * v / L * np.tan(delta)
    return xn


def jacobians_ackermann(
    x: np.ndarray, u: np.ndarray, dt: float, L: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jacobiany spójne z modelem:
      phi+ = phi + 2*dt*v/L*tan(delta)
    Stan x=[x,y,phi], sterowanie u=[v,delta]
    """
    v = float(u[0])
    delta = float(u[1])
    phi = float(x[2])

    JacFx = np.array(
        [
            [1.0, 0.0, -dt * v * np.sin(phi)],
            [0.0, 1.0, dt * v * np.cos(phi)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    sec2 = 1.0 / (np.cos(delta) ** 2)
    JacFu = np.array(
        [
            [dt * np.cos(phi), 0.0],
            [dt * np.sin(phi), 0.0],
            [2.0 * dt * np.tan(delta) / L, 2.0 * dt * v * sec2 / L],
        ],
        dtype=float,
    )

    return JacFx, JacFu


def plot_ellipse_like_matlab(ax, x: np.ndarray, P: np.ndarray, n_sigma: float):
    Pxy = P[0:2, 0:2]
    mu = x[0:2]

    if np.any(np.diag(Pxy) == 0):
        return
    if np.any(~np.isfinite(Pxy)) or np.any(~np.isfinite(mu)):
        return

    vals, vecs = np.linalg.eigh(Pxy)
    if np.any(vals <= 0):
        return

    ang = np.arange(0.0, 2.0 * np.pi + 0.1, 0.1)
    y = n_sigma * np.vstack([np.cos(ang), np.sin(ang)])

    el = vecs @ (np.diag(np.sqrt(vals)) @ y)
    el = np.hstack([el, el[:, [0]]]) + mu.reshape(2, 1)

    ax.plot(el[0, :], el[1, :], lw=1.0)


def draw_robot_like_matlab(ax, Xr: np.ndarray, col: str):
    p = 0.02
    a = ax.axis()
    l1 = (a[1] - a[0]) * p
    l2 = (a[3] - a[2]) * p

    Ptri = np.array([[-1, 1, 0, -1], [-1, -1, 3, -1]], dtype=float)

    theta = float(Xr[2] - np.pi / 2.0)
    c = np.cos(theta)
    s = np.sin(theta)
    Rm = np.array([[c, -s], [s, c]], dtype=float)

    Ptri = Rm @ Ptri
    Ptri[0, :] = Ptri[0, :] * l1 + float(Xr[0])
    Ptri[1, :] = Ptri[1, :] * l2 + float(Xr[1])

    ax.plot(Ptri[0, :], Ptri[1, :], col, lw=0.8)
    ax.plot([float(Xr[0])], [float(Xr[1])], marker="+", color=col, ms=6)


def ellipse_params_from_Pxy(Pxy: np.ndarray) -> tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(Pxy)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    a = float(np.sqrt(max(vals[0], 0.0)))
    b = float(np.sqrt(max(vals[1], 0.0)))
    alpha = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return a, b, alpha


def main():
    if not os.path.isfile(SYNCED_RAW_PATH):
        raise FileNotFoundError(f"Brak pliku: {SYNCED_RAW_PATH}")

    df = pd.read_csv(SYNCED_RAW_PATH)
    required = {"frame_id", "time", "spd_mean_raw", "servo_mean_deg_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn: {sorted(missing)}")

    df = df.sort_values("time").reset_index(drop=True)

    if T_MIN is not None:
        df = df[df["time"] >= T_MIN]
    if T_MAX is not None:
        df = df[df["time"] <= T_MAX]
    df = df.reset_index(drop=True)

    t = df["time"].to_numpy(dtype=float)
    n = len(df)

    # ===== v_mps =====
    spd_mean_raw = df["spd_mean_raw"].to_numpy(dtype=float)
    v_mps = -spd_mean_raw / 10.0 / 60.0 * 2.0 * np.pi * R

    # ===== delta_rad (z offsetem) =====
    servo_mean_deg_raw = df["servo_mean_deg_raw"].to_numpy(dtype=float)

    if OFFSET_MODE == "fwd":
        offset_deg = np.full_like(servo_mean_deg_raw, OFFSET_FWD, dtype=float)
    elif OFFSET_MODE == "rev":
        offset_deg = np.full_like(servo_mean_deg_raw, OFFSET_REV, dtype=float)
    elif OFFSET_MODE == "auto":
        offset_deg = choose_offset_auto(v_mps)
    else:
        raise ValueError("OFFSET_MODE must be: 'auto', 'fwd', 'rev'")

    servo_cal_deg = -(servo_mean_deg_raw + offset_deg)
    delta_rad = servo_cal_deg * np.pi / 180.0

    # ===== Integracja + P =====
    X = np.zeros((n, 3), dtype=float)  # [x,y,phi]
    P_hist = np.zeros((n, 3, 3), dtype=float)

    P = P0.copy()
    P_hist[0] = P
    xk = np.array([0.0, 0.0, 0.0], dtype=float)
    X[0] = xk

    for k in range(n - 1):
        dt = float(t[k + 1] - t[k])
        if not np.isfinite(dt) or dt <= 0:
            X[k + 1] = xk
            P_hist[k + 1] = P
            continue

        uk = np.array([v_mps[k], delta_rad[k]], dtype=float)
        if np.any(~np.isfinite(uk)) or np.any(~np.isfinite(xk)):
            X[k + 1] = xk
            P_hist[k + 1] = P
            continue

        JacFx, JacFu = jacobians_ackermann(xk, uk, dt, L)
        P = JacFx @ P @ JacFx.T + JacFu @ Q @ JacFu.T
        xk = ackermann_step(xk, uk, dt, L)

        X[k + 1] = xk
        P_hist[k + 1] = P

    x = X[:, 0]
    y = X[:, 1]
    phi = X[:, 2]

    # ===== Parametry elipsy do CSV =====
    ell_a = np.full(n, np.nan, dtype=float)
    ell_b = np.full(n, np.nan, dtype=float)
    ell_alpha = np.full(n, np.nan, dtype=float)

    for k in range(n):
        Pxy = P_hist[k, 0:2, 0:2]
        if np.any(~np.isfinite(Pxy)):
            continue
        try:
            a, b, alpha = ellipse_params_from_Pxy(Pxy)
        except Exception:
            continue
        ell_a[k] = a
        ell_b[k] = b
        ell_alpha[k] = alpha

    # ===== synced_with_traj.csv =====
    df_out = df.copy()
    df_out["v_mps"] = v_mps
    df_out["offset_deg_used"] = offset_deg
    df_out["delta_rad"] = delta_rad
    df_out["x"] = x
    df_out["y"] = y
    df_out["phi_rad"] = phi

    df_out["P_xx"] = P_hist[:, 0, 0]
    df_out["P_xy"] = P_hist[:, 0, 1]
    df_out["P_xphi"] = P_hist[:, 0, 2]
    df_out["P_yx"] = P_hist[:, 1, 0]
    df_out["P_yy"] = P_hist[:, 1, 1]
    df_out["P_yphi"] = P_hist[:, 1, 2]
    df_out["P_phix"] = P_hist[:, 2, 0]
    df_out["P_phiy"] = P_hist[:, 2, 1]
    df_out["P_phiphi"] = P_hist[:, 2, 2]

    df_out["sigma_x_m"] = np.sqrt(np.maximum(df_out["P_xx"].to_numpy(), 0.0))
    df_out["sigma_y_m"] = np.sqrt(np.maximum(df_out["P_yy"].to_numpy(), 0.0))
    df_out["sigma_phi_rad"] = np.sqrt(
        np.maximum(df_out["P_phiphi"].to_numpy(), 0.0)
    )

    df_out["ellipse_a_1sigma_m"] = ell_a
    df_out["ellipse_b_1sigma_m"] = ell_b
    df_out["ellipse_alpha_rad"] = ell_alpha

    df_out.to_csv(OUT_WITH_TRAJ, index=False, encoding="utf-8")
    print(f"Zapisano: {OUT_WITH_TRAJ} (wiersze={len(df_out)})")

    # ===== synced_simple.csv =====
    df_simple = df_out[["frame_id", "time", "v_mps", "delta_rad"]].copy()
    df_simple.to_csv(OUT_SIMPLE, index=False, encoding="utf-8")
    print(f"Zapisano: {OUT_SIMPLE} (wiersze={len(df_simple)})")

    # ===== path.csv (zgodnie z README) =====
    df_path = pd.DataFrame(
        {
            "pos_x": x,
            "pos_y": y,
            "current_speed": v_mps,
            "current_angle": phi,  # kurs
            "timestep": t,  # czas absolutny
        }
    )
    df_path.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Zapisano: {OUT_PATH} (wiersze={len(df_path)})")

    # ===== WYKRESY: sterowanie =====
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, constrained_layout=True
    )

    axes[0].plot(t, v_mps, "-", lw=1.2)
    axes[0].set_title("v(t) [m/s]")
    axes[0].set_ylabel("v [m/s]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, delta_rad, "-", lw=1.2, color="tab:red")
    axes[1].set_title("delta(t) [rad]")
    axes[1].set_ylabel("delta [rad]")
    axes[1].grid(True, alpha=0.3)

    dt_arr = np.diff(t)
    axes[2].plot(t[1:], dt_arr, "-", lw=1.2, color="tab:green")
    axes[2].set_title("dT między klatkami [s]")
    axes[2].set_xlabel("czas [s]")
    axes[2].set_ylabel("dT [s]")
    axes[2].grid(True, alpha=0.3)

    # ===== WYKRESY: niepewności (błędy 1-sigma) =====
    fig_err, ax_err = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, constrained_layout=True
    )

    sigma_x = df_out["sigma_x_m"].to_numpy(dtype=float)
    sigma_y = df_out["sigma_y_m"].to_numpy(dtype=float)
    sigma_phi = df_out["sigma_phi_rad"].to_numpy(dtype=float)

    ax_err[0].plot(t, sigma_x, "-", lw=1.2, color="tab:blue")
    ax_err[0].set_title("Niepewność pozycji: sigma_x(t)")
    ax_err[0].set_ylabel("sigma_x [m]")
    ax_err[0].grid(True, alpha=0.3)

    ax_err[1].plot(t, sigma_y, "-", lw=1.2, color="tab:orange")
    ax_err[1].set_title("Niepewność pozycji: sigma_y(t)")
    ax_err[1].set_ylabel("sigma_y [m]")
    ax_err[1].grid(True, alpha=0.3)

    ax_err[2].plot(t, sigma_phi, "-", lw=1.2, color="tab:purple")
    ax_err[2].set_title("Niepewność kursu: sigma_phi(t)")
    ax_err[2].set_xlabel("czas [s]")
    ax_err[2].set_ylabel("sigma_phi [rad]")
    ax_err[2].grid(True, alpha=0.3)

    # ===== Trajektoria i błąd =====
    fig2, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    ax.plot(x, y, "-k", lw=2.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajektoria i błąd (elipsy)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-0.75, 1.0])

    for k in range(DRAW_FROM_K, n):
        if (k - 1) % DRAW_EVERY != 0:
            continue
        ax.plot([x[k]], [y[k]], "ko", ms=4)
        draw_robot_like_matlab(ax, X[k], "r")
        plot_ellipse_like_matlab(ax, X[k], P_hist[k], ELLIPSE_NSIGMA)

    plt.show()


if __name__ == "__main__":
    main()