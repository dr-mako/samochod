from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class Config:
    img_file: str = "frame_0020.jpg"

    # 1 = P_tab.mat, 2 = P_tab_refined.mat
    plik: int = 1
    mat_file: str = "P_tab.mat"
    refined_file: str = "P_tab_refined.mat"

    poly_deg: int = 2
    min_pts: int = 5
    use_robust_fallback: bool = True

    out_mat: str = "Clean_P_tab.mat"
    out_csv: str | None = None  # np. "Clean_P_tab.csv"
    show_plot: bool = True


def _load_ptab(cfg: Config) -> np.ndarray:
    if cfg.plik == 2:
        path = cfg.refined_file
        var = "P_tab_refined"
    else:
        path = cfg.mat_file
        var = "P_tab"

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Brak pliku: {path}")

    S = loadmat(path)
    if var not in S:
        raise KeyError(f"{path} nie zawiera zmiennej '{var}'")

    P = np.asarray(S[var], dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 4:
        raise ValueError(f"{var} musi mieć wymiar Nx4, a ma: {P.shape}")

    return P


def _unique_rows(P: np.ndarray) -> np.ndarray:
    # odpowiednik unique(P, "rows") z MATLAB
    P2 = np.ascontiguousarray(P)
    view = P2.view([("", P2.dtype)] * P2.shape[1])
    _, idx = np.unique(view, return_index=True)
    return P2[np.sort(idx)]


def _pca_rotation_from_ref_line(
    src_xy: np.ndarray,
    sq_x: np.ndarray,
    sq_y: np.ndarray,
    y_lines: np.ndarray,
    Wsrc: int,
    Hsrc: int,
    min_pts: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Zwraca:
      c_src: [cx, cy]
      R: macierz 2x2 (src->wrk) dla obrót o -theta
      theta: kąt osi głównej PCA
      mid_square_y: wartość referencyjna sq_y
    """
    Ny = len(y_lines)
    mid_square_y = float(y_lines[int(round((Ny + 1) / 2)) - 1])

    idx_ref = sq_y == mid_square_y
    ref_src = src_xy[idx_ref]
    ref_src = _unique_rows(ref_src)

    if ref_src.shape[0] < min_pts:
        raise ValueError(
            f"Za mało punktów na referencyjnej linii sq_y=={mid_square_y}: "
            f"{ref_src.shape[0]}"
        )

    c_src = np.array([Wsrc / 2 - 1.0, Hsrc / 2 - 1.0], dtype=np.float64)

    ref0 = ref_src - c_src
    C = np.cov(ref0.T, bias=False)
    vals, vecs = np.linalg.eigh(C)
    k = int(np.argmax(vals))
    direction = vecs[:, k]
    theta = float(np.arctan2(direction[1], direction[0]))

    ct = float(np.cos(-theta))
    st = float(np.sin(-theta))
    R = np.array([[ct, -st], [st, ct]], dtype=np.float64)

    return c_src, R, theta, mid_square_y


def src2wrk(P: np.ndarray, c_src: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (P - c_src) @ R


def wrk2src(Q: np.ndarray, c_src: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (Q @ R.T) + c_src


def _fit_poly(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    return np.polyfit(x, y, deg)


def _polyval(p: np.ndarray, x: np.ndarray | float) -> np.ndarray | float:
    return np.polyval(p, x)


def _intersection_via_root(
    pH: np.ndarray,
    pV: np.ndarray,
    y0: float,
    y_bounds: tuple[float, float] | None,
) -> float:
    """
    Rozwiązuje y = H(V(y)) <=> H(V(y)) - y = 0.
    Używa root_scalar:
    - najpierw próbuje bracketing w zakresie danych (jeśli y_bounds)
    - potem fallback na metodę siecznych z punktem startowym
    """
    def fun(y: float) -> float:
        x = float(_polyval(pV, y))        # V_x_of_y
        yH = float(_polyval(pH, x))       # H_y_of_x
        return yH - y

    if y_bounds is not None:
        a, b = float(y_bounds[0]), float(y_bounds[1])
        if np.isfinite(a) and np.isfinite(b) and b > a:
            # Spróbuj znaleźć znak na końcach; jak nie, i tak spróbujemy secant.
            fa = fun(a)
            fb = fun(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa == 0.0:
                return a
            if np.isfinite(fa) and np.isfinite(fb) and fb == 0.0:
                return b
            if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
                sol = root_scalar(fun, bracket=(a, b), method="brentq")
                if sol.converged:
                    return float(sol.root)

    # Fallback: secant (odpowiednik "fzero z jednym startem")
    sol = root_scalar(fun, x0=float(y0), x1=float(y0) + 1.0, method="secant")
    if not sol.converged:
        raise RuntimeError("root finding failed")
    return float(sol.root)


def main():
    cfg = Config()

    # 1) Obraz (tylko rozmiar + wizualizacja)
    img = cv2.imread(cfg.img_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Brak obrazu: {cfg.img_file}")
    Hsrc, Wsrc = img.shape[:2]

    P_tab = _load_ptab(cfg)

    src_x = P_tab[:, 0]
    src_y = P_tab[:, 1]
    sq_x = P_tab[:, 2]
    sq_y = P_tab[:, 3]

    src_xy = np.stack([src_x, src_y], axis=1)

    x_lines = np.unique(sq_x)
    y_lines = np.unique(sq_y)

    Nx = len(x_lines)
    Ny = len(y_lines)

    print(f"Wykryto linie: Nx={Nx} (sq_x), Ny={Ny} (sq_y)")

    # 3) Układ roboczy (PCA) globalnie na referencyjnej linii sq_y
    c_src, R, theta, mid_square_y = _pca_rotation_from_ref_line(
        src_xy=src_xy,
        sq_x=sq_x,
        sq_y=sq_y,
        y_lines=y_lines,
        Wsrc=Wsrc,
        Hsrc=Hsrc,
        min_pts=cfg.min_pts,
    )
    print(f"Ref sq_y(mid)={mid_square_y} | theta={theta:.4f} rad")

    # 4) Pre-ekstrakcja punktów dla linii
    Hsrc_by_y: list[np.ndarray] = []
    for yv in y_lines:
        idx = sq_y == yv
        P = _unique_rows(src_xy[idx])
        Hsrc_by_y.append(P)

    Vsrc_by_x: list[np.ndarray] = []
    for xv in x_lines:
        idx = sq_x == xv
        P = _unique_rows(src_xy[idx])
        Vsrc_by_x.append(P)

    # Stabilniejsze starty y0: mediana y w układzie roboczym dla każdej pionowej
    V_y0 = np.full(Nx, np.nan, dtype=np.float64)
    for ix in range(Nx):
        V_wrk = src2wrk(Vsrc_by_x[ix], c_src, R)
        if V_wrk.shape[0] >= 1:
            V_y0[ix] = float(np.median(V_wrk[:, 1]))

    clean_rows: list[list[float]] = []

    # 5) Przecięcia wszystkich par
    for iy, yv in enumerate(y_lines):
        for ix, xv in enumerate(x_lines):
            H_src = Hsrc_by_y[iy]
            V_src = Vsrc_by_x[ix]

            if H_src.shape[0] < cfg.min_pts or V_src.shape[0] < cfg.min_pts:
                continue

            H_wrk = src2wrk(H_src, c_src, R)
            V_wrk = src2wrk(V_src, c_src, R)

            H_wrk = H_wrk[np.argsort(H_wrk[:, 0])]  # po x
            V_wrk = V_wrk[np.argsort(V_wrk[:, 1])]  # po y

            degH = min(cfg.poly_deg, H_wrk.shape[0] - 1)
            degV = min(cfg.poly_deg, V_wrk.shape[0] - 1)
            if degH < 1 or degV < 1:
                continue

            def try_fit_and_intersect(deg_h: int, deg_v: int) -> tuple[float, float]:
                pH = _fit_poly(H_wrk[:, 0], H_wrk[:, 1], deg_h)  # y=H(x)
                pV = _fit_poly(V_wrk[:, 1], V_wrk[:, 0], deg_v)  # x=V(y)

                y0 = V_y0[ix]
                if not np.isfinite(y0):
                    y0 = float(np.median(V_wrk[:, 1]))

                y_bounds = (float(np.min(V_wrk[:, 1])), float(np.max(V_wrk[:, 1])))

                y_star = _intersection_via_root(pH, pV, y0=y0, y_bounds=y_bounds)
                x_star = float(_polyval(pV, y_star))
                return x_star, y_star

            ok = False
            x_star = y_star = np.nan

            try:
                x_star, y_star = try_fit_and_intersect(degH, degV)
                ok = True
            except Exception:
                if cfg.use_robust_fallback:
                    try:
                        x_star, y_star = try_fit_and_intersect(1, 1)
                        ok = True
                    except Exception:
                        ok = False

            if not ok:
                continue

            Pint_wrk = np.array([[x_star, y_star]], dtype=np.float64)
            Pint_src = wrk2src(Pint_wrk, c_src, R)[0]

            clean_rows.append([float(Pint_src[0]), float(Pint_src[1]), float(xv), float(yv)])

    Clean_P_tab = np.asarray(clean_rows, dtype=np.float64)
    print(f"Zbudowano Clean_P_tab: {len(Clean_P_tab)} punktów (z {Nx*Ny} możliwych)")

    # 6) Zapis .mat (zgodnie z MATLAB style)
    savemat(
        cfg.out_mat,
        {
            "Clean_P_tab": Clean_P_tab,
            "xLines": x_lines,
            "yLines": y_lines,
            "Nx": Nx,
            "Ny": Ny,
            "midSquareY": mid_square_y,
            "polyDeg": cfg.poly_deg,
            "minPts": cfg.min_pts,
            "c_src": c_src,
            "R": R,
        },
    )
    print(f"Zapisano: {cfg.out_mat}")

    # (Opcjonalnie) CSV
    if cfg.out_csv is not None:
        import csv

        with open(cfg.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pixel_x", "pixel_y", "square_x", "square_y"])
            w.writerows(Clean_P_tab.tolist())
        print(f"Zapisano: {cfg.out_csv}")

    # 7) Wizualizacja
    if cfg.show_plot:
        plt.figure(figsize=(12, 7), constrained_layout=True)
        plt.imshow(img, cmap="gray")
        plt.scatter(Clean_P_tab[:, 0], Clean_P_tab[:, 1], s=15, c="lime")
        plt.title("Punkty narożników szachownicy na obrazie (Clean_P_tab)")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()