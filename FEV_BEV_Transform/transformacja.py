from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import LinearNDInterpolator


@dataclass(frozen=True)
class Config:
    img_path: str = "frame_0012.jpg"

    clean_ptab_mat: str = "P_map.mat"
    clean_ptab_var: str = "Clean_P_tab"

    height_y: int = 420
    width_x: int = 840

    out_map_mat: str = "map.mat"

    # outlier threshold: median(r) + k * MAD
    outlier_k: float = 3.0

    # opcjonalny podgląd
    show_plots: bool = True

    # jeśli True, zapisze też J podglądowe jako png
    save_preview_png: bool = False
    preview_png: str = "bev_preview.png"


def mad(x: np.ndarray) -> float:
    """
    MAD jak w MATLAB: mad(x,1) = median(|x - median(x)|)
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def mat2gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - mn) / (mx - mn)
    return np.clip(y, 0.0, 1.0)


def dedup_mean_by_square(
    pix_x: np.ndarray, pix_y: np.ndarray, sq_x: np.ndarray, sq_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplikacja po (sq_x, sq_y) + uśrednienie pix_x, pix_y.
    """
    XY = np.stack([sq_x, sq_y], axis=1)
    XYu, inv = np.unique(XY, axis=0, return_inverse=True)

    pix_x_u = np.zeros(len(XYu), dtype=np.float64)
    pix_y_u = np.zeros(len(XYu), dtype=np.float64)
    cnt = np.zeros(len(XYu), dtype=np.int64)

    for i, g in enumerate(inv):
        pix_x_u[g] += float(pix_x[i])
        pix_y_u[g] += float(pix_y[i])
        cnt[g] += 1

    pix_x_u /= np.maximum(cnt, 1)
    pix_y_u /= np.maximum(cnt, 1)

    square_x_u = XYu[:, 0].astype(np.float64)
    square_y_u = XYu[:, 1].astype(np.float64)

    return pix_x_u, pix_y_u, square_x_u, square_y_u


def build_linear_interp(
    square_x: np.ndarray, square_y: np.ndarray, values: np.ndarray
) -> LinearNDInterpolator:
    pts = np.stack([square_x, square_y], axis=1)
    return LinearNDInterpolator(pts, values, fill_value=np.nan, rescale=False)


def main():
    cfg = Config()

    if not os.path.isfile(cfg.img_path):
        raise FileNotFoundError(f"Brak obrazu: {cfg.img_path}")
    if not os.path.isfile(cfg.clean_ptab_mat):
        raise FileNotFoundError(f"Brak pliku: {cfg.clean_ptab_mat}")

    # 1) obraz: tylko Hsrc, Wsrc + podgląd
    img = cv2.imread(cfg.img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Nie udało się wczytać: {cfg.img_path}")
    Hsrc, Wsrc = img.shape[:2]
    original_01 = img.astype(np.float32) / 255.0

    # 3) wczytaj Clean_P_tab
    S = loadmat(cfg.clean_ptab_mat)
    if cfg.clean_ptab_var not in S:
        raise KeyError(
            f"{cfg.clean_ptab_mat} nie zawiera zmiennej '{cfg.clean_ptab_var}'"
        )
    P_tab = np.asarray(S[cfg.clean_ptab_var], dtype=np.float64)
    if P_tab.ndim != 2 or P_tab.shape[1] != 4:
        raise ValueError(f"Clean_P_tab musi mieć rozmiar Nx4, a ma: {P_tab.shape}")

    pix_x = P_tab[:, 0]
    pix_y = P_tab[:, 1]
    square_x = P_tab[:, 2]
    square_y = P_tab[:, 3]

    # 4) deduplikacja + średnia
    pix_x_u, pix_y_u, square_x_u, square_y_u = dedup_mean_by_square(
        pix_x, pix_y, square_x, square_y
    )

    # 5) outliery: interpolant -> błąd -> odrzuć
    F0x = build_linear_interp(square_x_u, square_y_u, pix_x_u)
    F0y = build_linear_interp(square_x_u, square_y_u, pix_y_u)

    px_hat = F0x(square_x_u, square_y_u)
    py_hat = F0y(square_x_u, square_y_u)

    r = np.hypot(pix_x_u - px_hat, pix_y_u - py_hat)
    r = np.asarray(r, dtype=np.float64)

    thr = float(np.nanmedian(r) + cfg.outlier_k * mad(r))
    good = np.isfinite(r) & (r <= thr)

    removed = int(np.count_nonzero(~good))
    total = int(len(r))
    pct = 100.0 * removed / max(1, total)

    print(
        f"Punkty: {total} -> {int(np.count_nonzero(good))} "
        f"(usunięto {removed}, {pct:.2f}%), thr={thr:.3f} px"
    )

    # 6) finalne interpolanty
    Inv_Fx = build_linear_interp(square_x_u[good], square_y_u[good], pix_x_u[good])
    Inv_Fy = build_linear_interp(square_x_u[good], square_y_u[good], pix_y_u[good])

    # 7) siatka docelowa w układzie drogi
    x_min = float(np.nanmin(square_x_u))
    x_max = float(np.nanmax(square_x_u))
    y_min = float(np.nanmin(square_y_u))
    y_max = float(np.nanmax(square_y_u))

    piks_droga_x = np.linspace(x_min, x_max, cfg.width_x, dtype=np.float64)
    piks_droga_y = np.linspace(y_max, y_min, cfg.height_y, dtype=np.float64)  # odwrócone
    droga_X, droga_Y = np.meshgrid(piks_droga_x, piks_droga_y)

    # 8) mapa do obrazu źródłowego (UWAGA: MATLAB generuje 1-based; my generujemy 0-based)
    Xmap0 = Inv_Fx(droga_X, droga_Y)  # kolumny 0..W-1
    Ymap0 = Inv_Fy(droga_X, droga_Y)  # wiersze 0..H-1

    out = (
        ~np.isfinite(Xmap0)
        | ~np.isfinite(Ymap0)
        | (Xmap0 < 0.0)
        | (Xmap0 > float(Wsrc - 1))
        | (Ymap0 < 0.0)
        | (Ymap0 > float(Hsrc - 1))
    )
    Xmap0 = Xmap0.astype(np.float32)
    Ymap0 = Ymap0.astype(np.float32)
    Xmap0[out] = np.nan
    Ymap0[out] = np.nan

    # 10) resampling (podgląd) – cv2.remap potrzebuje map bez NaN
    map_x = Xmap0.copy()
    map_y = Ymap0.copy()
    invalid = ~np.isfinite(map_x) | ~np.isfinite(map_y)
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0

    J = cv2.remap(
        original_01,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    J = mat2gray(J)

    # Zapis map w standardzie MATLAB (1-based) do map.mat:
    # żeby segment_lanes_morphology (który używa logiki MATLAB) nie musiał się zmieniać.
    Xmap_matlab = Xmap0.copy().astype(np.float32)
    Ymap_matlab = Ymap0.copy().astype(np.float32)

    # tam gdzie valid -> +1, NaN zostaje NaN
    valid = np.isfinite(Xmap_matlab) & np.isfinite(Ymap_matlab)
    Xmap_matlab[valid] = Xmap_matlab[valid] + 1.0
    Ymap_matlab[valid] = Ymap_matlab[valid] + 1.0

    savemat(cfg.out_map_mat, {"Xmap": Xmap_matlab, "Ymap": Ymap_matlab})
    print(f"Zapisano: {cfg.out_map_mat} (Xmap/Ymap, MATLAB 1-based)")

    if cfg.save_preview_png:
        plt.imsave(cfg.preview_png, J, cmap="gray")
        print(f"Zapisano podgląd: {cfg.preview_png}")

    if cfg.show_plots:
        plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.imshow(original_01, cmap="gray")
        plt.title("Obraz źródłowy (gray)")
        plt.axis("off")

        plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.imshow(J, cmap="gray")
        plt.title("Obraz po korekcie (BEV preview)")
        plt.axis("off")

        # diagnostyka outlierów
        plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.imshow(original_01, cmap="gray")
        plt.scatter(pix_x_u[good], pix_y_u[good], s=10, c="lime", label="good")
        plt.scatter(pix_x_u[~good], pix_y_u[~good], s=20, c="red", label="outliers")
        plt.title("Punkty użyte (good/outliers)")
        plt.legend()
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    main()