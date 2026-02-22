from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from scipy.io import loadmat, savemat


@dataclass(frozen=True)
class Config:
    map_file: str = "map.mat"
    frame_file: str = "frame_0689.jpg"
    out_file: str = "roi_mask.mat"

    # Zakładamy standard projektu: Xmap/Ymap w map.mat są MATLAB 1-based
    maps_are_matlab_1based: bool = True

    show_bev_preview: bool = True


def mat2gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - mn) / (mx - mn)
    return np.clip(y, 0.0, 1.0)


def load_maps(map_file: str) -> tuple[np.ndarray, np.ndarray]:
    S = loadmat(map_file)
    if "Xmap" not in S or "Ymap" not in S:
        raise KeyError(f"{map_file} musi zawierać: Xmap, Ymap")
    Xmap = np.asarray(S["Xmap"], dtype=np.float32)
    Ymap = np.asarray(S["Ymap"], dtype=np.float32)
    if Xmap.shape != Ymap.shape:
        raise ValueError(f"Xmap/Ymap shape mismatch: {Xmap.shape} vs {Ymap.shape}")
    return Xmap, Ymap


def remap_bev_gray01(
    img_01: np.ndarray, Xmap: np.ndarray, Ymap: np.ndarray, matlab_1based: bool
) -> np.ndarray:
    """
    MATLAB: J = interp2(img, Xmap, Ymap, "linear", 0)
    Python: cv2.remap (mapy muszą być 0-based)
    """
    map_x = Xmap.copy()
    map_y = Ymap.copy()

    if matlab_1based:
        # MATLAB 1..W / 1..H  -> OpenCV 0..W-1 / 0..H-1
        valid = np.isfinite(map_x) & np.isfinite(map_y)
        map_x[valid] = map_x[valid] - 1.0
        map_y[valid] = map_y[valid] - 1.0

    # cv2.remap nie lubi NaN w mapach -> ustaw na -1 i użyj borderValue=0
    invalid = ~np.isfinite(map_x) | ~np.isfinite(map_y)
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0

    J = cv2.remap(
        img_01.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return J


def poly2mask(roi_pos_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Odpowiednik MATLAB poly2mask(roiPos(:,1), roiPos(:,2), H, W)
    roi_pos_xy: Nx2 (x,y) w układzie obrazu.
    """
    poly = np.asarray(roi_pos_xy, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("roiPos musi być Nx2 i N>=3")

    poly_i = np.round(poly).astype(np.int32)
    poly_i[:, 0] = np.clip(poly_i[:, 0], 0, W - 1)
    poly_i[:, 1] = np.clip(poly_i[:, 1], 0, H - 1)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_i], 1)
    return mask.astype(bool)


class RoiPolygonPicker:
    """
    Matplotlib GUI:
    - klikaj wierzchołki wielokąta
    - zakończ: dwuklik albo Enter
    """
    def __init__(self, ax: plt.Axes):
        self._verts: list[tuple[float, float]] | None = None
        self._selector = PolygonSelector(
            ax,
            self._on_select,
            useblit=True,
            lineprops={"color": "lime", "linewidth": 2},
            markerprops={
                "marker": "o",
                "markersize": 5,
                "mec": "lime",
                "mfc": "lime",
            },
        )

    def _on_select(self, verts: list[tuple[float, float]]):
        self._verts = verts

    def get(self) -> np.ndarray:
        plt.show()
        if not self._verts or len(self._verts) < 3:
            raise RuntimeError("Nie wybrano ROI (min. 3 punkty).")
        return np.asarray(self._verts, dtype=np.float32)


def main():
    cfg = Config()

    if not os.path.isfile(cfg.map_file):
        raise FileNotFoundError(f"Brak pliku: {cfg.map_file}")
    if not os.path.isfile(cfg.frame_file):
        raise FileNotFoundError(f"Brak pliku: {cfg.frame_file}")

    Xmap, Ymap = load_maps(cfg.map_file)

    img = cv2.imread(cfg.frame_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Nie udało się wczytać: {cfg.frame_file}")
    img_01 = img.astype(np.float32) / 255.0

    # BEV
    J = remap_bev_gray01(img_01, Xmap, Ymap, matlab_1based=cfg.maps_are_matlab_1based)
    J = mat2gray(J)

    if cfg.show_bev_preview:
        plt.figure(figsize=(11, 6), constrained_layout=True)
        plt.imshow(J, cmap="gray", vmin=0, vmax=1)
        plt.title("BEV (J) - podgląd")
        plt.axis("off")

    # Wybór ROI wielokątem
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.imshow(J, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Kliknij wierzchołki ROI (dwuklik lub Enter kończy)")
    ax.set_axis_off()

    picker = RoiPolygonPicker(ax)
    roi_pos = picker.get()  # Nx2: [x,y] w układzie obrazu J

    H, W = J.shape
    roi_mask = poly2mask(roi_pos, H, W)

    # Zapis .mat zgodny z MATLAB
    savemat(
        cfg.out_file,
        {
            "roiMask": roi_mask.astype(np.uint8),  # 0/1
            "roiPos": roi_pos.astype(np.float32),
        },
    )
    print(f"Zapisano: {cfg.out_file}")
    print(f"roiMask: shape={roi_mask.shape}, nnz={int(roi_mask.sum())}")
    print(f"roiPos: points={len(roi_pos)}")

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.imshow(roi_mask, cmap="gray")
    plt.title("ROI mask (1 = zostaw, 0 = wytnij)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()