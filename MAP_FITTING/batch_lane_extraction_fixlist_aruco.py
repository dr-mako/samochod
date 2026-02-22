from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter


# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:
    in_dir_main: str = r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-19"
    frames_subdir: str = "frames"

    map_file: str = "map.mat"
    map_inv_file: str = "map_inv.mat"   # <-- NOWE
    roi_file: str = "roi_mask.mat"
    use_roi: bool = True

    out_dir_name: str = "out_lane_aruco_bev"

    # Lane segmentation
    bg_disk_radius: int = 30
    smooth_sigma: float = 1.5
    close_radius: int = 7
    min_area: int = 200

    # ArUco
    aruco_dict_name: str = "DICT_4X4_50"

    temporal_window: int = 5

    save_bev: bool = True
    max_frames: int | None = None


# ============================================================
# HELPERS
# ============================================================
def mat2gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx <= mn:
        return np.zeros_like(x)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def remap_bev(img_01, Xmap, Ymap):
    map_x = Xmap - 1.0
    map_y = Ymap - 1.0
    invalid = ~np.isfinite(map_x) | ~np.isfinite(map_y)
    map_x[invalid] = -1
    map_y[invalid] = -1

    return cv2.remap(
        img_01,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# MAIN
# ============================================================
def main():

    cfg = Config()

    in_dir = Path(cfg.in_dir_main) / cfg.frames_subdir
    out_dir = Path(cfg.in_dir_main) / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "png"
    out_png.mkdir(exist_ok=True)

    if cfg.save_bev:
        out_bev = out_dir / "bev"
        out_bev.mkdir(exist_ok=True)

    # --- LOAD MAPS ---
    S = loadmat(cfg.map_file)
    Xmap = S["Xmap"].astype(np.float32)
    Ymap = S["Ymap"].astype(np.float32)
    Hdst, Wdst = Xmap.shape

    S_inv = loadmat(cfg.map_inv_file)
    Bev_X = S_inv["Bev_pix_X"].astype(np.float32)
    Bev_Y = S_inv["Bev_pix_Y"].astype(np.float32)

    # ROI
    roi_mask = np.ones((Hdst, Wdst), bool)
    if cfg.use_roi:
        R = loadmat(cfg.roi_file)
        roi_mask = R["roiMask"].astype(bool)

    files = sorted(in_dir.glob("*.jpg"))
    if cfg.max_frames is not None:
        files = files[: cfg.max_frames]

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, cfg.aruco_dict_name)
    )
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    id_history = []

    summary_path = out_dir / "summary.csv"
    summary_file = open(summary_path, "w", encoding="utf-8")
    summary_file.write(
        "frame,raw_id,smoothed_id,"
        "marker_cam_x,marker_cam_y,"
        "marker_bev_x,marker_bev_y,"
        "lane_pixels\n"
    )

    try:
        for i, fn in enumerate(files, 1):

            # ---- LOAD IMAGE ----
            file_bytes = np.fromfile(str(fn), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img01 = gray.astype(np.float32) / 255.0

            # ---- ARUCO ----
            corners, ids, _ = detector.detectMarkers(gray)

            best_id = np.nan
            best_corners = None

            if ids is not None:
                areas = [
                    cv2.contourArea(c.reshape(-1, 1, 2).astype(np.float32))
                    for c in corners
                ]
                idx = int(np.argmax(areas))
                best_id = int(ids.flatten()[idx])
                best_corners = corners[idx].reshape(4, 2)

            # ---- SMOOTHING ----
            id_history.append(best_id)
            if len(id_history) > cfg.temporal_window:
                id_history.pop(0)

            valid_ids = [x for x in id_history if not np.isnan(x)]
            smoothed_id = (
                max(set(valid_ids), key=valid_ids.count)
                if valid_ids
                else np.nan
            )

            # ---- BEV ----
            J = remap_bev(img01, Xmap, Ymap)
            J = mat2gray(J)
            J[~roi_mask] = 0

            bg = opening(J, disk(cfg.bg_disk_radius))
            Jn = mat2gray(J - bg)
            Jn2 = gaussian_filter(Jn, cfg.smooth_sigma)

            t = cv2.threshold(
                (Jn2 * 255).astype(np.uint8),
                0,
                255,
                cv2.THRESH_OTSU,
            )[0] / 255

            BW = Jn2 > t
            BW = closing(BW, disk(cfg.close_radius))
            BW = remove_small_objects(BW, min_size=cfg.min_area)
            BW &= roi_mask

            # ---- OVERLAY BEV ----
            canvas = cv2.cvtColor(
                (J * 255).astype(np.uint8),
                cv2.COLOR_GRAY2BGR,
            )

            E = find_boundaries(BW, mode="outer")
            yy, xx = np.nonzero(E)
            canvas[yy, xx] = (0, 0, 0)

            marker_cam_x = np.nan
            marker_cam_y = np.nan
            marker_bev_x = np.nan
            marker_bev_y = np.nan

            if best_corners is not None:

                # --- centroid in camera ---
                marker_cam_x = float(np.mean(best_corners[:, 0]))
                marker_cam_y = float(np.mean(best_corners[:, 1]))

                u = int(round(marker_cam_x))
                v = int(round(marker_cam_y))

                if (
                    0 <= v < Bev_X.shape[0]
                    and 0 <= u < Bev_X.shape[1]
                ):
                    marker_bev_x = Bev_X[v, u]
                    marker_bev_y = Bev_Y[v, u]

                # --- draw in BEV if valid ---
                if np.isfinite(marker_bev_x) and np.isfinite(marker_bev_y):

                    x_bev = int(round(marker_bev_x))
                    y_bev = int(round(marker_bev_y))

                    cv2.drawMarker(
                        canvas,
                        (x_bev, y_bev),
                        (0, 0, 255),
                        cv2.MARKER_TILTED_CROSS,
                        18,
                        2,
                    )

                    cv2.putText(
                        canvas,
                        f"ID={smoothed_id}",
                        (x_bev + 10, y_bev),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # ---- SAVE PNG ----
            out_path = out_png / f"{fn.stem}.png"
            success, buffer = cv2.imencode(".png", canvas)
            if success:
                buffer.tofile(str(out_path))

            # ---- SAVE BEV CLEAN ----
            if cfg.save_bev:
                bev_path = out_bev / f"{fn.stem}.png"
                success, buffer = cv2.imencode(
                    ".png", (J * 255).astype(np.uint8)
                )
                if success:
                    buffer.tofile(str(bev_path))

            # ---- SUMMARY ----
            summary_file.write(
                f"{fn.name},{best_id},{smoothed_id},"
                f"{marker_cam_x},{marker_cam_y},"
                f"{marker_bev_x},{marker_bev_y},"
                f"{int(np.count_nonzero(BW))}\n"
            )
            summary_file.flush()

            if i % 50 == 0 or i == len(files):
                print(f"{i}/{len(files)}")

    except KeyboardInterrupt:
        print("Interrupted by user. Partial summary saved.")

    finally:
        summary_file.close()

    print("Done.")


if __name__ == "__main__":
    main()