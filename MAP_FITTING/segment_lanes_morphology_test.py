# segment_lanes_morphology_test.py
# ============================================================
# Diagnostic lane segmentation on a list of frames:
# bird's-eye -> background normalization -> threshold -> morphology
# -> length filter -> optional glare removal (safe) -> perimeter/skeleton
# + console diagnostics and plots
# ============================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.morphology import (
    disk,
    opening,
    closing,
    remove_small_objects,
    skeletonize,
)
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter


PlotMode = Literal["perim", "skel", "mask"]
Preset = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class SegParams:
    bg_disk_radius: int
    smooth_sigma: float
    otsu_scale: float
    close_line_len0: int
    close_line_len90: int
    min_area: int
    use_length_filter: bool
    min_length: float
    use_glare_removal: bool


def get_params(preset: Preset) -> SegParams:
    if preset == "A":
        return SegParams(
            bg_disk_radius=30,
            smooth_sigma=1.5,
            otsu_scale=1.00,
            close_line_len0=15,
            close_line_len90=15,
            min_area=200,
            use_length_filter=True,
            min_length=80,
            use_glare_removal=False,
        )
    if preset == "B":
        return SegParams(
            bg_disk_radius=60,
            smooth_sigma=2.5,
            otsu_scale=1.35,
            close_line_len0=11,
            close_line_len90=11,
            min_area=600,
            use_length_filter=True,
            min_length=140,
            use_glare_removal=False,
        )
    if preset == "C":
        return SegParams(
            bg_disk_radius=80,
            smooth_sigma=3.0,
            otsu_scale=1.55,
            close_line_len0=9,
            close_line_len90=9,
            min_area=900,
            use_length_filter=True,
            min_length=180,
            use_glare_removal=False,
        )
    if preset == "D":
        # D = C + anti-glare
        return SegParams(
            bg_disk_radius=80,
            smooth_sigma=3.0,
            otsu_scale=1.55,
            close_line_len0=9,
            close_line_len90=9,
            min_area=900,
            use_length_filter=True,
            min_length=180,
            use_glare_removal=True,
        )
    raise ValueError(f"Unknown preset: {preset}")


def mat2gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - mn) / (mx - mn)
    return np.clip(y, 0.0, 1.0)


def load_map_and_roi(map_file: str, roi_file: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    m = loadmat(map_file)
    r = loadmat(roi_file)

    # Expect Xmap, Ymap, roiMask as in MATLAB
    Xmap = np.asarray(m["Xmap"], dtype=np.float32)
    Ymap = np.asarray(m["Ymap"], dtype=np.float32)
    roi_mask = np.asarray(r["roiMask"], dtype=bool)

    h, w = Xmap.shape
    return Xmap, Ymap, roi_mask, h, w


def birdseye_warp_gray(
    img_gray_01: np.ndarray,
    Xmap: np.ndarray,
    Ymap: np.ndarray,
) -> np.ndarray:
    """
    MATLAB: J = interp2(imgSingle, Xmap, Ymap, "linear", 0)
    Here: use cv2.remap.
    Note: cv2.remap expects map_x/map_y in pixel coordinates.
    """
    # OpenCV remap uses (x,y) float32 maps
    map_x = Xmap.astype(np.float32)
    map_y = Ymap.astype(np.float32)

    # cv2.remap expects source image in 0..1 float32 is fine
    J = cv2.remap(
        img_gray_01.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return J


def line_selem(length: int, angle_deg: float) -> np.ndarray:
    """
    Approximation of MATLAB strel("line", len, angle).
    For 0 or 90 deg, use simple rectangular structuring elements.
    """
    length = max(1, int(length))
    if int(angle_deg) % 180 == 0:  # horizontal
        return np.ones((1, length), dtype=bool)
    if int(angle_deg) % 180 == 90:  # vertical
        return np.ones((length, 1), dtype=bool)

    # Fallback: small rotated line (rarely needed here)
    se = np.zeros((length, length), dtype=np.uint8)
    cv2.line(
        se,
        (0, length // 2),
        (length - 1, length // 2),
        255,
        1,
    )
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(se, M, (length, length)) > 0
    return rot


def length_filter(BW: np.ndarray, min_length: float) -> np.ndarray:
    """
    MATLAB: regionprops(...,"MajorAxisLength") filter.
    In skimage, regionprops gives major_axis_length for labeled components.
    """
    lab = label(BW, connectivity=2)
    out = np.zeros_like(BW, dtype=bool)
    for reg in regionprops(lab):
        if reg.major_axis_length >= float(min_length):
            out[lab == reg.label] = True
    return out


def safe_glare_removal(
    BW: np.ndarray,
    roi_area: int,
    trigger_bw_frac: float,
    max_largest_frac_of_roi: float,
    max_dominance: float,
    min_ratio12: float,
) -> tuple[np.ndarray, dict]:
    """
    Removes the largest connected component only if multiple safety conditions hold.
    Mirrors MATLAB logic and returns diagnostics.
    """
    diag = {
        "bwFrac": float(np.count_nonzero(BW) / max(1, roi_area)),
        "numObj": 0,
        "largestFrac": 0.0,
        "dominance": 0.0,
        "ratio12": float("inf"),
        "removed": False,
        "numObjAfter": 0,
        "largestFracAfter": 0.0,
        "why": "",
    }

    lab = label(BW, connectivity=2)
    regs = regionprops(lab)
    diag["numObj"] = len(regs)

    if diag["bwFrac"] <= trigger_bw_frac:
        diag["why"] = "bwFrac<=trigger"
        return BW, diag
    if len(regs) < 2:
        diag["why"] = "numObj<2"
        return BW, diag

    areas = np.array([r.area for r in regs], dtype=float)
    order = np.argsort(areas)[::-1]
    a1 = float(areas[order[0]])
    a2 = float(areas[order[1]])

    diag["largestFrac"] = a1 / max(1, roi_area)
    diag["dominance"] = a1 / max(1, np.count_nonzero(BW))
    diag["ratio12"] = a1 / max(1.0, a2)

    if diag["largestFrac"] < max_largest_frac_of_roi:
        diag["why"] = "largestFrac<thr"
        return BW, diag
    if diag["dominance"] > max_dominance:
        diag["why"] = "dominance>max"
        return BW, diag
    if diag["ratio12"] < min_ratio12:
        diag["why"] = "ratio12<min"
        return BW, diag

    # Remove largest component
    largest_label = regs[order[0]].label
    BW2 = BW.copy()
    BW2[lab == largest_label] = False
    diag["removed"] = True
    diag["why"] = "removed"

    lab2 = label(BW2, connectivity=2)
    regs2 = regionprops(lab2)
    diag["numObjAfter"] = len(regs2)
    if len(regs2) > 0:
        areas2 = np.array([r.area for r in regs2], dtype=float)
        diag["largestFracAfter"] = float(np.max(areas2) / max(1, roi_area))

    return BW2, diag


def main():
    preset: Preset = "C"
    diagnostic = True

    frames_dir = r"LOGI\frames"
    frames = [
        "frame_0687",
        # "frame_0726",
        # ...
    ]

    map_file = "map.mat"
    roi_file = "roi_mask.mat"
    plot_mode: PlotMode = "perim"  # "perim" | "skel" | "mask"

    # Anti-glare globals 
    trigger_bw_frac = 0.16
    max_largest_frac_of_roi = 0.12
    max_dominance = 0.85
    min_ratio12 = 1.60

    Xmap, Ymap, roi_mask, Hdst, Wdst = load_map_and_roi(map_file, roi_file)
    roi_area = int(np.count_nonzero(roi_mask))

    params = get_params(preset)

    for name in frames:
        img_path = os.path.join(frames_dir, f"{name}.jpg")
        print(f'\nImageName = "{name}"')

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        # Read grayscale and scale to 0..1 float
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read: {img_path}")
        img_01 = img.astype(np.float32) / 255.0

        # Bird's-eye via remap + normalize + ROI
        J = birdseye_warp_gray(img_01, Xmap, Ymap)
        J = mat2gray(J)
        J[~roi_mask] = 0.0

        # Background normalization + smoothing
        bg = opening(J, disk(params.bg_disk_radius))
        Jn = mat2gray(J - bg)
        Jn2 = gaussian_filter(Jn, sigma=float(params.smooth_sigma))

        # Threshold
        t_otsu = float(threshold_otsu(Jn2))
        thr_used = float(params.otsu_scale * t_otsu)
        BW = Jn2 > thr_used

        # Morphology
        BW = closing(BW, line_selem(params.close_line_len0, 0))
        BW = closing(BW, line_selem(params.close_line_len90, 90))
        BW = remove_small_objects(BW, min_size=int(params.min_area))

        # Length filter
        if params.use_length_filter:
            BW = length_filter(BW, params.min_length)

        # ROI
        BW = BW & roi_mask

        # Pre-glare diagnostics
        bw_frac = float(np.count_nonzero(BW) / max(1, roi_area))
        lab_pre = label(BW, connectivity=2)
        regs_pre = regionprops(lab_pre)
        num_obj = len(regs_pre)

        largest_frac = 0.0
        dominance = 0.0
        ratio12 = float("inf")

        if num_obj > 0:
            areas = np.array([r.area for r in regs_pre], dtype=float)
            areas_sorted = np.sort(areas)[::-1]
            a1 = float(areas_sorted[0])
            largest_frac = a1 / max(1, roi_area)
            dominance = a1 / max(1, np.count_nonzero(BW))
            if len(areas_sorted) >= 2:
                a2 = float(areas_sorted[1])
                ratio12 = a1 / max(1.0, a2)

        # Anti-glare (only if preset enables it)
        glare_removed = False
        why = "glareOFF"
        num_obj_after = num_obj
        largest_frac_after = largest_frac

        if params.use_glare_removal:
            BW, gd = safe_glare_removal(
                BW=BW,
                roi_area=roi_area,
                trigger_bw_frac=trigger_bw_frac,
                max_largest_frac_of_roi=max_largest_frac_of_roi,
                max_dominance=max_dominance,
                min_ratio12=min_ratio12,
            )
            glare_removed = bool(gd["removed"])
            why = str(gd["why"])
            num_obj_after = int(gd["numObjAfter"] if gd["removed"] else gd["numObj"])
            largest_frac_after = float(
                gd["largestFracAfter"] if gd["removed"] else gd["largestFrac"]
            )

        if diagnostic:
            print(
                "Preset {p} | thrUsed={thr:.4f} | BW/ROI={bw:.3f} | "
                "NumObj={no} | largestFrac={lf:.3f} | dominance={dom:.3f} | "
                "ratio12={r:.2f} | removed={rem} | NumObjAfter={na} | "
                "largestFracAfter={lfa:.3f} | {why}".format(
                    p=preset,
                    thr=thr_used,
                    bw=bw_frac,
                    no=num_obj,
                    lf=largest_frac,
                    dom=dominance,
                    r=ratio12,
                    rem=int(glare_removed),
                    na=num_obj_after,
                    lfa=largest_frac_after,
                    why=why,
                )
            )

        # Perimeter / skeleton / mask points
        E = find_boundaries(BW, mode="outer")
        SKE = skeletonize(BW)
        # Rough equivalent of bwmorph(...,"spur",10): prune small spurs
        # (not identical to MATLAB; keep as diagnostic)
        # A simple heuristic: erode tiny branches by repeated opening on skeleton
        for _ in range(10):
            SKE = SKE & closing(SKE, np.ones((3, 3), dtype=bool))

        if plot_mode == "perim":
            P = E
            ttl = "Punkty obrysu pasów (perimeter)"
        elif plot_mode == "skel":
            P = SKE
            ttl = "Punkty osi pasów (skeleton)"
        else:
            P = BW
            ttl = "Punkty maski pasów"

        # Visualizations
        plt.figure()
        plt.imshow(J, cmap="gray")
        plt.title(f"Bird's-eye (J) {name}")

        plt.figure()
        plt.imshow(Jn2, cmap="gray")
        plt.title(f"Jn2 {name}")

        plt.figure()
        plt.imshow(BW, cmap="gray")
        plt.title(f"BW {name} removed={glare_removed}")

        yy, xx = np.nonzero(P)
        plt.figure()
        plt.scatter(xx, yy, s=3, c="k")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.xlim([1, Wdst])
        plt.ylim([1, Hdst])
        plt.title(f"{ttl} {name}")
        plt.xlabel("x")
        plt.ylabel("y")

    plt.show()


if __name__ == "__main__":
    main()