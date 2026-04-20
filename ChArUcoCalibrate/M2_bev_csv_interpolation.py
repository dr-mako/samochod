# M2_bev_csv_interpolation.py
# ==============================================================
# BEV-CSV-INTERP  (Naive method)
# ==============================================================
#
# Description
# --------------------------------------------------------------
# This script generates a Bird's Eye View (BEV) transformation
# using point correspondences stored in a CSV file.
#
# The CSV contains pairs:
#
#     pixel_x , pixel_y   -> image coordinates
#     world_x , world_y   -> board coordinates (ChArUco grid)
#
# Instead of estimating a geometric camera model (homography or
# full calibration), this method builds a direct inverse mapping:
#
#     (world_x , world_y)  →  (pixel_x , pixel_y)
#
# using scattered interpolation (LinearNDInterpolator).
#
# The resulting mapping is used to create remap matrices which
# can be applied with OpenCV `cv2.remap()` to obtain the BEV image.
#
#
# Pipeline
# --------------------------------------------------------------
# 1. Load source image
# 2. Load pixel ↔ world correspondences from CSV
# 3. Remove duplicates (average points inside same board square)
# 4. Remove outliers using MAD statistics
# 5. Build inverse interpolation functions:
#
#       world → pixel
#
# 6. Generate regular BEV grid
# 7. Evaluate interpolators to produce remap matrices
# 8. Warp the image using cv2.remap()
# 9. Save MATLAB-compatible mapping (map.mat)
#
#
# Properties
# --------------------------------------------------------------
# Advantages
#
# + very simple pipeline
# + does not require camera calibration
# + works with arbitrary camera distortion
# + flexible (can use any sparse point set)
#
#
# Limitations
#
# - purely interpolative (no geometric model)
# - extrapolation outside measured area is unreliable
# - requires dense point coverage
# - sensitive to measurement noise
#
#
# Typical use
# --------------------------------------------------------------
# • quick BEV prototype
# • validation of detected calibration points
# • generating dense remap tables for MATLAB/OpenCV
#
#
# Input
# --------------------------------------------------------------
# image: source camera frame
#
# CSV columns:
#
# image_name
# pixel_x
# pixel_y
# world_x
# world_y
#
#
# Output
# --------------------------------------------------------------
# map.mat
#
# MATLAB/OpenCV remap matrices:
#
#     Xmap
#     Ymap
#
# used for BEV transformation.
#
#
# Notes
# --------------------------------------------------------------
# This is a *naive interpolation-based BEV* and should be treated
# as a baseline method. More accurate approaches typically rely
# on:
#
# • homography estimation
# • camera calibration + projection model
# • polynomial warps
#
# ==============================================================

from __future__ import annotations

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

IMG_PATH = "ChArUcoCalibrate/frame_0012.jpg"
CSV_PATH = "ChArUcoCalibrate/multi_points_" \
"hybrid.csv"

OUT_MAP = "ChArUcoCalibrate/map_interp.npz"

MM_PER_PIXEL = 2.0

MARGIN_FRONT = 300.0
MARGIN_BACK  = 50.0
MARGIN_SIDE  = 120.0

SHOW_PLOTS = True
OUTLIER_K = 3.0


# ------------------------------------------------
# helpers
# ------------------------------------------------

def mad(x):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))


def load_csv_points(path, frame_name):

    pix = []
    world = []

    with open(path) as f:
        r = csv.DictReader(f)

        for row in r:

            if row["image_name"] != frame_name:
                continue

            px = float(row["pixel_x"])
            py = float(row["pixel_y"])

            # world w mm (jak w M2)
            wx = float(row["world_x"])
            wy = float(row["world_y"])

            pix.append([px, py])
            world.append([wx, wy])

    pix = np.array(pix, np.float64)
    world = np.array(world, np.float64)

    return pix, world


def dedup_mean_by_square(pix, world):

    XYu, inv = np.unique(world, axis=0, return_inverse=True)

    pix_u = np.zeros((len(XYu), 2))
    cnt = np.zeros(len(XYu))

    for i, g in enumerate(inv):
        pix_u[g] += pix[i]
        cnt[g] += 1

    pix_u /= np.maximum(cnt[:, None], 1)

    return pix_u, XYu


def build_interp(x, y, v):
    pts = np.stack([x, y], axis=1)
    return LinearNDInterpolator(pts, v, fill_value=np.nan)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("image not loaded")

    Hsrc, Wsrc = img.shape[:2]
    original = img.astype(np.float32) / 255.0

    frame_name = os.path.basename(IMG_PATH)

    # ---------------------------------------
    # load CSV (RAW)
    # ---------------------------------------

    pix, world = load_csv_points(CSV_PATH, frame_name)

    print("points:", len(pix))

    # ---------------------------------------
    # 🔥 WORLD BOUNDS (z RAW danych jak M2)
    # ---------------------------------------

    wx_raw = world[:, 0]
    wy_raw = world[:, 1]

    xmin = wx_raw.min() - MARGIN_SIDE
    xmax = wx_raw.max() + MARGIN_SIDE    

    ymin = wy_raw.min() - MARGIN_FRONT
    ymax = wy_raw.max() + MARGIN_BACK

    print("M1 xmin,xmax:", xmin, xmax)
    print("M1 ymin,ymax:", ymin, ymax)

    width  = int((xmax - xmin) / MM_PER_PIXEL)
    height = int((ymax - ymin) / MM_PER_PIXEL)

    print("BEV:", width, height)

    # ---------------------------------------
    # dedup
    # ---------------------------------------

    pix_u, world_u = dedup_mean_by_square(pix, world)

    wx_u = world_u[:, 0]
    wy_u = world_u[:, 1]

    px_u = pix_u[:, 0]
    py_u = pix_u[:, 1]

    # ---------------------------------------
    # outlier removal
    # ---------------------------------------

    Fx0 = build_interp(wx_u, wy_u, px_u)
    Fy0 = build_interp(wx_u, wy_u, py_u)

    px_hat = Fx0(wx_u, wy_u)
    py_hat = Fy0(wx_u, wy_u)

    r = np.hypot(px_u - px_hat, py_u - py_hat)
    thr = np.nanmedian(r) + OUTLIER_K * mad(r)

    good = np.isfinite(r) & (r <= thr)

    print("filtered:", np.count_nonzero(good))

    # ---------------------------------------
    # final interpolators
    # ---------------------------------------

    Fx = build_interp(wx_u[good], wy_u[good], px_u[good])
    Fy = build_interp(wx_u[good], wy_u[good], py_u[good])

    # ---------------------------------------
    # world grid (IDENTYCZNY jak M2)
    # ---------------------------------------

    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymax, ymin, height)

    Xw, Yw = np.meshgrid(xs, ys)

    # ---------------------------------------
    # map
    # ---------------------------------------

    Xmap = Fx(Xw, Yw).astype(np.float32)
    Ymap = Fy(Xw, Yw).astype(np.float32)

    # ---------------------------------------
    # invalid mask
    # ---------------------------------------

    invalid = (
        ~np.isfinite(Xmap)
        | ~np.isfinite(Ymap)
        | (Xmap < 0)
        | (Xmap > Wsrc - 1)
        | (Ymap < 0)
        | (Ymap > Hsrc - 1)
    )

    Xmap[invalid] = np.nan
    Ymap[invalid] = np.nan

    # ---------------------------------------
    # save
    # ---------------------------------------

    np.savez(
        OUT_MAP,
        Xmap=Xmap,
        Ymap=Ymap
    )

    print("saved:", OUT_MAP)

    # ---------------------------------------
    # preview
    # ---------------------------------------

    map_x = Xmap.copy()
    map_y = Ymap.copy()

    invalid = ~np.isfinite(map_x) | ~np.isfinite(map_y)

    map_x[invalid] = -1
    map_y[invalid] = -1

    bev = cv2.remap(
        original,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    if SHOW_PLOTS:
        plt.imshow(bev, cmap="gray")
        plt.title("BEV (interp)")
        plt.axis("off")
        plt.show()


# ------------------------------------------------

if __name__ == "__main__":
    main()