# M4_bev_robust_homography.py
# ==============================================================
# BEV-METRIC-HOMOGRAPHY
# Bird's Eye View from ChArUco correspondences
# ==============================================================

# Description
# --------------------------------------------------------------
# This script generates a Bird's Eye View (BEV) image from
# ChArUco board correspondences using a fisheye camera model
# and robust planar homography estimation.
#
# The program removes fisheye distortion, estimates a robust
# homography using USAC_MAGSAC, and produces a metric BEV
# representation of the calibration board.
#
#
# Pipeline
# --------------------------------------------------------------
#
# 1. Load camera frame
#
# 2. Load ChArUco correspondences from CSV
#
#       pixel_x , pixel_y  → image coordinates
#       square_z , square_y → board coordinates
#
# 3. Convert board coordinates to millimeters
#
# 4. Remove fisheye distortion from image and points
#
# 5. Estimate robust homography using
#
#       cv2.USAC_MAGSAC
#
#       pixel → world
#
# 6. Define BEV scale using
#
#       mm_per_pixel
#
# 7. Generate BEV image using
#
#       cv2.warpPerspective
#
#
# Properties
# --------------------------------------------------------------
#
# Advantages
#
# + simple and mathematically consistent pipeline
# + robust homography estimation (USAC_MAGSAC)
# + fisheye distortion correction
# + metric BEV representation
#
#
# Limitations
#
# - assumes planar ground surface
# - accuracy depends on camera calibration
# - homography is valid only near the calibration plane
#
#
# Input
# --------------------------------------------------------------
#
# IMAGE
#     frame containing ChArUco board
#
# CSV
#     pixel ↔ board correspondences
#
# MODEL
#     saved fisheye camera calibration parameters
#
#
# Output
# --------------------------------------------------------------
#
# BEV image with metric scale defined by:
#
#     mm_per_pixel
#
#
# Notes
# --------------------------------------------------------------
#
# Homography is estimated using USAC_MAGSAC which is a robust
# estimator improving classical RANSAC.
#
# This pipeline is commonly used for planar BEV generation
# in robotics and computer vision.
#
# ==============================================================

# ==============================================================
# BEV-METRIC-HOMOGRAPHY
# Bird's Eye View from ChArUco correspondences
# ==============================================================

from __future__ import annotations

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

IMAGE = "ChArUcoCalibrate/frame_0012.jpg"
CSV = "ChArUcoCalibrate/multi_points_hybrid.csv"
MODEL = "ChArUcoCalibrate/camera_model.npz"

OUT_MAP = "ChArUcoCalibrate/map_homo.npz"

MM_PER_PIXEL = 2.0

MARGIN_FRONT = 300.0
MARGIN_BACK  = 50.0
MARGIN_SIDE  = 120.0


# --------------------------------------------------------
# load CSV (IDENTYCZNE jak M1)
# --------------------------------------------------------

def load_csv(path, frame_name):

    pix = []
    world = []

    with open(path) as f:
        r = csv.DictReader(f)

        for row in r:

            if row["image_name"] != frame_name:
                continue

            px = float(row["pixel_x"])
            py = float(row["pixel_y"])

            # KLUCZOWE: world_x, world_y (jak M1)
            wx = float(row["world_x"])
            wy = float(row["world_y"])

            pix.append([px, py])
            world.append([wx, wy])

    return (
        np.array(pix, np.float64),
        np.array(world, np.float64)
    )


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():

    # -------------------------------
    # load image
    # -------------------------------

    img = cv2.imread(IMAGE)
    if img is None:
        raise RuntimeError("image not loaded")

    Hsrc, Wsrc = img.shape[:2]

    # -------------------------------
    # camera model
    # -------------------------------

    data = np.load(MODEL)

    K = data["K"]
    D = data["D"]
    newK = data["newK"]
    map1 = data["map1"]
    map2 = data["map2"]

    # -------------------------------
    # undistort
    # -------------------------------

    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # -------------------------------
    # load points (jak M1)
    # -------------------------------

    import os
    frame_name = os.path.basename(IMAGE)

    pix, world = load_csv(CSV, frame_name)

    print("points:", len(pix))

    # ---------------------------------------
    #  WORLD BOUNDS (IDENTYCZNE jak M1)
    # ---------------------------------------

    wx_raw = world[:, 0]
    wy_raw = world[:, 1]

    xmin = wx_raw.min() - MARGIN_SIDE
    xmax = wx_raw.max() + MARGIN_SIDE

    ymin = wy_raw.min() - MARGIN_FRONT
    ymax = wy_raw.max() + MARGIN_BACK

    width  = int((xmax - xmin) / MM_PER_PIXEL)
    height = int((ymax - ymin) / MM_PER_PIXEL)

    print("M2 xmin,xmax:", xmin, xmax)
    print("M2 ymin,ymax:", ymin, ymax)
    print("BEV:", width, height)

    # -------------------------------
    # undistort points
    # -------------------------------

    pix_ud = cv2.fisheye.undistortPoints(
        pix.reshape(-1,1,2),
        K, D, P=newK
    ).reshape(-1,2)

    # -------------------------------
    # homography
    # -------------------------------

    H, mask = cv2.findHomography(
        pix_ud,
        world,
        cv2.USAC_MAGSAC,
        1.5
    )

    print("inliers:", int(mask.sum()))

    # -------------------------------
    # world grid (IDENTYCZNY jak M1)
    # -------------------------------

    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymax, ymin, height)

    Xw, Yw = np.meshgrid(xs, ys)

    # -------------------------------
    # world → pixel
    # -------------------------------

    Hinv = np.linalg.inv(H)

    pts = np.stack([Xw, Yw, np.ones_like(Xw)], axis=-1)
    pts = pts.reshape(-1, 3).T

    proj = Hinv @ pts
    proj /= proj[2]

    px = proj[0].reshape(height, width)
    py = proj[1].reshape(height, width)

    # -------------------------------
    # build map
    # -------------------------------

    Xmap = px.astype(np.float32)
    Ymap = py.astype(np.float32)

    # -------------------------------
    # invalid mask
    # -------------------------------

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

    # -------------------------------
    # save map
    # -------------------------------

    np.savez(
        OUT_MAP,
        Xmap=Xmap,
        Ymap=Ymap
    )

    print("map saved:", OUT_MAP)

    # -------------------------------
    # preview
    # -------------------------------

    map_x = Xmap.copy()
    map_y = Ymap.copy()

    invalid = ~np.isfinite(map_x) | ~np.isfinite(map_y)

    map_x[invalid] = -1
    map_y[invalid] = -1

    bev = cv2.remap(
        undist,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    plt.title("BEV (homography aligned)")
    plt.axis("off")
    plt.show()


# --------------------------------------------------------

if __name__ == "__main__":
    main()