# bev_fisheye_homography.py
# ==============================================================
# BEV-FISHEYE-HOMOGRAPHY
# ==============================================================
#
# Description
# --------------------------------------------------------------
# This script generates a Bird's Eye View (BEV) image using a
# fisheye camera model and planar homography estimated from
# ChArUco board correspondences.
#
# Unlike the naive interpolation method, this approach uses a
# geometric camera model which allows accurate perspective
# rectification even for wide-angle lenses.
#
#
# Pipeline
# --------------------------------------------------------------
#
# 1. Load calibration image
#
# 2. Load CSV correspondences:
#
#       pixel_x , pixel_y  → image coordinates
#       square_z , square_y → ChArUco board coordinates
#
# 3. Convert board coordinates to millimeters
#
# 4. Estimate fisheye camera model:
#
#       K  – camera matrix
#       D  – distortion coefficients
#
# 5. Undistort image and feature points
#
# 6. Select only bottom board markers
#    (closest to the camera) to stabilize BEV estimation
#
# 7. Estimate planar homography:
#
#       image → board plane
#
# 8. Define BEV coordinate system and scale
#
#       mm_per_pixel
#
# 9. Warp image using cv2.warpPerspective()
#
#
# Properties
# --------------------------------------------------------------
#
# Advantages
#
# + physically meaningful camera model
# + removes fisheye distortion
# + stable geometric transformation
# + requires only sparse calibration points
#
#
# Limitations
#
# - camera calibration from a single frame
# - assumes perfectly planar ground
# - accuracy depends on marker detection quality
#
#
# Typical use
# --------------------------------------------------------------
#
# • robotics
# • mobile robots
# • autonomous driving
# • floor plane rectification
#
#
# Input
# --------------------------------------------------------------
#
# image: ChArUco calibration frame
#
# CSV columns:
#
# pixel_x
# pixel_y
# square_z
# square_y
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
# This method represents a standard BEV pipeline:
#
#     camera model → undistortion → homography
#
# and should produce significantly more stable results than
# purely interpolative approaches.
#
# ==============================================================

from __future__ import annotations

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:

    image_path: str = "ChArUcoCalibrate/frame_0012.jpg"
    csv_path: str = "P_tab.csv"

    square_mm: float = 30.0

    mm_per_pixel: float = 2.0

    bev_offset_mm: float = 150.0

    show_plots: bool = True


def load_ptab_csv(path):

    pix = []
    world = []

    with open(path, newline="") as f:

        r = csv.DictReader(f)

        for row in r:

            px = float(row["pixel_x"])
            py = float(row["pixel_y"])

            sx = float(row["square_z"])
            sy = float(row["square_y"])

            pix.append([px, py])
            world.append([sx, sy])

    return np.array(pix,np.float32), np.array(world,np.float32)


def main():

    cfg = Config()

    img = cv2.imread(cfg.image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError("image not loaded")

    Hsrc,Wsrc = img.shape

    pix,world = load_ptab_csv(cfg.csv_path)

    world *= cfg.square_mm

    # ------------------------------------
    # center board
    # ------------------------------------

    cx = world[:,0].mean()
    cy = world[:,1].mean()

    world[:,0] -= cx
    world[:,1] -= cy

    # ------------------------------------
    # fisheye calibration
    # ------------------------------------

    obj = np.zeros((len(world),1,3),np.float32)
    obj[:,0,:2] = world

    imgp = pix.reshape(-1,1,2)

    K = np.zeros((3,3))
    D = np.zeros((4,1))

    rms,K,D,_,_ = cv2.fisheye.calibrate(
        [obj],[imgp],
        (Wsrc,Hsrc),
        K,D,None,None,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    )

    print("RMS:",rms)

    # ------------------------------------
    # undistort
    # ------------------------------------

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,D,(Wsrc,Hsrc),np.eye(3),balance=0
    )

    map1,map2 = cv2.fisheye.initUndistortRectifyMap(
        K,D,np.eye(3),newK,(Wsrc,Hsrc),cv2.CV_32FC1
    )

    undist = cv2.remap(img,map1,map2,cv2.INTER_LINEAR)

    # ------------------------------------
    # undistort points
    # ------------------------------------

    pix_ud = cv2.fisheye.undistortPoints(
        pix.reshape(-1,1,2),
        K,D,P=newK
    ).reshape(-1,2)

    # ------------------------------------
    # use only bottom markers
    # ------------------------------------

    mask = world[:,1] > np.percentile(world[:,1],40)

    world_h = world[mask]
    pix_h   = pix_ud[mask]

    # ------------------------------------
    # homography
    # ------------------------------------

    H,_ = cv2.findHomography(
        pix_h,
        world_h,
        cv2.USAC_MAGSAC,
        2.0
    )

    # ------------------------------------
    # BEV size from markers
    # ------------------------------------

    xmin = world[:,0].min()
    xmax = world[:,0].max()

    ymin = -cfg.bev_offset_mm
    ymax = world[:,1].max()

    bev_width_mm  = xmax - xmin
    bev_height_mm = ymax - ymin

    width  = int(bev_width_mm / cfg.mm_per_pixel)
    height = int(bev_height_mm / cfg.mm_per_pixel)

    S = np.array([
        [1/cfg.mm_per_pixel,0,-xmin/cfg.mm_per_pixel],
        [0,-1/cfg.mm_per_pixel,ymax/cfg.mm_per_pixel],
        [0,0,1]
    ])

    H_bev = S @ H

    bev = cv2.warpPerspective(
        undist,
        H_bev,
        (width,height)
    )

    # ------------------------------------

    if cfg.show_plots:

        plt.figure(figsize=(8,4))
        plt.imshow(img,cmap="gray")
        plt.title("source")
        plt.axis("off")

        plt.figure(figsize=(8,4))
        plt.imshow(undist,cmap="gray")
        plt.title("undistorted")
        plt.axis("off")

        plt.figure(figsize=(8,4))
        plt.imshow(bev,cmap="gray")
        plt.title("BEV")
        plt.axis("off")

        plt.show()


if __name__=="__main__":
    main()