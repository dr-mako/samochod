# bev_ransac_homography.py
# ==============================================================
# BEV-RANSAC-HOMOGRAPHY
# Bird's Eye View from planar correspondences
# ==============================================================

# Description
# --------------------------------------------------------------
# This script generates a Bird's Eye View (BEV) image using
# planar homography estimated with the RANSAC algorithm.
#
# Pixel coordinates from the camera image are matched with
# known world coordinates from a ChArUco calibration board.
# A homography is estimated using RANSAC to reject outliers.
#
# The transformation is then used to generate a metric BEV
# image using cv2.warpPerspective.
#
#
# Pipeline
# --------------------------------------------------------------
#
# 1. Load source image
#
# 2. Load pixel ↔ world correspondences from CSV
#
#       pixel_x , pixel_y
#       world_x , world_y
#
# 3. Load camera calibration model
#
#       K      camera matrix
#       D      fisheye distortion coefficients
#
# 4. Undistort image
#
# 5. Undistort pixel coordinates
#
# 6. Estimate planar homography using
#
#       cv2.findHomography(..., RANSAC)
#
# 7. Define BEV coordinate system and metric scale
#
#       mm_per_pixel
#
# 8. Generate BEV image using
#
#       cv2.warpPerspective
#
#
# Properties
# --------------------------------------------------------------
#
# Advantages
#
# + simple and robust pipeline
# + resistant to outliers thanks to RANSAC
# + works with sparse point correspondences
# + produces metric BEV representation
#
#
# Limitations
#
# - assumes planar ground surface
# - accuracy depends on camera calibration quality
# - extrapolation outside the calibration area may introduce
#   geometric distortion
#
#
# Input
# --------------------------------------------------------------
#
# IMAGE
#     camera frame containing ChArUco board
#
# CSV
#     pixel ↔ world correspondences
#
# MODEL
#     saved fisheye camera calibration parameters
#
#
# Output
# --------------------------------------------------------------
#
# BEV image with metric scaling defined by:
#
#     mm_per_pixel
#
#
# Notes
# --------------------------------------------------------------
#
# This method represents a classical BEV pipeline based on
# planar homography estimation using the RANSAC algorithm.
#
# For improved robustness, newer estimators such as
# USAC_MAGSAC may be used instead of classical RANSAC.
#
# ==============================================================

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE = "ChArUcoCalibrate/frame_0012.jpg"
CSV = "multi_points.csv"
MODEL = "camera_model.npz"

MM_PER_PIXEL = 2.0


# ------------------------------------------------
# load csv
# ------------------------------------------------
def load_points(path):

    pix = []
    world = []

    with open(path) as f:

        r = csv.DictReader(f)

        for row in r:

            px = float(row["pixel_x"])
            py = float(row["pixel_y"])
            x = float(row["world_x"])
            y = float(row["world_y"])

            pix.append([px, py])
            world.append([x, y])

    return np.array(pix, np.float32), np.array(world, np.float32)


# ------------------------------------------------
# main
# ------------------------------------------------
def main():

    img = cv2.imread(IMAGE)
    if img is None:
        raise RuntimeError("image not loaded")

    data = np.load(MODEL)

    K = data["K"]
    D = data["D"]
    newK = data["newK"]
    map1 = data["map1"]
    map2 = data["map2"]

    # ---------------------------------------
    # undistort
    # ---------------------------------------
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # ---------------------------------------
    # load dataset
    # ---------------------------------------
    pix, world = load_points(CSV)

    print("points:", len(pix))

    # ---------------------------------------
    # undistort pixel coordinates
    # ---------------------------------------
    pix_ud = cv2.fisheye.undistortPoints(
        pix.reshape(-1,1,2),
        K,
        D,
        P=newK
    ).reshape(-1,2)

    # ---------------------------------------
    # homography
    # ---------------------------------------

    
    H, mask = cv2.findHomography(
        pix_ud,
        world,
        cv2.USAC_MAGSAC,
        1.5
    )

    print("inliers:", np.sum(mask))

    # ---------------------------------------
    # WORLD BOUNDS (AUTOMATIC)
    # ---------------------------------------
    xmin = np.min(world[:,0])
    xmax = np.max(world[:,0])
    ymin = np.min(world[:,1])
    ymax = np.max(world[:,1])

    margin = 30

    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    print("world range X:", xmin, xmax)
    print("world range Y:", ymin, ymax)

    width = int((xmax - xmin) / MM_PER_PIXEL)
    height = int((ymax - ymin) / MM_PER_PIXEL)

    print("BEV:", width, height)

    # ---------------------------------------
    # scale matrix
    # ---------------------------------------
    S = np.array([
        [1/MM_PER_PIXEL, 0, -xmin/MM_PER_PIXEL],
        [0, -1/MM_PER_PIXEL, ymax/MM_PER_PIXEL],
        [0, 0, 1]
    ])

    H_bev = S @ H

    # ---------------------------------------
    # warp
    # ---------------------------------------
    bev = cv2.warpPerspective(
        undist,
        H_bev,
        (width, height)        
    )

    # ---------------------------------------
    # grid
    # ---------------------------------------
    for y in range(0, height, 100):
        cv2.line(bev, (0,y), (width,y), (0,255,0), 1)

    for x in range(0, width, 100):
        cv2.line(bev, (x,0), (x,height), (0,255,0), 1)

    # ---------------------------------------
    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    plt.title("BEV")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()