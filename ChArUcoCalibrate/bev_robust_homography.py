# bev_robust_homography.py
# ==============================================================
# BEV-CHARUCO-MAGSAC
# Robust Bird's Eye View from ChArUco board
# ==============================================================

# Description
# --------------------------------------------------------------
# This script generates a Bird's Eye View (BEV) image using
# robust homography estimation from ChArUco calibration points.
#
# The program loads detected ChArUco corner correspondences
# from CSV, removes fisheye distortion using a precomputed
# camera model, estimates a robust homography using
# USAC_MAGSAC (modern RANSAC variant), and generates a BEV
# image with metric scaling.
#
# Additionally, the script reconstructs a dense ChArUco grid
# using the estimated homography and saves the generated
# pixel↔world correspondences to a CSV file.
#
#
# Pipeline
# --------------------------------------------------------------
#
# 1. Load calibration image
#
# 2. Load ChArUco corner correspondences from CSV
#
#       pixel_x , pixel_y   → image coordinates
#       world_x , world_y   → board coordinates
#
# 3. Load fisheye camera model
#
#       K      camera matrix
#       D      distortion coefficients
#       newK   rectified camera matrix
#
# 4. Undistort image
#
# 5. Undistort pixel coordinates
#
# 6. Select bottom board markers
#    (markers closest to camera improve BEV stability)
#
# 7. Estimate robust homography using
#
#       cv2.USAC_MAGSAC
#
# 8. Reconstruct dense ChArUco grid
#
#       board → pixel
#
#
# 9. Generate BEV image using metric scaling
#
#       mm_per_pixel
#
# 10. Visualize BEV with reference grid
#
#
# Properties
# --------------------------------------------------------------
#
# Advantages
#
# + robust homography estimation (USAC_MAGSAC)
# + fisheye distortion correction
# + accurate BEV geometry
# + reconstruction of dense board grid
# + metric BEV scaling
#
#
# Limitations
#
# - assumes planar ground surface
# - accuracy depends on camera calibration
# - extrapolation outside board area may degrade
#
#
# Input
# --------------------------------------------------------------
#
# IMAGE
#     calibration frame containing ChArUco board
#
# CSV
#     detected ChArUco correspondences
#
# MODEL
#     saved camera calibration parameters
#
#
# Output
# --------------------------------------------------------------
#
# dense_points.csv
#
# dense pixel ↔ world correspondences reconstructed from
# the estimated homography
#
# BEV image preview
#
#
# Notes
# --------------------------------------------------------------
#
# Homography is estimated using USAC_MAGSAC which is a modern
# robust estimator outperforming classical RANSAC.
#
# This method produces stable BEV results for wide-angle cameras
# such as IMX219.
#
# ==============================================================


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


IMAGE = "ChArUcoCalibrate/frame_0012.jpg"
CSV = "multi_points.csv"
MODEL = "camera_model.npz"

MM_PER_PIXEL = 2.0

SQUARES_X = 27
SQUARES_Y = 13
SQUARE_MM = 30
MARKER_MM = 21

OUT_CSV = "dense_points.csv"

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

    # ---------------------------------------
    # load camera model
    # ---------------------------------------

    data = np.load(MODEL)

    K = data["K"]
    D = data["D"]
    newK = data["newK"]
    map1 = data["map1"]
    map2 = data["map2"]

    # ---------------------------------------
    # undistort image
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
        pix.reshape(-1, 1, 2),
        K,
        D,
        P=newK
    ).reshape(-1, 2)

    # ---------------------------------------
    # use only bottom markers
    # ---------------------------------------

    threshold = np.percentile(world[:,1], 60)

    mask_bottom = world[:,1] > threshold

    pix_h = pix_ud[mask_bottom]
    world_h = world[mask_bottom]

    print("bottom points:", len(pix_h))

    # ---------------------------------------
    # homography
    # ---------------------------------------

    H, mask = cv2.findHomography(
        pix_h,
        world_h,
        cv2.USAC_MAGSAC,
        2.0
    )

    print("inliers:", np.sum(mask))

    

    # ---------------------------------------
    # world bounds
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
    # show
    # ---------------------------------------

    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    plt.title("undistorted")
    plt.axis("off")

    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    plt.title("BEV")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()