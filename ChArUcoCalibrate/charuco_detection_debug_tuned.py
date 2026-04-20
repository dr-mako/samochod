from pathlib import Path
import cv2
import numpy as np


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

IMAGE = "ChArUcoCalibrate/frame_0012.jpg"

DICT = cv2.aruco.DICT_6X6_250

SQUARES_X = 27
SQUARES_Y = 13
SQUARE_MM = 30
MARKER_MM = 21


# ------------------------------------------------
# main
# ------------------------------------------------

def main():

    img = cv2.imread(IMAGE)
    if img is None:
        raise RuntimeError("image not loaded")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM,
        MARKER_MM,
        aruco_dict
    )

    # =================================================
    # 🔴 DEFAULT DETECTOR
    # =================================================

    corners_d, ids_d, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids_d is not None:
        _, ch_corners_d, ch_ids_d = cv2.aruco.interpolateCornersCharuco(
            corners_d, ids_d, gray, board
        )
        n_charuco_d = 0 if ch_ids_d is None else len(ch_ids_d)
    else:
        n_charuco_d = 0

    print("DEFAULT markers:", 0 if ids_d is None else len(ids_d))
    print("DEFAULT charuco:", n_charuco_d)


    vis_default = img.copy()
    if ids_d is not None:
        cv2.aruco.drawDetectedMarkers(vis_default, corners_d, ids_d)

    if ch_ids_d is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            vis_default, ch_corners_d, ch_ids_d, (0,255,0)
        )


    # =================================================
    # 🟢 TUNED DETECTOR
    # =================================================

    params = cv2.aruco.DetectorParameters()

    # 🔥 KLUCZOWE TUNINGI

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4

    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0

    params.polygonalApproxAccuracyRate = 0.03

    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners_t, ids_t, _ = detector.detectMarkers(gray)

    if ids_t is not None:
        _, ch_corners_t, ch_ids_t = cv2.aruco.interpolateCornersCharuco(
            corners_t, ids_t, gray, board
        )
        n_charuco_t = 0 if ch_ids_t is None else len(ch_ids_t)
    else:
        n_charuco_t = 0

    print("TUNED markers:", 0 if ids_t is None else len(ids_t))
    print("TUNED charuco:", n_charuco_t)


    vis_tuned = img.copy()
    if ids_t is not None:
        cv2.aruco.drawDetectedMarkers(vis_tuned, corners_t, ids_t)

    if ch_ids_t is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            vis_tuned, ch_corners_t, ch_ids_t, (0,255,0)
        )


    # =================================================
    # SHOW
    # =================================================

    vis_default = cv2.cvtColor(vis_default, cv2.COLOR_BGR2RGB)
    vis_tuned   = cv2.cvtColor(vis_tuned,   cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(vis_default)
    plt.title("DEFAULT")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(vis_tuned)
    plt.title("TUNED")
    plt.axis("off")

    plt.show()


# ------------------------------------------------

if __name__ == "__main__":
    main()