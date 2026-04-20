#adaptive detector selection
from pathlib import Path
import csv
import cv2
import numpy as np


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

IMAGE_DIR = Path("ChArUcoCalibrate")
OUT = "multi_points_hybrid.csv"

DICT = cv2.aruco.DICT_6X6_250

SQUARES_X = 27
SQUARES_Y = 13
SQUARE_MM = 30
MARKER_MM = 21

MIN_CORNERS = 20

FRAME_OFFSETS = {
    12: 0,
    13: 8,
    14: -8
}


# ------------------------------------------------
# DETECTORS
# ------------------------------------------------

def create_default_detector(aruco_dict):
    return cv2.aruco.ArucoDetector(aruco_dict)


def create_tuned_detector(aruco_dict):

    params = cv2.aruco.DetectorParameters()

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

    return cv2.aruco.ArucoDetector(aruco_dict, params)


# ------------------------------------------------
# TRY DETECTION
# ------------------------------------------------

def detect_charuco(detector, gray, board):

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None, 0

    ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if ch_ids is None:
        return None, None, len(ids)

    ch_ids = ch_ids.flatten()

    return ch_corners, ch_ids, len(ids)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM,
        MARKER_MM,
        aruco_dict
    )

    chess = board.getChessboardCorners()

    detector_default = create_default_detector(aruco_dict)
    detector_tuned   = create_tuned_detector(aruco_dict)

    rows = []

    images = sorted(
        p for p in IMAGE_DIR.glob("frame_*.jpg")
        if "_detect" not in p.name
        and int(p.stem.split("_")[1]) in FRAME_OFFSETS
    )

    print("images:", len(images))

    stats = {
        "tuned_used": 0,
        "fallback_used": 0,
        "failed": 0
    }

    for path in images:

        frame_id = int(path.stem.split("_")[1])
        offset_mm = FRAME_OFFSETS[frame_id] * SQUARE_MM

        print("\nprocessing:", path.name)

        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------
        # 1️ TRY TUNED
        # ---------------------------------------

        ch_corners, ch_ids, n_markers = detect_charuco(
            detector_tuned, gray, board
        )

        if ch_ids is not None and len(ch_ids) >= MIN_CORNERS:
            print(f"  tuned OK | markers={n_markers} charuco={len(ch_ids)}")
            stats["tuned_used"] += 1

        else:
            print("  tuned FAILED → fallback")

            # ---------------------------------------
            # 2 FALLBACK DEFAULT
            # ---------------------------------------

            ch_corners, ch_ids, n_markers = detect_charuco(
                detector_default, gray, board
            )

            if ch_ids is None or len(ch_ids) < MIN_CORNERS:
                print("  fallback FAILED")
                stats["failed"] += 1
                continue

            print(f"  fallback OK | markers={n_markers} charuco={len(ch_ids)}")
            stats["fallback_used"] += 1

        # ---------------------------------------
        # SAVE POINTS
        # ---------------------------------------

        image_name = path.name

        for i in range(len(ch_ids)):

            cid = int(ch_ids[i])

            px = float(ch_corners[i,0,0])
            py = float(ch_corners[i,0,1])

            obj = chess[cid]

            x = float(obj[0]) + offset_mm
            y = float(obj[1])

            rows.append((
                image_name,
                px,
                py,
                x,
                y,
                offset_mm
            ))

    print("\n--- SUMMARY ---")
    print("tuned used   :", stats["tuned_used"])
    print("fallback used:", stats["fallback_used"])
    print("failed       :", stats["failed"])
    print("total points :", len(rows))

    # ------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------

    with open(OUT, "w", newline="") as f:

        w = csv.writer(f)

        w.writerow([
            "image_name",
            "pixel_x",
            "pixel_y",
            "world_x",
            "world_y",
            "offset_x_mm"
        ])

        w.writerows(rows)

    print("saved:", OUT)


# ------------------------------------------------

if __name__ == "__main__":
    main()