from pathlib import Path
import csv
import cv2
import numpy as np


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

IMAGE_DIR = Path("ChArUcoCalibrate")
OUT = "multi_points_tuned.csv"

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
# main
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

    # ------------------------------------------------
    # 🔥 TUNED DETECTOR
    # ------------------------------------------------

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

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    rows = []

    images = sorted(
        p for p in IMAGE_DIR.glob("*.jpg")
        if "_detect" not in p.name
        and int(p.stem.split("_")[1]) in FRAME_OFFSETS
    )

    print("images:", len(images))

    total_markers = 0
    total_charuco = 0

    for path in images:

        frame_id = int(path.stem.split("_")[1])
        offset_mm = FRAME_OFFSETS[frame_id] * SQUARE_MM

        print("processing:", path.name, "offset:", offset_mm)

        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------
        # DETECTION (TUNED)
        # ------------------------------------------------

        corners, ids, _ = detector.detectMarkers(gray)

        n_markers = 0 if ids is None else len(ids)
        total_markers += n_markers

        if ids is None:
            continue

        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        if ch_ids is None:
            print("  no charuco corners")
            continue

        ch_ids = ch_ids.flatten()

        if len(ch_ids) < MIN_CORNERS:
            print("  too few corners:", len(ch_ids))
            continue

        total_charuco += len(ch_ids)

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

    print("total markers:", total_markers)
    print("total charuco:", total_charuco)
    print("points:", len(rows))

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