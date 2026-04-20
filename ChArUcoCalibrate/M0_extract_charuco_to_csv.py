from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------
# helpers
# --------------------------------------------------------

def get_aruco_dict(name: str):

    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_charuco_board(sx, sy, square_mm, marker_mm, aruco_dict):

    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (sx, sy),
            square_mm,
            marker_mm,
            aruco_dict
        )

    return cv2.aruco.CharucoBoard_create(
        sx,
        sy,
        square_mm,
        marker_mm,
        aruco_dict
    )


def detect_markers(gray, aruco_dict):

    params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=params
        )

    return corners, ids


# --------------------------------------------------------
# main
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default="ChArUcoCalibrate/frame_0012.jpg")
    parser.add_argument("--out", default="P_map.csv")

    parser.add_argument("--dict", default="DICT_6X6_250")

    parser.add_argument("--squares-x", type=int, default=27)
    parser.add_argument("--squares-y", type=int, default=13)

    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=21.0)

    parser.add_argument(
        "--units",
        choices=["mm", "squares"],
        default="squares"
    )

    args = parser.parse_args()

    # ----------------------------------------------------
    # load image
    # ----------------------------------------------------

    img = cv2.imread(args.image)

    if img is None:
        raise RuntimeError("image not loaded")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------------------
    # board
    # ----------------------------------------------------

    aruco_dict = get_aruco_dict(args.dict)

    board = make_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        aruco_dict
    )

    # ----------------------------------------------------
    # detect markers
    # ----------------------------------------------------

    marker_corners, marker_ids = detect_markers(gray, aruco_dict)

    if marker_ids is None:
        raise RuntimeError("no markers detected")

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board
    )

    if charuco_corners is None:
        raise RuntimeError("charuco detection failed")

    # ----------------------------------------------------
    # board coordinates
    # ----------------------------------------------------

    board_pts = board.getChessboardCorners().astype(np.float64)

    rows = []

    for i in range(len(charuco_ids)):

        cid = int(charuco_ids[i,0])

        px = float(charuco_corners[i,0,0])
        py = float(charuco_corners[i,0,1])

        obj = board_pts[cid]

        x_mm = float(obj[0])
        y_mm = float(obj[1])

        if args.units == "mm":

            world_x = x_mm
            world_y = y_mm

        else:

            world_x = x_mm / args.square_mm
            world_y = y_mm / args.square_mm

        rows.append((px, py, world_x, world_y))

    # ----------------------------------------------------
    # save CSV
    # ----------------------------------------------------

    with open(args.out, "w", newline="") as f:

        w = csv.writer(f)

        w.writerow([
            "pixel_x",
            "pixel_y",
            "square_x",
            "square_y"
        ])

        w.writerows(rows)

    print("saved:", args.out)
    print("corners:", len(rows))

    # ----------------------------------------------------
    # debug
    # ----------------------------------------------------

    xs = [r[2] for r in rows]
    ys = [r[3] for r in rows]

    print("world range X:", min(xs), max(xs))
    print("world range Y:", min(ys), max(ys))

    # ----------------------------------------------------
    # visualization
    # ----------------------------------------------------

    vis = img.copy()

    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    out = Path(args.image).with_name(
        Path(args.image).stem + "_detected.jpg"
    )

    cv2.imwrite(str(out), vis)

    print("debug image:", out)


if __name__ == "__main__":
    main()