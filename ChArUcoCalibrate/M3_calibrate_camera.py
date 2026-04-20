from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

IMAGE_DIR = Path("ChArUcoCalibrate")

DICT_NAME = "DICT_6X6_250"

SQUARE_MM = 30.0
MARKER_MM = 21.0

SQUARES_X = 27
SQUARES_Y = 13

MIN_CORNERS = 20

OUTPUT = "camera_model.npz"


# ------------------------------------------------
# helpers
# ------------------------------------------------

def get_aruco_dict(name):

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown dictionary {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(aruco_dict):

    try:
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            float(SQUARE_MM),
            float(MARKER_MM),
            aruco_dict,
        )
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            float(SQUARE_MM),
            float(MARKER_MM),
            aruco_dict,
        )

    return board


# ------------------------------------------------
# main
# ------------------------------------------------

def main():

    aruco_dict = get_aruco_dict(DICT_NAME)
    board = make_board(aruco_dict)

    image_paths = sorted(
        p for p in IMAGE_DIR.glob("frame_*.jpg")
        if "_detect" not in p.name
    )

    print("Images:", len(image_paths))

    objpoints = []
    imgpoints = []

    img_size = None

    chess = board.getChessboardCorners()


    for path in image_paths:

        print("Processing:", path.name)

        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray.shape[::-1]

        # ---------------------------------------------
        # detect aruco
        # ---------------------------------------------

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict
        )

        if ids is None:
            print("  no markers")
            continue


        # ---------------------------------------------
        # charuco interpolation
        # ---------------------------------------------

        ret = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        charuco_corners = ret[1]
        charuco_ids = ret[2]

        if charuco_ids is None:
            print("  no charuco")
            continue

        if len(charuco_ids) < MIN_CORNERS:
            print("  too few corners:", len(charuco_ids))
            continue


        print("  corners:", len(charuco_ids))


        obj = chess[charuco_ids.flatten()].copy()

        obj = obj.reshape(-1,1,3).astype(np.float32)
        imgp = charuco_corners.reshape(-1,1,2).astype(np.float32)

        objpoints.append(obj)
        imgpoints.append(imgp)


    print()
    print("Used images:", len(objpoints))


    # ------------------------------------------------
    # fisheye calibration
    # ------------------------------------------------

    K = np.zeros((3,3))
    D = np.zeros((4,1))

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_FIX_K3
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_size,
        K,
        D,
        None,
        None,
        flags=flags
    )


    print()
    print("RMS:", rms)

    print()
    print("K:")
    print(K)

    print()
    print("D:")
    print(D)

    # ------------------------------------------------
    # reprojection test
    # ------------------------------------------------

    print()
    print("Reprojection test on first image")

    img = cv2.imread(str(image_paths[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    ret = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    charuco_corners = ret[1]
    charuco_ids = ret[2]

    if charuco_ids is None:
        raise RuntimeError("No charuco corners in reprojection test")

    # ---------------------------------------------
    # przygotowanie danych
    # ---------------------------------------------

    obj = chess[charuco_ids.flatten()].astype(np.float64)
    obj = obj.reshape(-1,1,3)

    imgp = charuco_corners.reshape(-1,1,2).astype(np.float64)

    # ---------------------------------------------
    # undistort points (dla solvePnP)
    # ---------------------------------------------

    undist = cv2.fisheye.undistortPoints(
        imgp,
        K,
        D,
        P=K
    )

    # ---------------------------------------------
    # solve pose
    # ---------------------------------------------

    ok, rvec, tvec = cv2.solvePnP(
        obj,
        undist,
        K,
        None
    )

    # ---------------------------------------------
    # project back (FISHEYE!)
    # ---------------------------------------------

    proj, _ = cv2.fisheye.projectPoints(
        obj,
        rvec,
        tvec,
        K,
        D
    )

    # reshape do porównania
    proj = proj.reshape(-1,2)
    imgp_flat = imgp.reshape(-1,2)

    err = np.linalg.norm(proj - imgp_flat, axis=1)

    print("mean reprojection error:", err.mean())
    print("max reprojection error:", err.max())

    # ---------------------------------------------
    # wizualizacja
    # ---------------------------------------------

    vis = img.copy()

    for p, q in zip(imgp_flat, proj):

        px, py = int(p[0]), int(p[1])
        qx, qy = int(q[0]), int(q[1])

        cv2.circle(vis, (px, py), 4, (0,255,0), -1)  # detected
        cv2.circle(vis, (qx, qy), 2, (0,0,255), -1)  # projected

    cv2.imwrite("reprojection_test.jpg", vis)

    print("saved reprojection_test.jpg")

    # ------------------------------------------------
    # undistort maps (przydatne dla BEV)
    # ------------------------------------------------

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, np.eye(3), balance=0
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, img_size, cv2.CV_32FC1
    )


    # ------------------------------------------------
    # save
    # ------------------------------------------------

    np.savez(
        OUTPUT,
        K=K,
        D=D,
        newK=newK,
        map1=map1,
        map2=map2,
        rms=rms
    )

    print()
    print("Saved camera model:", OUTPUT)


if __name__ == "__main__":
    main()