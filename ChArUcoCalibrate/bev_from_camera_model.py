# BEV IPM ray-plane intersection

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.aruco as aruco

IMAGE = "ChArUcoCalibrate/frame_0012.jpg"
MODEL = "ChArUcoCalibrate/camera_model.npz"

PIX_PER_MM = 2.0


# ------------------------------------------------
# pose
# ------------------------------------------------

def estimate_camera_pose(img, K, D):

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    board = aruco.CharucoBoard((27, 13), 30.0, 21.0, aruco_dict)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, K, D, None, None
    )

    return rvec, tvec


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    img = cv2.imread(IMAGE)
    data = np.load(MODEL)

    K = data["K"]
    D = data["D"]
    map1 = data["map1"]
    map2 = data["map2"]
    newK = data["newK"]

    # ---------------------------------------
    # pose
    # ---------------------------------------

    rvec, tvec = estimate_camera_pose(img, K, D)
    R, _ = cv2.Rodrigues(rvec)

    # ---------------------------------------
    # undistort (tylko do wizualizacji)
    # ---------------------------------------

    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    h, w = undist.shape[:2]

    # ---------------------------------------
    # płaszczyzna planszy
    # ---------------------------------------

    normal = R[:, 2]
    point = tvec.flatten()

    # ---------------------------------------
    # zakres (stały — stabilny)
    # ---------------------------------------

    xmin, xmax = 0, 800
    ymin, ymax = 0, 400

    width  = int((xmax - xmin) * PIX_PER_MM)
    height = int((ymax - ymin) * PIX_PER_MM)

    print("BEV size:", width, height)

    # ---------------------------------------
    # mapy
    # ---------------------------------------

    map_x = np.full((h, w), -1, np.float32)
    map_y = np.full((h, w), -1, np.float32)

    # ---------------------------------------
    # IPM (fisheye-correct rays)
    # ---------------------------------------

    for v in range(h):
        for u in range(w):

            # 🔥 klucz: fisheye undistort punktu
            pts = np.array([[[u, v]]], dtype=np.float32)

            undist_pt = cv2.fisheye.undistortPoints(
                pts,
                K,
                D,
                P=newK
            )

            x = undist_pt[0, 0, 0]
            y = undist_pt[0, 0, 1]

            ray = np.array([x, y, 1.0])

            denom = normal @ ray
            if abs(denom) < 1e-3:
                continue

            t = (normal @ point) / denom
            if t <= 0 or t > 2000:
                continue

            P = ray * t

            Pw = R.T @ (P - tvec.flatten())

            Xw = Pw[0]
            Yw = Pw[1]

            bev_x = int((Xw - xmin) * PIX_PER_MM)
            bev_y = int((ymax - Yw) * PIX_PER_MM)

            if 0 <= bev_x < width and 0 <= bev_y < height:
                map_x[v, u] = bev_x
                map_y[v, u] = bev_y

    # ---------------------------------------
    # składanie BEV
    # ---------------------------------------

    bev = np.zeros((height, width, 3), dtype=np.uint8)

    for v in range(h):
        for u in range(w):
            bx = int(map_x[v, u])
            by = int(map_y[v, u])
            if bx >= 0 and by >= 0:
                bev[by, bx] = undist[v, u]

    # ---------------------------------------

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    plt.title("BEV (FINAL)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()