from __future__ import annotations

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE = "ChArUcoCalibrate/frame_0012.jpg"
CSV   = "ChArUcoCalibrate/P_map.csv"

MM_PER_PIXEL = 2.0

SQUARE_MM = 30


def load_csv(path):

    pix = []
    world = []

    with open(path) as f:

        r = csv.DictReader(f)

        for row in r:

            px = float(row["pixel_x"])
            py = float(row["pixel_y"])

            sx = float(row["square_z"])
            sy = float(row["square_y"])

            pix.append([px,py])
            world.append([sx,sy])

    pix = np.array(pix,np.float32)
    world = np.array(world,np.float32)

    return pix,world


def main():

    img = cv2.imread(IMAGE)

    if img is None:
        raise RuntimeError("image not loaded")


    pix,world = load_csv(CSV)

    world *= SQUARE_MM


    # ------------------------------------------------
    # CENTER BOARD (ważne!)
    # ------------------------------------------------

    cx = world[:,0].mean()
    cy = world[:,1].mean()

    world[:,0] -= cx
    world[:,1] -= cy


    # ------------------------------------------------
    # HOMOGRAPHY
    # ------------------------------------------------

    H,_ = cv2.findHomography(
        pix,
        world
    )


    # ------------------------------------------------
    # WORLD LIMITS
    # ------------------------------------------------

    xmin = world[:,0].min()
    xmax = world[:,0].max()

    ymin = world[:,1].min()
    ymax = world[:,1].max()

    print("world X:",xmin,xmax)
    print("world Y:",ymin,ymax)


    # ------------------------------------------------
    # BEV SIZE
    # ------------------------------------------------

    width  = int((xmax-xmin)/MM_PER_PIXEL)
    height = int((ymax-ymin)/MM_PER_PIXEL)

    print("BEV size:",width,height)


    # ------------------------------------------------
    # SCALE MATRIX
    # ------------------------------------------------

    S = np.array([

        [1/MM_PER_PIXEL,0,-xmin/MM_PER_PIXEL],
        [0,-1/MM_PER_PIXEL,ymax/MM_PER_PIXEL],
        [0,0,1]

    ])


    Hbev = S @ H


    # ------------------------------------------------
    # WARP
    # ------------------------------------------------

    bev = cv2.warpPerspective(
        img,
        Hbev,
        (width,height)
    )


    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title("source")

    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(bev,cv2.COLOR_BGR2RGB))
    plt.title("BEV")

    plt.show()


if __name__=="__main__":
    main()