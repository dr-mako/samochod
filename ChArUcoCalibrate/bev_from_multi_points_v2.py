import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV = "multi_points.csv"
MODEL = "camera_model.npz"
IMAGE_DIR = Path("ChArUcoCalibrate")

MM_PER_PIXEL = 2.0


# ------------------------------------------------
# load csv
# ------------------------------------------------
def load_points(path):

    pix = []
    world = []
    image_names = []

    with open(path) as f:

        r = csv.DictReader(f)

        for row in r:

            px = float(row["pixel_x"])
            py = float(row["pixel_y"])
            x = float(row["world_x"])
            y = float(row["world_y"])
            name = row["image_name"]

            pix.append([px, py])
            world.append([x, y])
            image_names.append(name)

    pix = np.array(pix, np.float32)
    world = np.array(world, np.float32)
    image_names = np.array(image_names)

    image_list = sorted(set(image_names))

    return pix, world, image_names, image_list


# ------------------------------------------------
# main
# ------------------------------------------------
def main():

    data = np.load(MODEL)

    K = data["K"]
    D = data["D"]
    newK = data["newK"]
    map1 = data["map1"]
    map2 = data["map2"]

    # ---------------------------------------
    # load dataset
    # ---------------------------------------
    pix, world, image_names, image_list = load_points(CSV)

    print("points:", len(pix))
    print("images:", image_list)

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

    # ---------------------------------------
    # global BEV
    # ---------------------------------------
    bev_global = np.zeros((height, width, 3), np.uint8)

    # ---------------------------------------
    # warp each image
    # ---------------------------------------
    for name in image_list:

        print("warp:", name)

        idx = image_names == name

        pix_i = pix_ud[idx]
        world_i = world[idx]

        if len(pix_i) < 4:
            print("too few points")
            continue

        H_i, mask = cv2.findHomography(
            pix_i,
            world_i,
            cv2.RANSAC,
            3.0
        )

        img = cv2.imread(str(IMAGE_DIR / name))

        if img is None:
            print("image not loaded:", name)
            continue

        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        H_bev_i = S @ H_i

        bev = cv2.warpPerspective(
            undist,
            H_bev_i,
            (width, height)
        )

        mask_img = bev.sum(axis=2) > 0

        bev_global[mask_img] = bev[mask_img]

    # ---------------------------------------
    # grid
    # ---------------------------------------
    for y in range(0, height, 100):
        cv2.line(bev_global, (0,y), (width,y), (0,255,0), 1)

    for x in range(0, width, 100):
        cv2.line(bev_global, (x,0), (x,height), (0,255,0), 1)

    # ---------------------------------------
    # show
    # ---------------------------------------
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(bev_global, cv2.COLOR_BGR2RGB))
    plt.title("BEV multi-frame")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()