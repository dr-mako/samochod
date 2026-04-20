# bulid_dense_dataset.py
# synteza rasnac i homografi metodą zagęszczenia csv
# synthetic + measured dataset
# grid densification

'''
używa wyłącznie punktów ChArUco z CSV
robi triangulację Delaunay w przestrzeni world
używa barycentrycznej interpolacji
tworzy mapę pixel → BEV
generuje obraz BEV przez cv2.remap
'''
from __future__ import annotations

import os
import csv
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from scipy.io import savemat


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

@dataclass(frozen=True)
class Config:

    img_path: str = "ChArUcoCalibrate/frame_0012.jpg"
    csv_path: str = "multi_points.csv"

    width_x: int = 840
    height_y: int = 420

    out_map_mat: str = "map.mat"

    show_plots: bool = True


# ------------------------------------------------
# CSV loader
# ------------------------------------------------

def load_csv_points(path, frame_name):

    pix_x = []
    pix_y = []
    world_x = []
    world_y = []

    with open(path) as f:

        r = csv.DictReader(f)

        for row in r:

            if row["image_name"] != frame_name:
                continue

            pix_x.append(float(row["pixel_x"]))
            pix_y.append(float(row["pixel_y"]))
            world_x.append(float(row["world_x"]))
            world_y.append(float(row["world_y"]))

    return (
        np.array(pix_x),
        np.array(pix_y),
        np.array(world_x),
        np.array(world_y),
    )


# ------------------------------------------------
# mat2gray
# ------------------------------------------------

def mat2gray(x):

    x = np.asarray(x, dtype=np.float32)

    mn = np.nanmin(x)
    mx = np.nanmax(x)

    if mx <= mn:
        return np.zeros_like(x)

    y = (x - mn) / (mx - mn)

    return np.clip(y, 0, 1)


# ------------------------------------------------
# main
# ------------------------------------------------

def main():

    cfg = Config()

    img = cv2.imread(cfg.img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError("image not loaded")

    Hsrc, Wsrc = img.shape

    original_01 = img.astype(np.float32) / 255.0

    frame_name = os.path.basename(cfg.img_path)

    # --------------------------------------------
    # load Charuco points
    # --------------------------------------------

    pix_x, pix_y, world_x, world_y = load_csv_points(
        cfg.csv_path,
        frame_name
    )

    print("Charuco points:", len(pix_x))

    world_pts = np.stack([world_x, world_y], axis=1)

    # --------------------------------------------
    # Delaunay triangulation in world space
    # --------------------------------------------

    tri = Delaunay(world_pts)

    # --------------------------------------------
    # world grid (BEV)
    # --------------------------------------------

    xmin = np.min(world_x)
    xmax = np.max(world_x)

    ymin = np.min(world_y)
    ymax = np.max(world_y)

    xs = np.linspace(xmin, xmax, cfg.width_x)
    ys = np.linspace(ymax, ymin, cfg.height_y)

    grid_x, grid_y = np.meshgrid(xs, ys)

    map_x = np.full_like(grid_x, np.nan, dtype=np.float32)
    map_y = np.full_like(grid_y, np.nan, dtype=np.float32)

    # --------------------------------------------
    # barycentric interpolation
    # --------------------------------------------

    for j in range(cfg.height_y):
        for i in range(cfg.width_x):

            wx = grid_x[j, i]
            wy = grid_y[j, i]

            simplex = tri.find_simplex([[wx, wy]])[0]

            if simplex < 0:
                continue

            verts = tri.simplices[simplex]

            T = tri.transform[simplex]

            bary = np.dot(T[:2], [wx, wy] - T[2])
            bary = np.append(bary, 1 - bary.sum())

            px = np.dot(bary, pix_x[verts])
            py = np.dot(bary, pix_y[verts])

            map_x[j, i] = px
            map_y[j, i] = py

    # --------------------------------------------
    # remove invalid pixels
    # --------------------------------------------

    invalid = (
        ~np.isfinite(map_x)
        | ~np.isfinite(map_y)
        | (map_x < 0)
        | (map_x > Wsrc-1)
        | (map_y < 0)
        | (map_y > Hsrc-1)
    )

    map_x[invalid] = -1
    map_y[invalid] = -1

    # --------------------------------------------
    # remap image
    # --------------------------------------------

    bev = cv2.remap(
        original_01,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    bev = mat2gray(bev)

    # --------------------------------------------
    # save MATLAB maps
    # --------------------------------------------

    Xmap_matlab = map_x.copy()
    Ymap_matlab = map_y.copy()

    valid = np.isfinite(Xmap_matlab) & np.isfinite(Ymap_matlab)

    Xmap_matlab[valid] += 1
    Ymap_matlab[valid] += 1

    savemat(
        cfg.out_map_mat,
        {
            "Xmap": Xmap_matlab,
            "Ymap": Ymap_matlab
        }
    )

    print("saved:", cfg.out_map_mat)

    # --------------------------------------------
    # plots
    # --------------------------------------------

    if cfg.show_plots:

        plt.figure(figsize=(12,6))
        plt.imshow(original_01, cmap="gray")
        plt.title("source image")
        plt.axis("off")

        plt.figure(figsize=(12,6))
        plt.imshow(bev, cmap="gray")
        plt.title("BEV (triangulation refinement)")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    main()