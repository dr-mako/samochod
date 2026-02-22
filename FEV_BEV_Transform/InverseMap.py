from __future__ import annotations
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import LinearNDInterpolator
import cv2


# ============================================================
# CONFIG
# ============================================================

IMG_PATH = "frame_0021.jpg"
CLEAN_PTAB = "Clean_P_tab.mat"
OUT_MAP_INV = "map_inv.mat"


# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("Image not found")

Hsrc, Wsrc = img.shape


# ============================================================
# LOAD P_TAB
# ============================================================

S = loadmat(CLEAN_PTAB)
P_tab = np.asarray(S["Clean_P_tab"], dtype=np.float64)

pix_x    = P_tab[:, 0]
pix_y    = P_tab[:, 1]
square_x = P_tab[:, 2]
square_y = P_tab[:, 3]


# ============================================================
# DEDUP (jak w MATLAB)
# ============================================================

XY = np.stack([square_x, square_y], axis=1)
XYu, inv = np.unique(XY, axis=0, return_inverse=True)

pix_x_u = np.zeros(len(XYu))
pix_y_u = np.zeros(len(XYu))
cnt = np.zeros(len(XYu))

for i, g in enumerate(inv):
    pix_x_u[g] += pix_x[i]
    pix_y_u[g] += pix_y[i]
    cnt[g] += 1

pix_x_u /= np.maximum(cnt, 1)
pix_y_u /= np.maximum(cnt, 1)

square_x_u = XYu[:, 0]
square_y_u = XYu[:, 1]


# ============================================================
# INVERSE INTERPOLATORS (kamera -> BEV)
# ============================================================

pts_cam = np.stack([pix_x_u, pix_y_u], axis=1)

Cam2Bev_Fx = LinearNDInterpolator(
    pts_cam, square_x_u, fill_value=np.nan
)

Cam2Bev_Fy = LinearNDInterpolator(
    pts_cam, square_y_u, fill_value=np.nan
)


# ============================================================
# BUILD FULL INVERSE MAP
# ============================================================

pix_X, pix_Y = np.meshgrid(
    np.arange(Wsrc),
    np.arange(Hsrc)
)

Bev_X = Cam2Bev_Fx(pix_X, pix_Y)
Bev_Y = Cam2Bev_Fy(pix_X, pix_Y)

invalid = ~np.isfinite(Bev_X) | ~np.isfinite(Bev_Y)
Bev_X[invalid] = np.nan
Bev_Y[invalid] = np.nan


# ============================================================
# SAVE
# ============================================================

savemat(OUT_MAP_INV, {
    "Bev_X": Bev_X.astype(np.float32),
    "Bev_Y": Bev_Y.astype(np.float32),
})

print("Saved map_inv.mat (camera -> BEV)")