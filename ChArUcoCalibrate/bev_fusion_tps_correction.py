import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE = "ChArUcoCalibrate/frame_0012.jpg"

MAP_M1  = "ChArUcoCalibrate/map_interp.npz"
MAP_M2  = "ChArUcoCalibrate/map_homo.npz"
MAP_TPS = "ChArUcoCalibrate/map_tps.npz"

MODEL  = "ChArUcoCalibrate/camera_model.npz"


# ------------------------------------------------
# remap helper
# ------------------------------------------------

def remap_image(img, Xmap, Ymap):

    invalid = ~np.isfinite(Xmap) | ~np.isfinite(Ymap)

    map_x = Xmap.copy()
    map_y = Ymap.copy()

    map_x[invalid] = -1
    map_y[invalid] = -1

    bev = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return bev, ~invalid


# ------------------------------------------------
# main
# ------------------------------------------------

def main():

    # ---------------------------------------
    # load image
    # ---------------------------------------

    img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("image not loaded")

    original = img.astype(np.float32) / 255.0

    # ---------------------------------------
    # undistort (dla M2)
    # ---------------------------------------

    data = np.load(MODEL)
    map1 = data["map1"]
    map2 = data["map2"]

    undist = cv2.remap(original, map1, map2, cv2.INTER_LINEAR)

    # ---------------------------------------
    # load maps
    # ---------------------------------------

    data1 = np.load(MAP_M1)
    X1 = data1["Xmap"]
    Y1 = data1["Ymap"]

    data2 = np.load(MAP_M2)
    X2 = data2["Xmap"]
    Y2 = data2["Ymap"]

    data3 = np.load(MAP_TPS)
    X3 = data3["Xmap"]
    Y3 = data3["Ymap"]

    print("shapes:", X1.shape, X2.shape, X3.shape)

    # ---------------------------------------
    # remap
    # ---------------------------------------

    bev_M1, valid1 = remap_image(original, X1, Y1)
    bev_M2, valid2 = remap_image(undist,   X2, Y2)
    bev_TPS, valid_TPS = remap_image(original, X3, Y3)

    # ---------------------------------------
    # EXTRA (jak wcześniej)
    # ---------------------------------------

    extra_M1 = valid1 & (~valid2)

    # ---------------------------------------
    # FUSION STEP 1 (M2 + extension)
    # ---------------------------------------

    bev_fused = bev_M2.copy()
    bev_fused[extra_M1] = bev_M1[extra_M1]

    # ---------------------------------------
    # 🔥 FUSION STEP 2 (TPS correction - SMART)
    # ---------------------------------------

    # różnica TPS vs M2
    err = np.abs(bev_TPS - bev_M2)

    # delta (co chcemy dodać)
    delta = bev_TPS - bev_M2
    delta = np.clip(delta, -0.2, 0.2)

    # 🔥 NOWA MASKA (zamiast valid_TPS)
    alpha = 1.0 - (err / 0.2)
    alpha = np.clip(alpha, 0.0, 1.0)

    # wygładzenie
    alpha = cv2.GaussianBlur(alpha, (21,21), 0)

    # korekta
    bev_corrected = bev_fused + alpha * delta

    # ---------------------------------------
    # DEBUG
    # ---------------------------------------

    diff_M2_TPS = np.abs(bev_TPS - bev_M2)

    # ---------------------------------------
    # VISUAL
    # ---------------------------------------

    def crop_valid(img, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return img[y0:y1, x0:x1]

    valid_all = valid1 | valid2 | valid_TPS

    bev_M2_v = crop_valid(bev_M2, valid_all)
    bev_TPS_v = crop_valid(bev_TPS, valid_all)
    bev_corr_v = crop_valid(bev_corrected, valid_all)
    diff_v = crop_valid(diff_M2_TPS, valid_all)

    # ---------------------------------------

    plt.figure(figsize=(14,6))

    plt.subplot(2,2,1)
    plt.imshow(bev_M2_v, cmap="gray")
    plt.title("M2 (base)")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(bev_TPS_v, cmap="gray")
    plt.title("TPS")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(bev_corr_v, cmap="gray")
    plt.title("FUSED (M2 + TPS correction)")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(diff_v, cmap="inferno")
    plt.title("|TPS - M2|")
    plt.axis("off")

    plt.show()


# ------------------------------------------------

if __name__ == "__main__":
    main()