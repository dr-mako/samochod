import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE = "ChArUcoCalibrate/frame_0012.jpg"

MAP_M1 = "ChArUcoCalibrate/map_interp.npz"
MAP_M2 = "ChArUcoCalibrate/map_homo.npz"
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

    print("shapes:", X1.shape, X2.shape)

    if X1.shape != X2.shape:
        raise RuntimeError("maps must have same shape")

    # ---------------------------------------
    # remap independently
    # ---------------------------------------

    bev_M1, valid1 = remap_image(original, X1, Y1)
    bev_M2, valid2 = remap_image(undist,   X2, Y2)

    # ---------------------------------------
    # 🔥 EXTRA REGION (KLUCZ)
    # ---------------------------------------

    extra_M1 = valid1 & (~valid2)

    print("M1 extra pixels:", np.count_nonzero(extra_M1))
    print("ratio:", np.count_nonzero(extra_M1) / extra_M1.size)

    # wartości w tym regionie
    if np.any(extra_M1):
        print("M1 values in extra region:",
              bev_M1[extra_M1].min(),
              bev_M1[extra_M1].max())

    # ---------------------------------------
    # bounding box
    # ---------------------------------------

    ys, xs = np.where(extra_M1)

    if len(ys) > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        print("bbox:", y0, y1, x0, x1)
    else:
        print("NO EXTRA REGION")
        return

    # ---------------------------------------
    # FUSION (tylko extra!)
    # ---------------------------------------

    bev_fused = bev_M2.copy()
    bev_fused[extra_M1] = bev_M1[extra_M1]

    # ---------------------------------------
    # DIFFERENCE MAP
    # ---------------------------------------

    diff = np.abs(bev_M1 - bev_M2)

    # ---------------------------------------
    # PREPARE BETTER VISUALIZATION
    # ---------------------------------------

    def crop_valid(img, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return img[y0:y1, x0:x1]

    # wspólny obszar danych
    valid_all = valid1 | valid2

    # crop wszystkiego
    bev_M2_v = crop_valid(bev_M2, valid_all)
    bev_M1_v = crop_valid(bev_M1, valid_all)
    bev_fused_v = crop_valid(bev_fused, valid_all)
    extra_M1_v = crop_valid(extra_M1, valid_all)
    diff_v = crop_valid(diff, valid_all)

    # kolorowy overlay (lepszy niż grayscale)
    overlay_color = cv2.cvtColor((bev_M2_v*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay_color[extra_M1_v] = [255, 0, 0]


    # ---------------------------------------
    # VISUALIZACJA
    # ---------------------------------------

    plt.figure(figsize=(14,6))

    plt.subplot(2,3,1)
    plt.imshow(bev_M2_v, cmap="gray")
    plt.title("M2 (base)")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(bev_M1_v, cmap="gray")
    plt.title("M1 (interp)")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(bev_fused_v, cmap="gray")
    plt.title("FUSED (extra only)")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(extra_M1_v, cmap="gray")
    plt.title("extra_M1 mask")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(diff_v, cmap="inferno")  # 🔥 lepsze niż jet
    plt.title("difference |M1-M2|")
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.imshow(overlay_color)  # 🔥 kolor zamiast gray
    plt.title("extra over M2 (RED)")
    plt.axis("off")

    plt.show()

    plt.imshow(bev_fused_v, cmap="gray")
    plt.title("FUSED (extra only)")
    plt.axis("off")

    plt.show()

    # ---------------------------------------
    # ZOOM NA EXTRA REGION
    # ---------------------------------------

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(bev_M2[y0:y1, x0:x1], cmap="gray")
    plt.title("M2 zoom")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(bev_M1[y0:y1, x0:x1], cmap="gray")
    plt.title("M1 zoom")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(bev_fused[y0:y1, x0:x1], cmap="gray")
    plt.title("FUSED zoom")
    plt.axis("off")

    plt.show()


    
# ------------------------------------------------

if __name__ == "__main__":
    main()