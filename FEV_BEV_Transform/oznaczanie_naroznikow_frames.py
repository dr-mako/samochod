import csv
import os

import cv2
import matplotlib.pyplot as plt

# ============================================================
# USTAWIENIA
# ============================================================
ROI_DIR = "roi_frames_20"
ROI_IMAGE_NAME = "frame_3.jpg"  # zmieniaj na kolejne
ROI_COORD_CSV = "roi_pix_coord.csv"
OUT_CSV = "wykryte_narozniki.csv"

fig_width = 4 * 10
fig_height = 3 * 10


def read_roi_row(roi_dir: str, roi_coord_csv: str, roi_file: str) -> dict:
    path = os.path.join(roi_dir, roi_coord_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku ROI CSV: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"roi_file", "x_tl", "y_tl", "w", "h"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV {path} musi mieć kolumny: roi_file, x_tl, y_tl, w, h"
            )

        for row in reader:
            if row["roi_file"] == roi_file:
                return {
                    "x_tl": int(float(row["x_tl"])),
                    "y_tl": int(float(row["y_tl"])),
                    "w": int(float(row["w"])),
                    "h": int(float(row["h"])),
                }

    raise ValueError(f"Nie znaleziono wpisu roi_file='{roi_file}' w {path}")


print("Punkty wprowadzaj kolumnami po liniach pionowych od dołu do góry")
square_x = input("Podaj współrzędną X0 kwadratów: ")
square_y0 = input("Podaj początkową wspłrzędną Y0 (najlepiej 0): ")

try:
    square_x = float(square_x)
    square_y0 = float(square_y0)
except ValueError:
    print("Podana wartość nie jest liczbą. Program zostanie zakończony.")
    raise SystemExit(1)

if not os.path.isdir(ROI_DIR):
    print(f"Nie znaleziono katalogu: {ROI_DIR}")
    raise SystemExit(1)

os.chdir(ROI_DIR)

try:
    roi_info = read_roi_row(".", ROI_COORD_CSV, ROI_IMAGE_NAME)
except Exception as e:
    print(f"Błąd odczytu ROI: {e}")
    raise SystemExit(1)

x_tl = roi_info["x_tl"]
y_tl = roi_info["y_tl"]
roi_w = roi_info["w"]
roi_h = roi_info["h"]

print(f"Pracujesz na ROI: {ROI_IMAGE_NAME}")
print(f"ROI w oryginale: TL=(x_tl={x_tl}, y_tl={y_tl}), w={roi_w}, h={roi_h}")
print("Kliknięcia zapisuję w oryginale: x=x_tl+x_roi, y=y_tl+y_roi")

image = cv2.imread(ROI_IMAGE_NAME)
if image is None:
    print(f"Nie udało się wczytać obrazu: {ROI_IMAGE_NAME}")
    raise SystemExit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = gray

coords = []  # kliknięcia w układzie ROI

plt.figure(figsize=(fig_width, fig_height))
plt.imshow(img, cmap="gray")
plt.title("Kliknij na narożniki (prawy przycisk myszy, aby zakończyć)")


def on_click(event):
    if event.button == 3:
        plt.close()
        return
    if event.xdata is None or event.ydata is None:
        return
    coords.append([event.xdata, event.ydata])
    plt.plot(event.xdata, event.ydata, "go", markersize=3)
    plt.draw()


cid = plt.gcf().canvas.mpl_connect("button_press_event", on_click)
plt.show()

# Zapis do OUT_CSV w układzie ORYGINAŁU
if coords:
    rows = []
    for lp_square, (x_roi, y_roi) in enumerate(coords):
        calculated_y = square_y0 + lp_square
        x_orig = float(x_roi) + float(x_tl)
        y_orig = float(y_roi) + float(y_tl)
        rows.append([x_orig, y_orig, square_x, calculated_y])

    mode = "a" if os.path.exists(OUT_CSV) else "w"
    with open(OUT_CSV, mode=mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["pixel_x", "pixel_y", "square_x", "square_y"])
        w.writerows(rows)

# ============================================================
# POPRAWIONY PODGLĄD:
# - wczytujemy współrzędne ORYGINALNE
# - filtrujemy punkty, które należą do TEGO ROI po oryginalnych granicach
# - przeliczamy na lokalne: x_loc = x_orig - x_tl, y_loc = y_orig - y_tl
# ============================================================
if os.path.exists(OUT_CSV):
    saved_coords = []
    with open(OUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            saved_coords.append(
                [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            )

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(img, cmap="gray")
    plt.title("Oznaczone punkty (podgląd dla bieżącego ROI)")

    x0, y0 = x_tl, y_tl
    x1, y1 = x_tl + roi_w, y_tl + roi_h

    for x_orig, y_orig, sx, sy in saved_coords:
        # filtr: punkt musi należeć do tego ROI w układzie ORYGINALNYM
        if not (x0 <= x_orig < x1 and y0 <= y_orig < y1):
            continue

        # przeliczenie na układ lokalny ROI
        x_loc = x_orig - x_tl
        y_loc = y_orig - y_tl

        plt.plot(x_loc, y_loc, "ro", markersize=6)
        plt.text(
            x_loc,
            y_loc,
            f"({sx:.0f}, {sy:.0f})",
            color="yellow",
            fontsize=10,
        )

    out_path = "wykryte_narozniki_podglad.jpg"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.show()
else:
    print(f"Nie znaleziono pliku CSV: {OUT_CSV}")