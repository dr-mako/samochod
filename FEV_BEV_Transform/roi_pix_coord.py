import csv
from pathlib import Path
import cv2

def append_csv_row(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def main():
    img_path = Path("frame_0020.jpg")
    out_dir = Path("roi_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "roi_pix_coord.csv"
    fieldnames = ["roi_id", "roi_file", "x_tl", "y_tl", "w", "h"]

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Nie mogę wczytać obrazu: {img_path}")

    H, W = img.shape[:2]

    print("Instrukcja:")
    print("- Zaznacz ROI myszką i zatwierdź ENTER lub SPACE.")
    print("- Anuluj ROI klawiszem 'c'.")
    print("- Zakończ program klawiszem 'q' lub ESC w oknie ROI.")
    print("- Alternatywnie: zamknij okno ROI (X) aby zakończyć.")
    print()
    print("CSV zapisuje współrzędne lewego górnego rogu ROI (x_tl, y_tl).")

    roi_id = 1

    while True:
        win_name = "Wybierz ROI (ENTER/SPACE=OK, c=cancel, q/ESC=exit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # selectROI zwraca (x, y, w, h) w pikselach ORYGINALNEGO obrazu,
        # gdzie (x, y) to lewy górny róg ROI.
        rect = cv2.selectROI(win_name, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win_name)

        x, y, w, h = map(int, rect)

        # Wyjście: użytkownik nacisnął q/ESC albo zamknął okno -> rect bywa (0,0,0,0)
        # Uwaga: (0,0,0,0) może też oznaczać "nic nie wybrano". Traktujemy jako koniec.
        if w == 0 or h == 0:
            print("Koniec (brak ROI / anulowanie / wyjście).")
            break

        # Zabezpieczenie granic
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        x = max(x, 0)
        y = max(y, 0)
        w = x2 - x
        h = y2 - y

        roi = img[y : y + h, x : x + w].copy()

        roi_file = f"frame_{roi_id}.jpg"
        roi_path = out_dir / roi_file
        cv2.imwrite(str(roi_path), roi)

        append_csv_row(
            csv_path,
            {
                "roi_id": roi_id,
                "roi_file": roi_file,
                "x_tl": x,
                "y_tl": y,
                "w": w,
                "h": h,
            },
            fieldnames,
        )

        print(
            f"ROI #{roi_id}: zapisano {roi_file}, "
            f"TL=(x={x}, y={y}), w={w}, h={h}"
        )

        roi_id += 1

    print(f"Gotowe. ROI zapisane w: {out_dir}")
    print(f"CSV: {csv_path}")
    print()
    print("Odzyskiwanie punktów klikniętych w ROI:")
    print("x_orig = x_tl + x_roi,  y_orig = y_tl + y_roi")


if __name__ == "__main__":
    main()
