# program do recznego oznaczania narożników szachownicy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Parametry rozmiaru wykresu
fig_width = 4 * 10  # Szerokość w calach
fig_height = 3 * 10  # Wysokość w calach

# Pobranie współrzędnych x i y0 od użytkownika
print("Punkty wprowadzaj kolumnami po liniach pionowych od dołu do góry")
square_x = input("Podaj współrzędną X0 kwadratów: ")
square_y0 = input("Podaj początkową wspłrzędną Y0 (najlepiej 0):")

# Sprawdzenie, czy wprowadzone wartości są liczbami
try:
    square_x = float(square_x)
    square_y0 = float(square_y0)
except ValueError:
    print("Podana wartość nie jest liczbą. Program zostanie zakończony.")
    exit()

# Wczytanie obrazu
# image_path = "roi_image.jpg"
image_path = "frame_0021.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"Nie udało się wczytać obrazu: {image_path}")
    exit()

# Konwersja na skalę szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = gray

coords = []  # Lista wspórzędnych kliknięć

# Ustawienie rozmiaru okna i wyświetlenie obrazu
plt.figure(figsize=(fig_width, fig_height))
plt.imshow(img, cmap="gray")
plt.title("Kliknij na narożniki (prawy przycisk myszy, aby zakończyć)")

# Funkcja do zbierania kliknięć
def on_click(event):
    if event.button == 3:  # Prawy przycisk myszy
        plt.close()  # Zamyka okno, co kończy program
    else:
        if event.xdata is None or event.ydata is None:
            return  # klik poza osią
        coords.append([event.xdata, event.ydata])  # Zapisz kliknięte wspłrzędne
        plt.plot(event.xdata, event.ydata, "go", markersize=3)  # Narysuj punkt
        plt.draw()  # Odśwież wykres

# Ustawienie zdarzenia klikniccia
cid = plt.gcf().canvas.mpl_connect("button_press_event", on_click)

# Oczekiwanie na zakończenie
plt.show()

# Po zakończeniu kliknięć zapisujemy dane
csv_file = "wykryte punkty naroznikow.csv"
if coords:
    coords_with_square_xy = []
    for lp_square, (x, y) in enumerate(coords):
        calculated_y = square_y0 + lp_square
        coords_with_square_xy.append([x, y, square_x, calculated_y])

    mode = "a" if os.path.exists(csv_file) else "w"
    with open(csv_file, mode=mode, newline="") as file:
        writer = csv.writer(file)
        if mode == "w":  # Dodaj nagłówki, jeśli plik jest nowy
            writer.writerow(["pixel_x", "pixel_y", "square_x", "square_y"])
        writer.writerows(coords_with_square_xy)

# Otwórz plik CSV i narysuj punkty na nowym wykresie + ZAPISZ OBRAZEK Z OPISAMI
if os.path.exists(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Pomijamy nagłówki
        saved_coords = [
            [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            for row in reader
        ]

    # Nowy wykres z oznaczonymi punktami
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(img, cmap="gray")
    plt.title("Oznaczone punkty")

    # Rysowanie punktów i podpisów
    for coord in saved_coords:
        plt.plot(coord[0], coord[1], "ro", markersize=6)  # Punkty na czerwono
        plt.text(
            coord[0],
            coord[1],
            f"({coord[2]:.0f}, {coord[3]:.0f})",
            color="yellow",
            fontsize=10,
        )

    # Zapis widoku Matplotlib (z kropkami i współrzędnymi) do pliku JPG
    out_path = "wykryte punkty naroznikow.jpg"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)

    plt.show()
else:
    print(f"Nie znaleziono pliku CSV: {csv_file}")