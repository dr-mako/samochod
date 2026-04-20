#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path

import cv2
import numpy as np

# program drukuje na A2 za pomocą polecenia (kratka 34 mm):
# python generate_charuco_sheet_svg.py --squares-x 11 --squares-y 8 --square-mm 34 --marker-mm 24 --margin-mm 10
# albo 35 mm ale 1 kratka mniej
# python generate_charuco_sheet_svg.py --squares-x 11 --squares-y 7 --square-mm 35 --marker-mm 25 --margin-mm 10
# duża plansza 840 na 420 - wpisz w konsoli:
# python .\generate_charuco_sheet_svg.py --dict DICT_6X6_250 --squares-x 27 --squares-y 13 --square-mm 30 --marker-mm 21 --margin-mm 15 --dpi 600 --out-png charuco_840x420.png --out-svg charuco_840x420.svg

# jeśli chcesz słownik dict DICT_4X4_50 - uruchamiaj
'''
python generate_charuco_sheet_svg.py \
  --dict DICT_4X4_50 \
  --squares-x 14 \
  --squares-y 7 \
  --square-mm 60 \
  --marker-mm 40 \
  --margin-mm 0 \
  --dpi 600 \
  --out-png charuco_840x420.png \
  --out-svg charuco_840x420.svg
'''

#python generate_charuco_sheet_svg.py --dict DICT_4X4_50 --squares-x 14 --squares-y 7 --square-mm 60 --marker-mm 40 --margin-mm 0 --dpi 600 --out-png charuco_840x420.png --out-svg charuco_840x420.svg

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def get_aruco_dict(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(
            f"Nieznany słownik: {dict_name}. "
            f"Przykład: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250."
        )
    dict_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_len: float,
    marker_len: float,
    aruco_dict,
):
    # OpenCV używa długości w "jednostkach świata" (np. metry),
    # ale do generowania obrazu istotny jest głównie stosunek marker/square.
    try:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len,
            marker_len,
            aruco_dict,
        )
    except TypeError:
        # kompatybilność ze starszymi wersjami OpenCV
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_len,
            marker_len,
            aruco_dict,
        )
    return board


def render_board_image(board, out_w_px: int, out_h_px: int, margin_px: int):
    out_size = (out_w_px, out_h_px)

    if hasattr(board, "generateImage"):
        img = board.generateImage(out_size, marginSize=margin_px, borderBits=1)
        return img

    if hasattr(board, "draw"):
        try:
            img = board.draw(out_size, marginSize=margin_px, borderBits=1)
        except TypeError:
            img = board.draw(out_size)
        return img

    # bardzo stary fallback:
    img = np.zeros((out_h_px, out_w_px), dtype=np.uint8)
    cv2.aruco.drawPlanarBoard(board, out_size, img, marginSize=margin_px, borderBits=1)
    return img


def build_svg_with_embedded_png(
    png_bytes: bytes,
    width_mm: float,
    height_mm: float,
) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")

    # SVG2: href, ale zostawiam też xlink dla kompatybilności.
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width_mm}mm" height="{height_mm}mm"
     viewBox="0 0 {width_mm} {height_mm}">
  <image x="0" y="0"
         width="{width_mm}" height="{height_mm}"
         xlink:href="data:image/png;base64,{b64}"
         href="data:image/png;base64,{b64}" />
</svg>
"""
    return svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", dest="dict_name", default="DICT_4X4_50")
    parser.add_argument("--squares-x", type=int, default=11)
    parser.add_argument("--squares-y", type=int, default=8)
    parser.add_argument("--square-mm", type=float, default=35.0)
    parser.add_argument("--marker-mm", type=float, default=25.0)
    parser.add_argument("--margin-mm", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out-png", type=Path, default=Path("charuco.png"))
    parser.add_argument("--out-svg", type=Path, default=Path("charuco.svg"))
    args = parser.parse_args()

    if args.marker_mm >= args.square_mm:
        raise ValueError("--marker-mm musi być < --square-mm (np. 25mm vs 35mm).")

    aruco_dict = get_aruco_dict(args.dict_name)

    # Stosunek ma znaczenie; możesz też użyć metrów (0.035 i 0.025).
    square_len = 1.0
    marker_len = args.marker_mm / args.square_mm

    board = make_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_len=square_len,
        marker_len=marker_len,
        aruco_dict=aruco_dict,
    )

    board_w_mm = args.squares_x * args.square_mm + 2.0 * args.margin_mm
    board_h_mm = args.squares_y * args.square_mm + 2.0 * args.margin_mm

    out_w_px = mm_to_px(board_w_mm, args.dpi)
    out_h_px = mm_to_px(board_h_mm, args.dpi)
    margin_px = mm_to_px(args.margin_mm, args.dpi)

    img = render_board_image(board, out_w_px, out_h_px, margin_px)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.out_png), img)
    if not ok:
        raise RuntimeError(f"Nie udało się zapisać PNG: {args.out_png}")

    png_bytes = args.out_png.read_bytes()
    svg = build_svg_with_embedded_png(png_bytes, board_w_mm, board_h_mm)

    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.write_text(svg, encoding="utf-8")

    print("Zrobione:")
    print(f"- PNG: {args.out_png.resolve()}")
    print(f"- SVG: {args.out_svg.resolve()}")
    print(f"Rozmiar fizyczny: {board_w_mm:.1f}mm x {board_h_mm:.1f}mm, dpi={args.dpi}")


if __name__ == "__main__":
    main()