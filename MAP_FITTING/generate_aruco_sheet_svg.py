from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    out_prefix: str = "aruco_sheet_A4"

    page_w_mm: float = 210.0
    page_h_mm: float = 297.0

    margin_mm: float = 10.0
    gap_mm: float = 30.0

    marker_size_mm: float = 60.0
    label_font_size_mm: float = 6.0

    aruco_dict_name: str = "DICT_4X4_50"
    ids: tuple[int, ...] = tuple(range(0, 20))

    marker_px: int = 200


def get_aruco_dict(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown aruco dict: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def svg_header(w_mm: float, h_mm: float) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_mm}mm" height="{h_mm}mm" '
        f'viewBox="0 0 {w_mm} {h_mm}">\n'
    )


def svg_footer() -> str:
    return "</svg>\n"


def svg_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str | None = None,
    stroke_w: float | None = None,
) -> str:
    s = f'<rect x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" fill="{fill}"'
    if stroke is not None and stroke_w is not None:
        s += f' stroke="{stroke}" stroke-width="{stroke_w:.4f}"'
    s += "/>\n"
    return s


def svg_text(x: float, y: float, text: str, font_size_mm: float) -> str:
    return (
        f'<text x="{x:.4f}" y="{y:.4f}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{font_size_mm:.4f}mm" fill="black">'
        f"{text}</text>\n"
    )


def marker_to_bool_grid(marker_img_u8: np.ndarray) -> np.ndarray:
    return marker_img_u8 < 128


def vectorize_marker_as_rects(
    grid_black: np.ndarray, x0_mm: float, y0_mm: float, size_mm: float
) -> str:
    n = grid_black.shape[0]
    cell = size_mm / float(n)

    out = ""
    out += svg_rect(x0_mm, y0_mm, size_mm, size_mm, fill="white", stroke="black", stroke_w=0.3)

    ys, xs = np.nonzero(grid_black)
    for y, x in zip(ys.tolist(), xs.tolist()):
        out += svg_rect(x0_mm + x * cell, y0_mm + y * cell, cell, cell, fill="black")
    return out


def compute_layout(cfg: Config) -> tuple[int, int, int]:
    usable_w = cfg.page_w_mm - 2 * cfg.margin_mm
    usable_h = cfg.page_h_mm - 2 * cfg.margin_mm

    block_w = cfg.marker_size_mm
    block_h = cfg.marker_size_mm + cfg.label_font_size_mm * 2.0

    step_x = block_w + cfg.gap_mm
    step_y = block_h + cfg.gap_mm

    cols = int(np.floor((usable_w + cfg.gap_mm) / step_x))
    rows = int(np.floor((usable_h + cfg.gap_mm) / step_y))
    capacity = cols * rows
    return cols, rows, capacity


def main():
    cfg = Config()
    aruco_dict = get_aruco_dict(cfg.aruco_dict_name)

    cols, rows, capacity = compute_layout(cfg)
    if cols <= 0 or rows <= 0 or capacity <= 0:
        raise ValueError("Markers do not fit on the page with current settings.")

    ids = list(cfg.ids)
    pages = int(np.ceil(len(ids) / capacity))

    print(f"Layout: cols={cols}, rows={rows}, capacity/page={capacity}, pages={pages}")

    for p in range(pages):
        start = p * capacity
        end = min((p + 1) * capacity, len(ids))
        ids_page = ids[start:end]

        svg = ""
        svg += svg_header(cfg.page_w_mm, cfg.page_h_mm)
        svg += svg_rect(0, 0, cfg.page_w_mm, cfg.page_h_mm, fill="white")

        for k, marker_id in enumerate(ids_page):
            r = k // cols
            c = k % cols

            x_mm = cfg.margin_mm + c * (cfg.marker_size_mm + cfg.gap_mm)
            y_mm = cfg.margin_mm + r * (
                cfg.marker_size_mm + cfg.label_font_size_mm * 2.0 + cfg.gap_mm
            )

            marker_img = np.zeros((cfg.marker_px, cfg.marker_px), dtype=np.uint8)
            cv2.aruco.generateImageMarker(
                aruco_dict, int(marker_id), cfg.marker_px, marker_img, 1
            )
            grid_black = marker_to_bool_grid(marker_img)

            svg += vectorize_marker_as_rects(
                grid_black, x_mm, y_mm, cfg.marker_size_mm
            )

            label_y = y_mm + cfg.marker_size_mm + cfg.label_font_size_mm * 1.6
            #svg += svg_text(x_mm, label_y, f"ID {int(marker_id)}", cfg.label_font_size_mm)

        svg += svg_footer()

        out_path = Path(f"{cfg.out_prefix}_p{p+1:02d}.svg")
        out_path.write_text(svg, encoding="utf-8")
        print(f"Saved: {out_path} | markers={len(ids_page)}")


if __name__ == "__main__":
    main()