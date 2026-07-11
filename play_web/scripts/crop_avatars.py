#!/usr/bin/env python3
"""Crop avatar sprites from the sheet using per-cell content bbox + padding."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "client/assets/avatars/spritesheet.jpg"
OUT_DIR = ROOT / "client/assets/avatars"

COLS = 5
ROWS = 2
PADDING_PX = 10
# Treat near-white JPEG background as empty.
BG_THRESHOLD = 250


def content_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    px = img.load()
    w, h = img.size
    minx, miny, maxx, maxy = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a <= 10:
                continue
            if r >= BG_THRESHOLD and g >= BG_THRESHOLD and b >= BG_THRESHOLD:
                continue
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
    if maxx < 0:
        return None
    return minx, miny, maxx + 1, maxy + 1


def crop_with_padding(cell: Image.Image, padding: int) -> Image.Image:
    bbox = content_bbox(cell)
    if bbox is None:
        return cell
    minx, miny, maxx, maxy = bbox
    cw, ch = cell.size
    minx = max(0, minx - padding)
    miny = max(0, miny - padding)
    maxx = min(cw, maxx + padding)
    maxy = min(ch, maxy + padding)
    return cell.crop((minx, miny, maxx, maxy))


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing spritesheet: {SRC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sheet = Image.open(SRC).convert("RGBA")
    w, h = sheet.size
    cell_w, cell_h = w // COLS, h // ROWS

    idx = 0
    for row in range(ROWS):
        for col in range(COLS):
            cell = sheet.crop(
                (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
            )
            cropped = crop_with_padding(cell, PADDING_PX)
            out_path = OUT_DIR / f"avatar-{idx}.png"
            cropped.save(out_path)
            print(f"{out_path.name}: {cropped.size[0]}x{cropped.size[1]}")
            idx += 1


if __name__ == "__main__":
    main()
