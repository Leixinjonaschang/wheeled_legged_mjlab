"""Compose per-terrain images (from render_terrains.py) into one figure.

Each terrain is rendered with its own camera and a transparent background. The
tiles are cropped with one shared box (union of their alpha masks, so they keep
the same scale) and laid out in rows on a transparent canvas, shorter rows
centred, e.g. 4 / 4 / 3.

Usage:
  uv run python scripts/vis/compose_terrain_grid.py [--row-sizes 6 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = (
    "flat",
    "discrete_obstacles",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "pyramid_stair_inv",
    "pyramid_stair",
    "random_stairs",
    "random_spread",
    "stepping_stones",
    "tilted_grid",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("logs/terrain_maps/single"))
    parser.add_argument("--out", type=Path, default=Path("logs/terrain_maps"))
    parser.add_argument("--row-sizes", type=int, nargs="+", default=[4, 4, 3])
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.03,
        help="Gap between tiles, fraction of tile width.",
    )
    parser.add_argument(
        "--margin", type=int, default=20, help="Padding around the crop, px."
    )
    args = parser.parse_args()
    if sum(args.row_sizes) != len(NAMES):
        raise ValueError(f"--row-sizes must sum to {len(NAMES)} (got {args.row_sizes})")

    imgs = {n: Image.open(args.dir / f"{n}.png").convert("RGBA") for n in NAMES}

    # Union of the alpha masks gives one crop box shared by all tiles.
    mask = np.zeros(np.asarray(next(iter(imgs.values()))).shape[:2], dtype=bool)
    for im in imgs.values():
        mask |= np.asarray(im)[..., 3] > 0
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    m = args.margin
    box = (
        max(xs.min() - m, 0),
        max(ys.min() - m, 0),
        min(xs.max() + 1 + m, w),
        min(ys.max() + 1 + m, h),
    )
    crops = {n: im.crop(box) for n, im in imgs.items()}
    tw, th = box[2] - box[0], box[3] - box[1]

    gap = int(args.spacing * tw)
    font_size = th // 11
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label_h = int(font_size * 1.6)

    ncols = max(args.row_sizes)
    width = ncols * tw + (ncols - 1) * gap
    nrows = len(args.row_sizes)
    sheet = Image.new("RGBA", (width, nrows * th + (nrows - 1) * gap), (0, 0, 0, 0))
    labelled = Image.new(
        "RGBA", (width, nrows * (th + label_h) + (nrows - 1) * gap), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(labelled)

    i = 0
    for r, n_in_row in enumerate(args.row_sizes):
        x0 = (
            width - (n_in_row * tw + (n_in_row - 1) * gap)
        ) // 2  # centre shorter rows
        for c in range(n_in_row):
            name = NAMES[i]
            x = x0 + c * (tw + gap)
            sheet.alpha_composite(crops[name], (x, r * (th + gap)))
            y = r * (th + label_h + gap)
            labelled.alpha_composite(crops[name], (x, y))
            draw.text(
                (x + tw // 2, y + th + label_h // 2),
                name,
                font=font,
                fill=(30, 30, 30, 255),
                anchor="mm",
            )
            i += 1

    sheet.save(args.out / "all_terrains_grid.png")
    labelled.save(args.out / "all_terrains_grid_labeled.png")
    print(
        f"[compose_terrain_grid] -> {args.out / 'all_terrains_grid.png'} {sheet.size}"
    )


if __name__ == "__main__":
    main()
