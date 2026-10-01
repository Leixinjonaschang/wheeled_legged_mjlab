"""Render each training terrain type on its own, then compose them into one figure.

Every unique sub-terrain in ``TERRAINS_CFG`` is generated alone at difficulty=1.0
in the matte-grey, light-shaded style of render_terrain_mosaic.py (no border) and
rendered with its own camera centred on it and a transparent background. All
tiles are cropped with one shared box (so they keep the same scale) and composed
into rows on a transparent canvas, shorter rows centred (default 4 / 4 / 3).
Individual tiles are also saved. Each tile is closed into a solid block with a
shared bottom so all blocks look equally thick (``--no-block`` disables this).

Usage:
  MUJOCO_GL=egl uv run python scripts/vis/render_terrain_tiles.py \
    [--azimuth 120 --elevation -30] [--row-sizes 4 4 3] [--base-bottom -1.2]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_terrain_mosaic import (
    TERRAINS_CFG,
    build_model,
    light_dir,
    render,
    unique_sub_terrains,
)


def render_tile(name, sub_cfg, args) -> np.ndarray:
    key_az = args.azimuth + args.light_azimuth
    # A 1x1 grid without border; gap > 0 additionally closes it into a block.
    model, _ = build_model(
        {name: sub_cfg},
        1,
        1,
        args.seed,
        0.0,
        1.0 if args.block else 0.0,
        args.base_bottom,
        key_dir=light_dir(key_az, args.light_elevation),
        side_dir=light_dir(key_az - 100.0, 30.0),
        fill_dir=light_dir(key_az + 180.0, 55.0),
    )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, args.base_bottom / 2 if args.block else 0.0)
    cam.distance = args.distance or 2.1 * max(TERRAINS_CFG.size)
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    img, _ = render(model, args.tile_width, args.tile_height, cam, args.supersample)
    return img


def compose(
    crops: dict[str, Image.Image],
    row_sizes: list[int],
    spacing: float,
    row_spacing: float,
    font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    """Lay tiles out in rows (shorter rows centred) on a transparent canvas.

    With ``font`` each tile gets its name underneath; cells are widened so that
    the longest name fits.
    """
    tw, th = next(iter(crops.values())).size
    cell_w = tw
    label_h = 0
    if font is not None:
        cell_w = max(tw, int(max(font.getlength(n) for n in crops) * 1.08))
        label_h = int(font.size * 1.7)
    gap_x, gap_y = int(spacing * tw), int(row_spacing * th)
    ncols = max(row_sizes)
    width = ncols * cell_w + (ncols - 1) * gap_x
    height = len(row_sizes) * (th + label_h) + (len(row_sizes) - 1) * gap_y
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    names = iter(crops)
    for r, n_in_row in enumerate(row_sizes):
        x0 = (width - (n_in_row * cell_w + (n_in_row - 1) * gap_x)) // 2
        y = r * (th + label_h + gap_y)
        for c in range(n_in_row):
            name = next(names)
            x = x0 + c * (cell_w + gap_x)
            canvas.alpha_composite(crops[name], (x + (cell_w - tw) // 2, y))
            if font is not None:
                draw.text(
                    (x + cell_w // 2, y + th + label_h // 2),
                    name,
                    font=font,
                    fill=(40, 40, 40, 255),
                    anchor="mm",
                )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("logs/terrain_maps"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-sizes", type=int, nargs="+", default=[4, 4, 3])
    parser.add_argument("--tile-width", type=int, default=1600)
    parser.add_argument("--tile-height", type=int, default=1200)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--azimuth", type=float, default=120.0)
    parser.add_argument("--elevation", type=float, default=-30.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--light-azimuth", type=float, default=115.0)
    parser.add_argument("--light-elevation", type=float, default=45.0)
    parser.add_argument(
        "--no-block",
        dest="block",
        action="store_false",
        help="Render the raw terrain geometry instead of equal-thickness blocks.",
    )
    parser.add_argument(
        "--base-bottom",
        type=float,
        default=-1.2,
        help="Shared block bottom z in metres; must stay below the deepest pit (~-1.08).",
    )
    parser.add_argument(
        "--row-spacing",
        type=float,
        default=0.25,
        help="Gap between rows, as a fraction of tile height.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.10,
        help="Gap between tiles, as a fraction of tile width.",
    )
    args = parser.parse_args()

    subs = unique_sub_terrains()
    if sum(args.row_sizes) != len(subs):
        raise ValueError(f"--row-sizes must sum to {len(subs)} (got {args.row_sizes})")
    tile_dir = args.out / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    tiles: dict[str, np.ndarray] = {}
    for name, sub_cfg in subs.items():
        tiles[name] = render_tile(name, sub_cfg, args)
        print(f"[render_terrain_tiles] rendered {name}")

    # One shared crop box keeps every block at the same scale and alignment.
    alpha = np.max([t[..., 3] for t in tiles.values()], axis=0)
    ys, xs = np.nonzero(alpha)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    crops = {n: Image.fromarray(t).crop(box) for n, t in tiles.items()}
    for n, im in crops.items():
        im.save(tile_dir / f"{n}.png")

    th = box[3] - box[1]
    font_size = th // 13
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    sheet = compose(crops, args.row_sizes, args.spacing, args.row_spacing)
    labelled = compose(crops, args.row_sizes, args.spacing, args.row_spacing, font)
    names = list(crops)

    sheet.save(args.out / "all_terrains_tiles.png")
    labelled.save(args.out / "all_terrains_tiles_labeled.png")
    print(
        f"[render_terrain_tiles] {len(names)} tiles -> {args.out / 'all_terrains_tiles.png'}"
        f" ({sheet.size[0]}x{sheet.size[1]}), single tiles in {tile_dir}"
    )


if __name__ == "__main__":
    main()
