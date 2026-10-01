"""Render all training terrain types as one combined terrain in a single MuJoCo scene.

Every unique sub-terrain in ``TERRAINS_CFG`` is generated at difficulty=1.0 and
placed as one tile of a single grid (the same tiling the training generator uses,
tiles abut directly), optionally surrounded by a flat border. The whole scene is then
rendered once from an oblique camera. Unused grid cells are filled with flat
ground. All surfaces are a uniform matte grey so that relief is conveyed purely
by lighting and shadows, and the background is made transparent via a depth
mask. A labelled copy is written alongside the raw render.

Usage:
  MUJOCO_GL=egl uv run python scripts/vis/render_terrain_mosaic.py \
    [--azimuth 180 --elevation -32] [--light-azimuth 115 --light-elevation 45]
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mjlab.terrains import BoxFlatTerrainCfg, TerrainGenerator, TerrainGeneratorCfg
from PIL import Image, ImageDraw, ImageFont

from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.terrain_cfg import (
    TERRAINS_CFG,
)


def unique_sub_terrains() -> dict:
    subs = {}
    for name, sub_cfg in TERRAINS_CFG.sub_terrains.items():
        base = name.split("__")[0]
        if base not in subs:
            subs[base] = copy.deepcopy(sub_cfg)
    return subs


TERRAIN_RGBA = (0.66, 0.66, 0.66, 1.0)
BORDER_RGBA = (0.90, 0.90, 0.90, 1.0)
SEAM_RGBA = (0.56, 0.56, 0.56, 1.0)


def recolor(spec: mujoco.MjSpec, border_geoms: set[str]) -> None:
    """Give every terrain geom (heightfields included) the same matte grey."""
    matte = spec.add_material(
        name="matte", specular=0.05, shininess=0.1, reflectance=0.0
    )
    for geom in spec.body("terrain").geoms:
        geom.material = matte.name
        if geom.name in border_geoms:
            geom.rgba[:] = BORDER_RGBA
        elif geom.name.startswith("seam_"):
            geom.rgba[:] = SEAM_RGBA
        else:
            geom.rgba[:] = TERRAIN_RGBA


def build_model(
    subs: dict,
    rows: int,
    cols: int,
    seed: int,
    border: float,
    gap: float,
    base_bottom: float,
    key_dir: np.ndarray,
    side_dir: np.ndarray,
    fill_dir: np.ndarray,
):
    filler = {f"_fill{i}": BoxFlatTerrainCfg() for i in range(rows * cols - len(subs))}
    gen_cfg = TerrainGeneratorCfg(
        seed=seed,
        curriculum=False,
        size=TERRAINS_CFG.size,
        border_width=border,
        border_height=1.0,
        num_rows=rows,
        num_cols=cols,
        color_scheme="none",
        sub_terrains={**subs, **filler},
    )
    gen = TerrainGenerator(gen_cfg)

    spec = mujoco.MjSpec()
    spec.visual.quality.shadowsize = 8192
    spec.visual.global_.offwidth = 8192
    spec.visual.global_.offheight = 8192
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.24, 0.24, 0.24]
    spec.visual.headlight.diffuse = [0.06, 0.06, 0.06]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.worldbody.add_body(name="terrain")

    # Place each terrain type in its own cell at maximum difficulty. With
    # gap > 0 the cells are spread apart and each becomes a free-floating block.
    body = spec.body("terrain")
    sx, sy = TERRAINS_CFG.size
    pitch_x, pitch_y = sx + gap, sy + gap
    x0 = -(rows * pitch_x - gap) / 2
    y0 = -(cols * pitch_y - gap) / 2
    centers: dict[str, np.ndarray] = {}
    half = 0.5 * np.array([sx, sy, 0.0])
    for i, (name, sub_cfg) in enumerate(gen_cfg.sub_terrains.items()):
        row, col = divmod(i, cols)
        corner = np.array([x0 + row * pitch_x, y0 + col * pitch_y, 0.0])
        first_geom = len(body.geoms)
        gen._create_terrain_geom(spec, corner, 1.0, sub_cfg, row, col)
        if gap > 0:
            add_block_base(spec, body.geoms[first_geom:], corner, base_bottom)
        if not name.startswith("_fill"):
            centers[name] = corner + half
    if gap > 0:
        # Floating blocks: no shared border and no seams between tiles.
        return finish_model(spec, set(), key_dir, side_dir, fill_dir), centers

    # Thin seam lines on the tile boundaries so neighbouring tiles stay distinct.
    for r in range(1, rows):
        body.add_geom(
            name=f"seam_r{r}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, cols * sy / 2, 0.002],
            pos=[x0 + r * sx, 0.0, 0.0],
            contype=0,
            conaffinity=0,
        )
    for c in range(1, cols):
        body.add_geom(
            name=f"seam_c{c}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[rows * sx / 2, 0.03, 0.002],
            pos=[0.0, y0 + c * sy, 0.0],
            contype=0,
            conaffinity=0,
        )

    num_terrain_geoms = len(spec.body("terrain").geoms)
    gen._add_terrain_border(spec)
    border_geoms = spec.body("terrain").geoms[num_terrain_geoms:]
    for i, geom in enumerate(border_geoms):
        geom.name = f"border_{i}"
    return finish_model(
        spec, {g.name for g in border_geoms}, key_dir, side_dir, fill_dir
    ), centers


def add_block_base(spec, tile_geoms, corner: np.ndarray, bottom: float) -> None:
    """Close a tile into a solid block: a slab below its lowest surface plus thin
    skirt walls on its outer faces, so the block sides read as one solid face even
    where the terrain geometry does not reach the tile edge (e.g. pits)."""
    lowest = 0.0
    for geom in tile_geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            lowest = min(lowest, geom.pos[2] - geom.size[2])
        elif geom.type == mujoco.mjtGeom.mjGEOM_HFIELD:
            # Heightfield bases scale with the terrain amplitude (up to ~0.9 m),
            # which would make blocks differ in thickness or poke out below a
            # shared bottom; keep only a thin base and let the block close it.
            hfield = spec.hfield(geom.hfieldname)
            hfield.size[3] = 0.02
            lowest = min(lowest, geom.pos[2] - hfield.size[3])
    body = spec.body("terrain")
    sx, sy = TERRAINS_CFG.size
    cx, cy = corner[0] + sx / 2, corner[1] + sy / 2

    def add_box(half_size, pos):
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_size,
            pos=pos,
            contype=0,
            conaffinity=0,
        )

    slab_top = lowest + 1e-3  # tuck under the terrain to avoid z-fighting
    if slab_top > bottom:
        add_box(
            [sx / 2, sy / 2, (slab_top - bottom) / 2], [cx, cy, (slab_top + bottom) / 2]
        )
    # Skirt walls stop just below the tile rim (z=0) and sit just inside the edge
    # (inset so they do not z-fight with heightfield side walls).
    wall_top, t, inset = -2e-3, 0.02, 2e-3
    hz, zc = (wall_top - bottom) / 2, (wall_top + bottom) / 2
    for sign in (-1, 1):
        add_box([t, sy / 2, hz], [cx + sign * (sx / 2 - t - inset), cy, zc])
        add_box([sx / 2, t, hz], [cx, cy + sign * (sy / 2 - t - inset), zc])


def finish_model(spec, border_names: set[str], key_dir, side_dir, fill_dir):
    recolor(spec, border_names)

    # A strong, fairly low key light casts the shadows that carry the relief. A
    # side light from the other flank gives the remaining vertical faces their own
    # tone (otherwise faces lit by the key match the tops), and a weak back fill
    # keeps shaded faces from going black.
    spec.worldbody.add_light(
        pos=[0.0, 0.0, 30.0],
        dir=key_dir,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.66, 0.66, 0.66],
        specular=[0.03, 0.03, 0.03],
        castshadow=True,
    )
    spec.worldbody.add_light(
        pos=[0.0, 0.0, 30.0],
        dir=side_dir,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.24, 0.24, 0.24],
        specular=[0.0, 0.0, 0.0],
        castshadow=False,
    )
    spec.worldbody.add_light(
        pos=[0.0, 0.0, 30.0],
        dir=fill_dir,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.10, 0.10, 0.10],
        specular=[0.0, 0.0, 0.0],
        castshadow=False,
    )
    return spec.compile()


def render(model, width, height, cam, supersample: int = 2):
    """Render RGBA: colour from the normal pass, alpha from a depth pass.

    Both passes are rendered at ``supersample`` times the output resolution and
    box-filtered down with premultiplied alpha, which gives anti-aliased edges
    against the transparent background.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ss = supersample
    with mujoco.Renderer(model, height=height * ss, width=width * ss) as renderer:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().astype(np.float32) / 255.0
        # Background pixels sit on the far clipping plane in the depth pass.
        # (Segmentation rendering is unusable here: MSAA blends the segment ids.)
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        far = model.vis.map.zfar * model.stat.extent
        mask = (depth < 0.999 * far).astype(np.float32)
        gl_cam = renderer.scene.camera[0]
        view = {
            "pos": np.array(gl_cam.pos),
            "forward": np.array(gl_cam.forward),
            "up": np.array(gl_cam.up),
            "center": gl_cam.frustum_center,
            "bottom": gl_cam.frustum_bottom,
            "top": gl_cam.frustum_top,
            "near": gl_cam.frustum_near,
        }

    def down(x):
        return x.reshape(height, ss, width, ss, *x.shape[2:]).mean(axis=(1, 3))

    alpha = down(mask)
    color = down(rgb * mask[..., None]) / np.maximum(alpha, 1e-6)[..., None]
    rgba = np.concatenate([color, alpha[..., None]], axis=-1)
    return (np.clip(rgba, 0.0, 1.0) * 255).round().astype(np.uint8), view


def project(p: np.ndarray, view: dict, width: int, height: int) -> tuple[float, float]:
    """Project a world point to pixel coordinates of the rendered image."""
    right = np.cross(view["forward"], view["up"])
    d = p - view["pos"]
    z = d @ view["forward"]
    x = (d @ right) * view["near"] / z
    y = (d @ view["up"]) * view["near"] / z
    fh = view["top"] - view["bottom"]
    fw = fh * width / height
    u = (x - view["center"]) / fw + 0.5
    v = (view["top"] - y) / fh
    return u * width, v * height


def light_dir(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Direction of travel of a light shining along ``azimuth`` and descending at
    ``elevation`` degrees above the horizon (MuJoCo camera azimuth convention)."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("logs/terrain_maps"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--border", type=float, default=2.0)
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Spacing between tiles in metres. > 0 renders each terrain as a "
        "separate floating block (no shared border); output gets a _floating suffix.",
    )
    parser.add_argument(
        "--base-bottom",
        type=float,
        default=-1.6,
        help="Bottom z of the floating blocks in metres (only used with --gap).",
    )
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--elevation", type=float, default=-32.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument(
        "--light-azimuth",
        type=float,
        default=115.0,
        help="Key-light heading relative to the camera azimuth, in degrees.",
    )
    parser.add_argument(
        "--light-elevation",
        type=float,
        default=45.0,
        help="Key-light angle above the horizon, in degrees (lower = longer shadows).",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    subs = unique_sub_terrains()
    if len(subs) > args.rows * args.cols:
        raise ValueError(
            f"{len(subs)} terrains do not fit a {args.rows}x{args.cols} grid"
        )
    # Lights are defined relative to the camera so the shading stays readable for
    # any --azimuth.
    key_az = args.azimuth + args.light_azimuth
    model, centers = build_model(
        subs,
        args.rows,
        args.cols,
        args.seed,
        args.border,
        args.gap,
        args.base_bottom,
        key_dir=light_dir(key_az, args.light_elevation),
        side_dir=light_dir(key_az - 100.0, 30.0),
        fill_dir=light_dir(key_az + 180.0, 55.0),
    )

    extent = max(
        args.rows * (TERRAINS_CFG.size[0] + args.gap),
        args.cols * (TERRAINS_CFG.size[1] + args.gap),
    )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Shift the focus toward the camera so the near edge is not clipped.
    cam.lookat[:] = (0.05 * extent, 0.0, 0.0)
    cam.distance = args.distance or 1.2 * extent
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    img, view = render(model, args.width, args.height, cam, args.supersample)

    stem = "all_terrains_floating" if args.gap > 0 else "all_terrains"
    raw_path = args.out / f"{stem}.png"
    Image.fromarray(img).save(raw_path)

    labelled = Image.fromarray(img)
    draw = ImageDraw.Draw(labelled)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", args.height // 55)
    except OSError:
        font = ImageFont.load_default()
    for name, center in centers.items():
        u, v = project(center, view, args.width, args.height)
        draw.text(
            (u, v),
            name,
            font=font,
            fill=(40, 40, 40, 255),
            anchor="mm",
            stroke_width=max(2, args.height // 400),
            stroke_fill=(255, 255, 255, 230),
        )
    labelled_path = args.out / f"{stem}_labeled.png"
    labelled.save(labelled_path)
    print(
        f"[render_terrain_mosaic] {len(centers)} terrains -> {raw_path}, {labelled_path}"
    )


if __name__ == "__main__":
    main()
