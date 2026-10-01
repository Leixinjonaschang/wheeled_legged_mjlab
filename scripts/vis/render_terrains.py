"""Render every training terrain type at maximum difficulty with MuJoCo.

Each sub-terrain in ``TERRAINS_CFG`` is generated alone (difficulty=1.0, i.e.
the hardest curriculum row) and rendered offscreen, each with its own camera
centred on it, from an oblique and a top-down view. Images have a transparent
background (alpha from a depth pass) so they can be composed into one figure
with compose_terrain_grid.py. Duplicate flat columns (``flat__0`` ...
``flat__9``) are rendered once.

Usage:
  MUJOCO_GL=egl uv run python scripts/vis/render_terrains.py
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mjlab.terrains import TerrainGenerator, TerrainGeneratorCfg
from PIL import Image

from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.terrain_cfg import (
    TERRAINS_CFG,
)


def build_model(name: str, sub_cfg, seed: int, border: float) -> mujoco.MjModel:
    gen_cfg = TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=TERRAINS_CFG.size,
        border_width=border,
        border_height=1.0,
        num_rows=1,
        num_cols=1,
        color_scheme="height",
        sub_terrains={name: copy.deepcopy(sub_cfg)},
        difficulty_range=(1.0, 1.0),
    )
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 8192
    spec.visual.global_.offheight = 8192
    spec.visual.quality.offsamples = 8
    spec.visual.quality.shadowsize = 8192
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[1.0, 1.0, 1.0],
        rgb2=[0.85, 0.88, 0.92],
        width=512,
        height=3072,
    )
    TerrainGenerator(gen_cfg).compile(spec)
    spec.worldbody.add_light(
        pos=[3.0, -4.0, 8.0],
        dir=[-0.3, 0.4, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.6, 0.6, 0.6],
        castshadow=True,
    )
    return spec.compile()


def render(
    model, width, height, lookat, distance, azimuth, elevation, ss: int = 2
) -> np.ndarray:
    """Render an RGBA image with a transparent background.

    Colour and depth are rendered at ``ss`` times the output resolution; pixels
    on the far clipping plane are background. Downsampling with premultiplied
    alpha gives anti-aliased edges without sky-coloured fringes.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    with mujoco.Renderer(model, height=height * ss, width=width * ss) as renderer:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().astype(np.float32) / 255.0
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render()
        renderer.disable_depth_rendering()
    far = model.vis.map.zfar * model.stat.extent
    mask = (depth < 0.999 * far).astype(np.float32)

    def down(x):
        return x.reshape(height, ss, width, ss, *x.shape[2:]).mean(axis=(1, 3))

    alpha = down(mask)
    color = down(rgb * mask[..., None]) / np.maximum(alpha, 1e-6)[..., None]
    rgba = np.concatenate([color, alpha[..., None]], axis=-1)
    return (np.clip(rgba, 0.0, 1.0) * 255).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("logs/terrain_maps/single"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--border", type=float, default=1.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-40.0)
    parser.add_argument("--supersample", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    tiles: list[tuple[str, np.ndarray]] = []
    size = max(TERRAINS_CFG.size)
    for name, sub_cfg in TERRAINS_CFG.sub_terrains.items():
        base = name.split("__")[0]
        if base in seen:
            continue
        seen.add(base)

        model = build_model(name, sub_cfg, args.seed, args.border)
        center = np.zeros(3)
        ss = args.supersample
        persp = render(
            model,
            args.width,
            args.height,
            center,
            2.1 * size,
            args.azimuth,
            args.elevation,
            ss,
        )
        top = render(
            model, args.height, args.height, center, 1.75 * size, 90.0, -90.0, ss
        )
        Image.fromarray(persp).save(args.out / f"{base}.png")
        Image.fromarray(top).save(args.out / f"{base}_top.png")
        tiles.append((base, persp))
        print(f"[render_terrains] {base}: difficulty=1.0 -> {args.out / base}.png")

    # Overview contact sheet.
    from PIL import ImageDraw, ImageFont

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tw, th = args.width // 2, args.height // 2
    sheet = Image.new("RGBA", (cols * tw, rows * th), (0, 0, 0, 0))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for i, (base, img) in enumerate(tiles):
        tile = Image.fromarray(img).resize((tw, th), Image.LANCZOS)
        ImageDraw.Draw(tile).text((16, 12), base, fill="black", font=font)
        sheet.alpha_composite(tile, ((i % cols) * tw, (i // cols) * th))
    sheet.save(args.out / "overview.png")
    print(f"[render_terrains] overview -> {args.out / 'overview.png'}")


if __name__ == "__main__":
    main()
