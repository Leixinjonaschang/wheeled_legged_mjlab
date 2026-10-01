"""Monte Carlo of mjlab's BoxRandomSpreadTerrainCfg box placement (same sampling as the
generator loop, yawed footprints) to choose the random_spread box count of the course section.

Raised-area fraction (any box footprint) on the robots' path band of the course section
(x in [0.2, 3.4] m of the section, |y - 1.8| <= 0.46 m: lane centre +- (0.3 m spawn + 0.156 m
wheel)) versus a training patch (8 m, border 0.5, platform 2.0, 64 boxes) outside its
platform (0.2 m inside the border, |x|, |y| >= 1.3 m from the centre, as terrain_audit.py),
per difficulty; also the centre pocket (|x - 1.8| <= 0.2 m) against the rest of the band.
Run: uv run --no-project --python 3.13 --with numpy python scripts/eval/mode_switch/tools/random_spread_density.py
"""

import numpy as np

W, L, H, YAW = (0.2, 0.7), (0.2, 0.9), (0.03, 0.25), (-45.0, 45.0)  # Training ranges.


def boxes(rng, n_base, d, size, border, platform):
    """(x, y, side_x, side_y, yaw) of the boxes the generator keeps."""
    kept, lo, hi = [], size / 2 - platform / 2, size / 2 + platform / 2
    for _ in range(int(n_base * (0.5 + 0.5 * d))):
        sx, sy = rng.uniform(*W), rng.uniform(*L)
        rng.uniform(*H)
        px = rng.uniform(border + sx / 2, size - border - sx / 2)
        py = rng.uniform(border + sy / 2, size - border - sy / 2)
        if lo - sx / 2 <= px <= hi + sx / 2 and lo - sy / 2 <= py <= hi + sy / 2:
            continue  # Centre skip.
        kept.append((px, py, sx, sy, np.deg2rad(rng.uniform(*YAW))))
    return kept


def covered(kept, x, y):
    out = np.zeros(x.shape, bool)
    for px, py, sx, sy, yaw in kept:
        u = np.cos(yaw) * (x - px) + np.sin(yaw) * (y - py)
        v = -np.sin(yaw) * (x - px) + np.cos(yaw) * (y - py)
        out |= (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2)
    return out


def main(draws=1500):
    rng = np.random.default_rng(20260928)
    g = np.arange(0.7, 7.3, 0.04)
    tx, ty = np.meshgrid(g, g, indexing="ij")
    outside = ~((np.abs(tx - 4) < 1.3) & (np.abs(ty - 4) < 1.3))
    xs, ys = np.arange(0.2, 3.4, 0.02), np.arange(1.34, 2.261, 0.02)
    cx, cy = np.meshgrid(xs, ys, indexing="ij")
    pocket = np.abs(xs - 1.8) <= 0.2
    for d in (0.25, 0.5, 0.75, 1.0):
        train = np.mean([covered(boxes(rng, 64, d, 8.0, 0.5, 2.0), tx, ty)[outside].mean() for _ in range(draws // 4)])
        line = [f"d={d:g}: training {train:.3f}"]
        for n, platform in ((17, 0.01), (17, -1.0), (15, -1.0), (14, -1.0), (13, -1.0)):
            cov = np.array([covered(boxes(rng, n, d, 3.6, 0.0, platform), cx, cy) for _ in range(draws // 3)])
            prof = cov.mean(axis=(0, 2))
            line.append(f"N={n}{' skip' if platform > 0 else ''} {cov.mean():.3f} ({cov.mean() / train:.2f}x, "
                        f"pocket/rest {prof[pocket].mean() / prof[~pocket].mean():.2f})")
        print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
