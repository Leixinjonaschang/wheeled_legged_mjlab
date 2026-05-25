# Terrain Roughness Indicator and Roughness-Conditioned Reward Plan

## Context From Current Code

`Mjlab-Velocity-Rough-WF-Tron1B` already has the pieces needed for a
roughness-conditioned reward path:

- `terrain_scan` is a base-mounted grid raycast used by actor and critic
  observations. It is not used for the roughness scalar.
- `wheel_height_scan` is mounted on `wheel_L_Link` and `wheel_R_Link`. It uses
  concentric rings and `TerrainHeightSensorCfg` default `reduction="min"`, so
  it returns per-wheel clearance with shape `[B, 2]`.
- `wheels_ground_contact` reports wheel-ground contact and tracks air time.
- Wheel velocity actions use `JointVelocityActionCfg(scale=10.0)` on
  `wheel_[LR]_Joint`.
- Existing wheel speed regularization is `wheel_joint_vel` with weight
  `-0.002`; this remains as the baseline wheel velocity penalty.

Because roughness needs local scan variation rather than a min-reduced
clearance, first implementation adds a dedicated wheel-level grid sensor.

## Phase 1: Roughness Signal

Add `wheel_roughness_scan` only for rough terrain:

- Sensor type: `TerrainHeightSensorCfg`
- Frames: `wheel_L_Link`, `wheel_R_Link`
- Pattern: `GridPatternCfg(size=(0.40, 0.40), resolution=0.10)`
- Reduction: `none`, yielding `[B, 2, 25]`
- Ray alignment: `world`
- Wheel radius normalization: `R = 0.127 m`

For each wheel-foot scan, compute:

```text
s_i = std(H_i)
a_i = max(H_i) - min(H_i)
j_i = max neighboring grid height jump
r_i = 0.3 * s_i / R + 0.3 * a_i / R + 0.4 * j_i / R
r = max(r_left, r_right)
```

`TerrainHeightSensor` reports clearance (`frame_z - hit_z`) instead of raw
terrain height. This is acceptable for `std`, `range`, and neighboring jumps
because these quantities are invariant to the sign flip plus constant frame
height offset.

## Phase 2: Reward Gate

Use clipped linear gating for the first version:

```text
lambda(r) = clip((r - 0.25) / (0.85 - 0.25), 0, 1)
```

The wider initial range is chosen from a small current-code inspection pass:
ring-scan robot-level roughness had approximate quantiles around median `0.33`
and P75 `0.87`. The new grid sensor should be logged and recalibrated after
short rollouts.

Debug logs:

- `Metrics/roughness_left_mean`
- `Metrics/roughness_right_mean`
- `Metrics/roughness_max_mean`
- `Metrics/roughness_lambda_mean`
- `Metrics/roughness_std_over_R_mean`
- `Metrics/roughness_range_over_R_mean`
- `Metrics/roughness_jump_over_R_mean`
- `Metrics/rough_clearance_target_mean`

## Phase 3: Roughness-Conditioned Rewards

Register these terms only when `rough=True`.

### Wheel Usage Penalty

```text
raw = lambda(r) * (omega_L^2 + omega_R^2)
weight = -5.0e-4
```

At `omega_L = omega_R = 10 rad/s`, the raw value is about `200`; at full gate
this contributes about `-0.1` reward rate. This is deliberately smaller than
the existing `wheel_joint_vel` baseline so it discourages excessive wheel
rolling on rough terrain without disabling wheel-leg coordination.

### Foot Clearance Reward

Use the existing min-reduced `wheel_height_scan` for current clearance:

```text
c_i = wheel_height_scan_i
c_i* = clamp(0.06 + 0.5 * a_i, max=0.18)
raw = lambda(r) * sum(in_air_i * exp(-((c_i - c_i*)^2) / 0.04^2))
weight = 0.25
```

The reward is active only when:

- `wheels_ground_contact.data.found == 0` for that wheel-foot
- `norm(command_xy) + abs(command_yaw) > 0.05`

The maximum raw value is about `2`, so full-gate contribution is about `+0.5`
reward rate, below the main velocity tracking rewards.

## Phase 4: Validation And Tuning

Initial validation:

- Unit test synthetic 5x5 grids for flat and step roughness.
- Config test rough vs flat sensor and reward registration.
- CPU smoke test with two rough envs: reset, reward compute, finite checks,
  roughness sensor shape `[B, 2, 25]`.

Training rollout checks:

- `lambda` should stay near zero on flat/smooth cells and rise on obstacles,
  stairs, stones, and random rough patches.
- `rough_wheel_usage` should not dominate `track_linear_velocity`.
- `rough_foot_clearance` should increase swing clearance on rough cells without
  encouraging permanent high-foot posture.

Tuning rules:

- If `lambda` saturates near 1 too often, raise `gate_max`.
- If rough terrains still produce low `lambda`, lower `gate_min/gate_max`.
- If the robot slows down or avoids useful wheel-leg coupling, reduce
  `rough_wheel_usage` weight.
- If feet stay high too long, reduce clearance weight, reduce target clamp, or
  tighten swing/contact activation.

## Explicit Non-Goals

- Do not hard-code a wheel/leg mode switch.
- Do not condition the entire reward function.
- Do not add a roughness-conditioned wheel slip penalty in the first version.
- Do not include base `terrain_scan` in the roughness scalar.
