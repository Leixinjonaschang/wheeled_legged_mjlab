# NaN Debug: Box Terrain Training

## Problem

`Mjlab-Velocity-Rough-WF-Tron1B` could fail during long training with box obstacle terrain enabled. The first reproduced failure was raised by the actor observation group, for example:

```text
ValueError: NaN/Inf detected in observation 'actor/base_ang_vel'
```

The target debug runs used 2048 environments, NaN guard, and per-term actor/critic observation checking:

```bash
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B \
  --agent.max-iterations 3500 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name nonfinite_fix_3500 \
  --enable-nan-guard True \
  --nan-obs-policy error
```

## Cause Identified

### Raw Physics NaN

The actor observation was not the root cause. It was the first visible failure point after one environment's raw physics state had already become non-finite.

Debug evidence showed NaN values in the underlying MuJoCo/Warp tensors before the actor/critic observation failure:

- `qpos`
- `qvel`
- `qacc`
- `qacc_warmstart`
- `sensordata`

The first reproduced bad environment was on `random_rough` terrain, not directly on a box sub-terrain. The current conclusion is that box terrain training exposed an intermittent single-environment physics instability; the corrupted physics state then propagated into observation terms such as `base_ang_vel`.

### Contact Sensor NaN

A later long run failed around iteration 7379 with a critic-only observation NaN:

```text
ValueError: The observation group 'critic' returned by the environment contains NaN values.
```

The runner debug dump identified the direct failing term as `critic/wheel_contact_forces` for a single environment. In that failure:

- `actor` observations were finite.
- `critic.wheel_contact_forces` contained NaN values.
- `qpos` and `qvel` were still finite.
- `qacc` contained NaN values.
- `sensordata` contained NaN values.
- The bad environment was on a `flat` terrain type at terrain level 8.

`wheel_contact_forces` is produced from `ContactSensor.data.force` through `foot_contact_forces()`. This makes the contact sensor force path the direct NaN outlet. This is related to the risk area discussed in `https://github.com/mujocolab/mjlab/issues/931`: contact sensor force/history can have unstable behavior in short contact events, although that issue describes force/history inconsistency rather than this exact NaN failure.

## Resolution

The fix is to detect non-finite physics and sensor state before it reaches the observation and policy update path.

Changes made:

- Added `--nan-obs-policy` support in `scripts/rsl_rl/train.py`.
- When NaN guard is enabled, actor and critic observation groups default to per-term `error` checking.
- Added runner-side NaN debug JSON dumps in `rsl_rl/rsl_rl/runners/on_policy_runner.py`.
- Added `non_finite_physics` termination in `src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py`.
- Registered `non_finite_physics` in the WF Tron1B velocity termination config.

`non_finite_physics` checks each environment and resets only the affected environments when any checked state becomes non-finite.

The checked state now includes:

- Raw MuJoCo/Warp state: `qpos`, `qvel`, `qacc`, `qacc_warmstart`, `sensordata`.
- Actuator/force outputs: `actuator_force`, `qfrc_actuator`.
- `ContactSensor` cached fields: `found`, `force`, `torque`, `dist`, `pos`, `normal`, `tangent`, air/contact time fields, and force/torque/dist history buffers.

`ctrl` is intentionally not part of this termination. If `ctrl` becomes non-finite, that points to the policy/action pipeline and should be treated as an algorithm-side error rather than silently reset.

## Verification

The fixed run completed 3500 training iterations with 2048 environments and the current terrain configuration preserved.

Final artifact:

```text
logs/rsl_rl/wf_tron1b_velocity/2026-05-21_01-57-39_nonfinite_fix_3500/model_3499.pt
```

During the fixed run, NaN guard still captured one non-finite physics event, but training did not crash. The new termination caught the affected environment, reset it, and training continued to completion.

## Operational Notes

- Keep `--enable-nan-guard True --nan-obs-policy error` for long debug runs.
- If a future run fails before termination can reset the environment, inspect `nan_guard/*.npz` and `nan_debug/*.json` in the run directory.
- Treat actor/critic observation NaNs as symptoms until raw physics tensors have been checked.
- For critic-only NaNs, check privileged observation terms first, especially `wheel_contact_forces` and other terms derived from `ContactSensor`.
