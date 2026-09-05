# WF-TRON1B Deployment Scale Notes

Current branch: `feature/action_delay`

## Recent Scale-Relevant Changes

- `7917be6` changed proprioceptive joint observations from `ALL_JOINT_NAMES` to
  `LEG_JOINT_NAMES`. Wheel joint positions are no longer part of actor
  proprioception.
- `9c825a7` split wheel velocity into its own observation term:
  `wheel_vel` now uses `WHEEL_JOINT_NAMES`.
- `wheel_vel` scale changed `0.5 -> 0.05` (actor, critic, and privileged
  encoder groups all use `0.05`). Deployment must apply the new value.
- Leg joint velocity observation remains `scale=0.05`.
- Action scale did not change in these recent commits:
  - leg position action: `scale=0.5`
  - wheel velocity action: `scale=10.0`
- The current uncommitted diff only changes reward weight
  `wheel_air_time_balance: -1.0 -> -4.0`; it does not change deployment scale.

## Actor Observation Layout

Deployment should build `actor_obs` in this exact term order:

| Term | Dim | Source | Scale |
| --- | ---: | --- | ---: |
| `base_ang_vel` | 3 | `robot/gyro` | 1.0 |
| `projected_gravity` | 3 | body-frame gravity projection | 1.0 |
| `joint_pos` | 6 | leg joint positions relative to default | 1.0 |
| `joint_vel` | 6 | leg joint velocities | 0.05 |
| `wheel_vel` | 2 | wheel joint velocities | 0.05 |
| `actions` | 8 | previous policy action, raw normalized action | 1.0 |
| `command` | 3 | body-frame command `[vx_b, vy_b, yaw_rate]` | 1.0 |

Single-frame actor observation dimension: `31`.

The representation student ONNX path uses two inputs:

- `actor_obs`: current 31-dim actor observation.
- `proprio_obs`: flattened actor history, `5 * 31 = 155` dims, using the same
  term order for each frame.

## Joint And Action Order

Leg joints are selected by these patterns:

- `abad_[LR]_Joint`
- `hip_[LR]_Joint`
- `knee_[LR]_Joint`

The XML actuator order for leg position control is:

1. `abad_L_Joint`
2. `hip_L_Joint`
3. `knee_L_Joint`
4. `abad_R_Joint`
5. `hip_R_Joint`
6. `knee_R_Joint`

Wheel joints are:

1. `wheel_L_Joint`
2. `wheel_R_Joint`

Policy action order follows action terms:

1. `leg_pos`: 6 dims, normalized action multiplied by `0.5`, then default joint
   position offset is added.
2. `wheel_vel`: 2 dims, normalized action multiplied by `10.0`, no default
   offset.

With the current runner wrapper, exported ONNX metadata includes:

- `observation_names`
- `actor_observation_names`
- `proprio_observation_names`
- `proprio_history_length`
- `action_names`
- `action_target_names`
- `action_scale`
- `default_joint_pos`

Prefer reading these fields from ONNX metadata in deployment instead of
hard-coding them.

## Command Frame

`UniformVelocityCommand.command` exposes body-frame command observations to the
policy. The world-frame command is used internally for reward and termination
logic, but deployment actor input should use:

```text
[vx_b, vy_b, yaw_rate]
```

where world linear velocity command is rotated into robot yaw frame before being
fed to the policy.

