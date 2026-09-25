# Training input migration

The training sources were synchronized from the working tree of
`/home/phi/Downloads/wheeled_legged_mjlab_mjlab160_depth`: commit `c628842`
plus its uncommitted dynamics randomization changes. Source files affected:
`env_cfgs.py`, `mdp/events.py`, and `mdp/observations.py`.
Tests and documentation were adapted locally rather than copied.

## Depth preprocessing

After the existing left crop, NaN, infinity, zero and negative depth values
become the maximum depth. Values are clipped to 0.2–2.0 meters and transformed
using `(depth - 0.2) / 1.8`, producing values in [0, 1]. Both synchronous and
asynchronous buffers use this preprocessing. Camera source data is preserved.

## Dynamics and observation settings

| Quantity | Migrated setting |
| --- | --- |
| Body mass and inertia | Density scaling 0.8–1.2 via pseudo-inertia |
| Leg Kp/Kd, wheel Kv | Independent scaling 0.8–1.2 from nominal gains |
| Base COM | ±0.03 m on each axis |
| Other link COM | ±0.015 m on each axis |
| Wheel friction | Shared 0.45–1.1 sample plus per-wheel ±0.04 |
| Encoder bias | ±0.015 rad |
| Reset base linear velocity | x/y ±0.3, z ±0.2 m/s |
| Reset base angular velocity | roll/pitch ±0.25, yaw ±0.2 rad/s |
| Reset leg position / velocity | ±0.3 rad / ±0.2 rad/s |
| Reset wheel velocity | ±0.5 rad/s |
| Push interval | 10–15 s |
| Push linear velocity | x ±0.5, y ±0.6, z ±0.2 m/s |
| Push angular velocity | roll/pitch ±0.35, yaw ±0.5 rad/s |
| Wheel velocity observation | Scale 0.05; noise ±0.2 rad/s before scaling |

Friction, density, COM, gains and encoder bias are sampled at startup.
Play still retains startup dynamics randomization. The custom friction
operation adds to the current shared sample. Reapplying PD randomization
scales nominal gains, avoiding accumulation. Inertia context compares sorted
principal moments because eigendecomposition can change their order.

## Privileged inputs and compatibility

`dynamics_context` expands from 13 to 87 values normalized to [-1, 1]:

| Slice | Quantity |
| --- | --- |
| 0:2 | Wheel friction |
| 2:10 | Encoder bias |
| 10:13 | Base COM |
| 13:37 | Other link COM |
| 37:46 | Body mass scales |
| 46:73 | Principal inertia scales |
| 73:79 | Leg Kp scales |
| 79:87 | Leg Kd and wheel Kv scales |

These extra quantities are privileged training inputs. Old representation
checkpoints with 13-dimensional context cannot be strictly resumed with the
expanded input dimensions. Start fresh training for the migrated configuration.
New deployment inputs must use wheel velocity scale 0.05 and normalized depth;
existing checkpoints retain their original preprocessing requirements.
Action scales remain 0.5 for leg position and 10.0 for wheel velocity.

## Validation

Use the declared MJLab 1.6.0 and Warp 1.14.0 dependencies; a pre-existing
MJLab 1.3.0 environment is incompatible with this project's command API.
Synchronize dependencies with `uv sync --group dev` before testing or training.

`tests/test_training_input_migration.py` covers invalid depth pixels, custom
depth ranges, preservation of camera data, and real CPU environment startup
and stepping. It checks mass/inertia consistency, COM context, friction
composition and repeated PD updates limited to selected environments.
Existing representation tests cover buffer timing, reset behavior, task
configuration and model inputs. CPU checks do not establish GPU training
convergence or sim-to-real performance.
