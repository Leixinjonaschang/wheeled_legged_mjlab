"""Record gait / stability / energy signals under the frozen evaluation protocol.

Re-simulates one checkpoint on one terrain class (same seeds, terrain and commands as the
main evaluation) and stores, per policy step:
  roughness gate lambda and gate flag exactly as in the roughness-conditioned reward
  (terrain_scan, gate_min 0, gate_max 0.5, 11 x 11 grid, threshold 0.65),
  wheel ground contact, wheel-bottom clearance and the reward's clearance target
  (0.03 m + 0.5 x local height range, capped at 0.18 m),
  mechanical joint power sum |tau qd| (legs, wheels), horizontal base speed,
  projected gravity and base angular velocity (body frame), policy-frame command,
  and the alive mask of the task criterion.

Run from the repo root:
  .venv/bin/python scripts/eval/lipm_diagnostics/behavior_record.py <ckpt> <terrain> <out.npz>
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, "scripts/eval")
import lipm_eval as L  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import wheeled_legged_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from wheeled_legged_mjlab.tasks.velocity.mdp.rewards import (  # noqa: E402
    _roughness_gate_active,
    _terrain_roughness_from_sensor,
)

WHEEL_RADIUS = 0.127
GATE_THRESHOLD = 0.65


def build(ckpt_name: str, env_cfg, num_envs: int, device: str = "cuda:0", spawn=None):
    """Env + policy for the frozen protocol; spawn(env) may override env origins."""
    spec = L._checkpoint(ckpt_name)
    agent_cfg = load_rl_cfg(spec.task_id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if spawn is None:
        rows, cols = L.spawn_layout(num_envs)
        t = env.scene.terrain
        r_t = torch.as_tensor(rows, device=device)
        c_t = torch.as_tensor(cols, device=device)
        t.terrain_levels[:] = r_t
        t.terrain_types[:] = c_t
        t.env_origins[:] = t.terrain_origins[r_t, c_t]
    else:
        spawn(env)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(spec.task_id) or MjlabOnPolicyRunner)(wrapper, asdict(agent_cfg), device=device)
    runner.load(spec.path, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return env, wrapper, policy, agent_cfg.clip_actions


def rollout(env, wrapper, policy, clip, num_steps: int, device: str = "cuda:0") -> dict:
    n = env.num_envs
    env.common_step_counter = 0
    obs = TensorDict(env.reset(seed=L.EVAL_SEED)[0], batch_size=[n])
    policy.reset()
    robot = env.scene["robot"]
    wheel_ids = torch.as_tensor(robot.find_joints("wheel_[LR]_Joint")[0], device=device)
    leg_ids = torch.as_tensor(
        robot.find_joints(("abad_[LR]_Joint", "hip_[LR]_Joint", "knee_[LR]_Joint"))[0], device=device
    )
    contact = env.scene["wheels_ground_contact"]
    clearance_sensor = env.scene["wheel_height_scan"]
    command = env.command_manager.get_term("twist")
    manager = env.termination_manager

    def buf(*shape, dtype=torch.float16):
        return torch.zeros((num_steps, n, *shape), dtype=dtype, device=device)

    rec = {
        "gate_lambda": buf(), "rough": buf(dtype=torch.bool), "wheel_contact": buf(2, dtype=torch.bool),
        "wheel_clearance": buf(2), "clearance_target": buf(2), "power_leg": buf(), "power_wheel": buf(),
        "speed": buf(), "gravity_b": buf(3), "ang_vel_b": buf(3), "command_b": buf(3),
        "pos_w": buf(3, dtype=torch.float32), "valid": buf(dtype=torch.bool),
    }
    alive = torch.ones(n, dtype=torch.bool, device=device)
    for k in range(num_steps):
        with torch.no_grad():
            actions = torch.clamp(policy(obs), -clip, clip)
        env._manual_reset_pending.zero_()
        obs, _, _, _ = wrapper.step(actions)
        terminated = manager.terminated.clone()
        stats = _terrain_roughness_from_sensor(
            env, "terrain_scan", wheel_radius=WHEEL_RADIUS, gate_min=0.0, gate_max=0.5,
            grid_shape=(11, 11), log=False,
        )
        heights = clearance_sensor.data.heights  # [B, 2, 25] wheel-centre height above terrain
        local_range = heights.max(-1).values - heights.min(-1).values
        rec["gate_lambda"][k] = stats.gate.view(n)
        rec["rough"][k] = _roughness_gate_active(stats.gate, GATE_THRESHOLD).view(n) > 0
        rec["wheel_contact"][k] = contact.data.found.view(n, -1)[:, :2] > 0
        rec["wheel_clearance"][k] = torch.clamp(heights.amin(-1) - WHEEL_RADIUS, min=0.0)
        rec["clearance_target"][k] = torch.clamp(0.03 + 0.5 * local_range, max=0.18)
        power = robot.data.qfrc_actuator * robot.data.joint_vel
        rec["power_leg"][k] = power[:, leg_ids].abs().sum(-1)
        rec["power_wheel"][k] = power[:, wheel_ids].abs().sum(-1)
        rec["speed"][k] = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
        rec["gravity_b"][k] = robot.data.projected_gravity_b
        rec["ang_vel_b"][k] = robot.data.root_link_ang_vel_b
        rec["command_b"][k] = command.command
        rec["pos_w"][k] = robot.data.root_link_pos_w
        rec["valid"][k] = alive & ~terminated
        alive &= ~terminated
    data = {k: v.cpu().numpy() for k, v in rec.items()}
    data["mass_kg"] = env.sim.model.body_mass[:, robot.indexing.body_ids].sum(-1).cpu().numpy()
    data["step_dt"] = np.float64(env.step_dt)
    return data


if __name__ == "__main__":
    ckpt_name, terrain, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    configure_torch_backends(allow_tf32=False, deterministic=True)
    spec = L._checkpoint(ckpt_name)
    cfg, _ = L.make_eval_env_cfg(spec.task_id, terrain, L.NUM_ENVS, L.EVAL_SEED)
    env, wrapper, policy, clip = build(ckpt_name, cfg, L.NUM_ENVS)
    data = rollout(env, wrapper, policy, clip, L.NUM_STEPS)
    np.savez_compressed(out_path, **data)
    print("saved", out_path)
