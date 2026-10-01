"""Energy and wheel-contact diagnostics under the frozen evaluation protocol.

Re-simulates one checkpoint on one terrain (same seeds, terrain, commands as the main
evaluation) and records, while the robot is alive under the task criterion:
mechanical joint power sum_j |tau_j * qd_j| (as in the joint_power reward), base
horizontal speed, wheel-torque saturation, wheel ground contact, and the local terrain
relief under the base (terrain_scan max - min height over 1 m x 1 m).

Run from the repo root:
  .venv/bin/python scripts/eval/lipm_diagnostics/energy_contact.py <ckpt> <terrain> <out.npz>
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

ckpt_name, terrain, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
N, T = L.NUM_ENVS, L.NUM_STEPS
device = "cuda:0"
spec = L._checkpoint(ckpt_name)
configure_torch_backends(allow_tf32=False, deterministic=True)
cfg, _ = L.make_eval_env_cfg(spec.task_id, terrain, N, L.EVAL_SEED)
agent_cfg = load_rl_cfg(spec.task_id)
env = ManagerBasedRlEnv(cfg=cfg, device=device)
rows, cols = L.spawn_layout(N)
t = env.scene.terrain
r_t = torch.as_tensor(rows, device=device)
c_t = torch.as_tensor(cols, device=device)
t.terrain_levels[:] = r_t
t.terrain_types[:] = c_t
t.env_origins[:] = t.terrain_origins[r_t, c_t]
wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(spec.task_id) or MjlabOnPolicyRunner)(wrapper, asdict(agent_cfg), device=device)
runner.load(spec.path, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
env.common_step_counter = 0
obs = TensorDict(env.reset(seed=L.EVAL_SEED)[0], batch_size=[N])
policy.reset()

robot = env.scene["robot"]
wheel_ids, _ = robot.find_joints("wheel_[LR]_Joint")
leg_ids, _ = robot.find_joints(("abad_[LR]_Joint", "hip_[LR]_Joint", "knee_[LR]_Joint"))
wheel_ids = torch.as_tensor(wheel_ids, device=device)
leg_ids = torch.as_tensor(leg_ids, device=device)
mass = env.sim.model.body_mass[:, robot.indexing.body_ids].sum(-1)  # Per-env, after DR.
contact = env.scene["wheels_ground_contact"]
scan = env.scene["terrain_scan"]
manager = env.termination_manager
wheel_force_limit = 40.0  # forcerange of the wheel velocity actuators in robot.xml.

series = {k: torch.zeros(T, N, device=device) for k in (
    "power_leg", "power_wheel", "speed", "relief", "wheel_saturated_frac",
)}
series["wheels_in_contact"] = torch.zeros(T, N, 2, device=device, dtype=torch.bool)
series["valid"] = torch.zeros(T, N, device=device, dtype=torch.bool)
alive = torch.ones(N, device=device, dtype=torch.bool)
for k in range(T):
    with torch.no_grad():
        actions = policy(obs)
    env._manual_reset_pending.zero_()
    obs, _, _, _ = wrapper.step(actions)
    terminated = manager.terminated.clone()
    power = robot.data.qfrc_actuator * robot.data.joint_vel
    series["power_leg"][k] = power[:, leg_ids].abs().sum(-1)
    series["power_wheel"][k] = power[:, wheel_ids].abs().sum(-1)
    series["speed"][k] = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    torque = robot.data.qfrc_actuator[:, wheel_ids].abs()
    series["wheel_saturated_frac"][k] = (torque >= 0.99 * wheel_force_limit).float().mean(-1)
    series["wheels_in_contact"][k] = contact.data.found.view(N, -1)[:, :2] > 0
    hit = scan.data.distances >= 0.0
    z = scan.data.hit_pos_w[..., 2]
    series["relief"][k] = (
        torch.where(hit, z, -torch.inf).amax(-1) - torch.where(hit, z, torch.inf).amin(-1)
    )
    series["valid"][k] = alive & ~terminated
    alive &= ~terminated

data = {k: v.cpu().numpy() for k, v in series.items()}
data["mass_kg"] = mass.cpu().numpy()
data["step_dt"] = np.float64(env.step_dt)
np.savez_compressed(out_path, **data)
print("saved", out_path)
