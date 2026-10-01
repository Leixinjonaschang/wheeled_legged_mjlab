"""Throwaway diagnostic: which non-wheel geoms trigger illegal_contact at the first failure.

Re-simulates the frozen evaluation (same seeds, terrain, commands) for one checkpoint
and terrain and records per-geom contact at the step where illegal_contact first fires.
"""
import json
import sys
from dataclasses import asdict

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

ckpt_name, terrain = sys.argv[1], sys.argv[2]
N, T = L.NUM_ENVS, L.NUM_STEPS
spec = L._checkpoint(ckpt_name)
configure_torch_backends(allow_tf32=False, deterministic=True)
cfg, _ = L.make_eval_env_cfg(spec.task_id, terrain, N, L.EVAL_SEED)
agent_cfg = load_rl_cfg(spec.task_id)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
rows, cols = L.spawn_layout(N)
t = env.scene.terrain
r_t = torch.as_tensor(rows, device="cuda:0")
c_t = torch.as_tensor(cols, device="cuda:0")
t.terrain_levels[:] = r_t
t.terrain_types[:] = c_t
t.env_origins[:] = t.terrain_origins[r_t, c_t]
wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(spec.task_id) or MjlabOnPolicyRunner)(wrapper, asdict(agent_cfg), device="cuda:0")
runner.load(spec.path, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
env.common_step_counter = 0
obs = TensorDict(env.reset(seed=L.EVAL_SEED)[0], batch_size=[N])
policy.reset()
sensor = env.scene["illegal_ground_contact"]
names = sensor.primary_names
manager = env.termination_manager
alive = torch.ones(N, dtype=torch.bool, device="cuda:0")
counts = {n: 0 for n in names}
n_illegal = 0
for k in range(T):
    with torch.no_grad():
        actions = policy(obs)
    env._manual_reset_pending.zero_()
    obs, _, _, _ = wrapper.step(actions)
    terminated = manager.terminated.clone()
    illegal = manager.get_term("illegal_contact").clone()
    first = alive & terminated & illegal
    if first.any():
        force = torch.norm(sensor.data.force_history, dim=-1)  # [B, n_geoms * slots, H]
        per_geom = (force > 10.0).any(-1).view(N, len(names), -1).any(-1)
        for j, name in enumerate(names):
            counts[name] += int(per_geom[first, j].sum())
        n_illegal += int(first.sum())
    alive &= ~terminated
print("RESULT", json.dumps({"ckpt": ckpt_name, "terrain": terrain, "illegal_first_failures": n_illegal,
                            "geoms_in_contact_at_failure": counts}))
