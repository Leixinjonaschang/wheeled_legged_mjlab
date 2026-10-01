"""Collect teacher / student latents of all checkpoints on identical observations.

One behaviour policy drives the robots (frozen evaluation protocol, one terrain class);
at every step the SAME observation is fed to every checkpoint's privileged (teacher)
encoder and recurrent student encoder, so latent-space statistics compare encoders on
identical inputs rather than on each policy's own state distribution.

Run from the repo root:
  .venv/bin/python scripts/eval/lipm_diagnostics/latent_collect.py \
      <behaviour ckpt> <terrain> <out.npz> [num_envs]
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

behaviour, terrain, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
T = L.NUM_STEPS
device = "cuda:0"
MODELS = tuple(spec.name for spec in L.CHECKPOINTS)  # Ours, RGGP, LPGP
configure_torch_backends(allow_tf32=False, deterministic=True)

behaviour_spec = L._checkpoint(behaviour)
cfg, _ = L.make_eval_env_cfg(behaviour_spec.task_id, terrain, N, L.EVAL_SEED)
env = ManagerBasedRlEnv(cfg=cfg, device=device)
rows, cols = L.spawn_layout(N)
t = env.scene.terrain
r_t = torch.as_tensor(rows, device=device)
c_t = torch.as_tensor(cols, device=device)
t.terrain_levels[:] = r_t
t.terrain_types[:] = c_t
t.env_origins[:] = t.terrain_origins[r_t, c_t]
agent_cfg = load_rl_cfg(behaviour_spec.task_id)
wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

models = {}
for name in MODELS:
    spec = L._checkpoint(name)
    spec_cfg = load_rl_cfg(spec.task_id)
    runner = (load_runner_cls(spec.task_id) or MjlabOnPolicyRunner)(wrapper, asdict(spec_cfg), device=device)
    runner.load(spec.path, load_cfg={"actor": True}, strict=True, map_location=device)
    models[name] = runner.get_inference_policy(device=device)

env.common_step_counter = 0
obs = TensorDict(env.reset(seed=L.EVAL_SEED)[0], batch_size=[N])
for model in models.values():
    model.reset()

robot = env.scene["robot"]
contact = env.scene["wheels_ground_contact"]
scan = env.scene["terrain_scan"]
command = env.command_manager.get_term("twist")
manager = env.termination_manager
clip = agent_cfg.clip_actions


def zeros(*shape, dtype=torch.float32):
    return torch.zeros((T, N, *shape), dtype=dtype, device=device)


rec = {
    "z_teacher": zeros(len(MODELS), 64, dtype=torch.float16),
    "z_student": zeros(len(MODELS), 64, dtype=torch.float16),
    "actions": zeros(8), "valid": zeros(dtype=torch.bool), "relief": zeros(),
    "wheel_contact": zeros(2, dtype=torch.bool), "lin_vel_b": zeros(3), "ang_vel_b": zeros(3),
    "command_b": zeros(3), "gravity_b": zeros(3),
}
dynamics_context = obs["dynamics_context"].detach().cpu().numpy()
alive = torch.ones(N, dtype=torch.bool, device=device)
for k in range(T):
    with torch.no_grad():
        for j, name in enumerate(MODELS):
            model = models[name]
            acting = name == behaviour
            latent, _, _ = model._get_student_outputs(
                obs, update_hidden_state=not acting, use_internal_state=True
            )
            rec["z_student"][k, :, j] = latent.half()
            rec["z_teacher"][k, :, j] = model.get_privileged_latent(obs).half()
        actions = models[behaviour](obs)
        actions = torch.clamp(actions, -clip, clip)
    hit = scan.data.distances >= 0.0
    z = scan.data.hit_pos_w[..., 2]
    rec["relief"][k] = torch.where(hit, z, -torch.inf).amax(-1) - torch.where(hit, z, torch.inf).amin(-1)
    rec["wheel_contact"][k] = contact.data.found.view(N, -1)[:, :2] > 0
    rec["lin_vel_b"][k] = robot.data.root_link_lin_vel_b
    rec["ang_vel_b"][k] = robot.data.root_link_ang_vel_b
    rec["command_b"][k] = command.command
    rec["gravity_b"][k] = robot.data.projected_gravity_b
    rec["actions"][k] = actions
    rec["valid"][k] = alive
    env._manual_reset_pending.zero_()
    obs, _, _, _ = wrapper.step(actions)
    alive &= ~manager.terminated

data = {k: v.cpu().numpy() for k, v in rec.items()}
data.update(models=np.array(MODELS), behaviour=behaviour, terrain=terrain,
            dynamics_context=dynamics_context, step_dt=np.float64(env.step_dt))
np.savez_compressed(out_path, **data)
print("saved", out_path, {k: v.shape for k, v in data.items() if hasattr(v, "shape")})
