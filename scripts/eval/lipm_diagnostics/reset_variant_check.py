"""Throwaway: flat-ground failure rate under reset / action-delay variants."""
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, "scripts/eval")
import lipm_eval as L  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import wheeled_legged_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from tensordict import TensorDict  # noqa: E402

variant, ckpt_name, terrain = sys.argv[1], sys.argv[2], sys.argv[3]
N, T = 256, 150
spec = L._checkpoint(ckpt_name)
configure_torch_backends(allow_tf32=False, deterministic=True)
cfg, train_cfg = L.make_eval_env_cfg(spec.task_id, terrain, N, L.EVAL_SEED)
if variant in ("no_joint_offset", "no_offsets_no_vel"):
    cfg.events["reset_leg_joints"].params["position_range"] = (0.0, 0.0)
if variant == "no_offsets_no_vel":
    for k in cfg.events["reset_base"].params["velocity_range"]:
        cfg.events["reset_base"].params["velocity_range"][k] = (0.0, 0.0)
if variant == "train_actions":  # action delay + observation noise as in training
    cfg.actions = train_cfg.actions
    cfg.observations["proprio_history"].enable_corruption = True
agent_cfg = load_rl_cfg(spec.task_id)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
rows, cols = L.spawn_layout(N)
t = env.scene.terrain
r_t = torch.as_tensor(rows, device="cuda:0"); c_t = torch.as_tensor(cols, device="cuda:0")
t.env_origins[:] = t.terrain_origins[r_t, c_t]
wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(spec.task_id) or MjlabOnPolicyRunner)(wrapper, asdict(agent_cfg), device="cuda:0")
runner.load(spec.path, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
env.common_step_counter = 0
obs = TensorDict(env.reset(seed=L.EVAL_SEED)[0], batch_size=[N])
policy.reset()
jp0 = env.scene["robot"].data.joint_pos.clone()
alive = torch.ones(N, dtype=torch.bool, device="cuda:0")
fail_t = torch.full((N,), -1, device="cuda:0")
for k in range(T):
    with torch.no_grad():
        a = policy(obs)
    env._manual_reset_pending.zero_()
    obs, _, _, _ = wrapper.step(a)
    term = env.termination_manager.terminated
    fail_t[alive & term] = k + 1
    alive &= ~term
failed = ~alive
abad = jp0[:, [0, 4]]
print(f"RESULT variant={variant} ckpt={ckpt_name} terrain={terrain} fail={int(failed.sum())}/{N} "
      f"mean_fail_step={fail_t[failed].float().mean().item() if failed.any() else float('nan'):.1f} "
      f"| failed abad_L mean={abad[failed, 0].mean().item() if failed.any() else float('nan'):.2f} "
      f"abad_R mean={abad[failed, 1].mean().item() if failed.any() else float('nan'):.2f}")
