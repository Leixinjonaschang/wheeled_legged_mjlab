"""RSL-RL runners for wheeled-legged velocity tasks."""

from __future__ import annotations

from pathlib import Path

import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from rsl_rl.models import (
    DepthRepresentationVelocityActorCritic,
    RepresentationActorCritic,
    RepresentationVelocityActorCritic,
)


def _action_scale_values(action_term) -> list[float]:
    scale = action_term.scale
    if hasattr(scale, "detach"):
        return scale[0].detach().cpu().tolist()
    return [float(scale)] * action_term.action_dim


def get_wheeled_legged_metadata(
    env: ManagerBasedRlEnv,
    run_path: str,
    policy=None,
) -> dict[str, list | str | float]:
    """Metadata export that supports mixed position/velocity action terms."""
    robot = env.scene["robot"]

    action_names: list[str] = []
    action_target_names: list[str] = []
    action_scales: list[float] = []
    for action_name in env.action_manager.active_terms:
        action_term = env.action_manager.get_term(action_name)
        action_names.extend([action_name] * action_term.action_dim)
        action_target_names.extend(action_term.target_names)
        action_scales.extend(_action_scale_values(action_term))

    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id
    ctrl_ids_natural = [
        joint_name_to_ctrl_id[joint_name]
        for joint_name in robot.joint_names
        if joint_name in joint_name_to_ctrl_id
    ]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
    observation_group = "actor" if "actor" in env.observation_manager.active_terms else "proprio_history"
    observation_names = env.observation_manager.active_terms[observation_group]
    observation_terms_scale = []
    observation_terms_flatten_history_dim = []
    observation_terms_history_length = []
    observation_terms_clip = []
    for observation_name in observation_names:
        cfg = env.observation_manager.get_term_cfg(observation_group, observation_name)
        scale = cfg.scale
        observation_terms_scale.append(
            scale.cpu().tolist() if hasattr(scale, "cpu") else (1.0 if scale is None else scale)
        )
        observation_terms_flatten_history_dim.append(cfg.flatten_history_dim)
        observation_terms_history_length.append(cfg.history_length)
        observation_terms_clip.append(
            [float("-inf"), float("inf")] if cfg.clip is None else list(cfg.clip)
        )

    metadata = {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness.tolist(),
        "joint_damping": joint_damping.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": observation_names,
        "observation_terms_scale": observation_terms_scale,
        "observation_terms_flatten_history_dim": observation_terms_flatten_history_dim,
        "observation_terms_history_length": observation_terms_history_length,
        "observation_terms_clip": observation_terms_clip,
        "action_names": action_names,
        "action_target_names": action_target_names,
        "action_scale": action_scales,
    }
    if isinstance(policy, RepresentationActorCritic):
        student_history_cfg = env.cfg.observations["actor_history"]
        metadata.update(
            {
                "policy_input_names": ["student_history"],
                "student_observation_names": env.observation_manager.active_terms["actor_history"],
                "student_history_length": str(student_history_cfg.history_length),
                "student_history_flatten_dim": str(student_history_cfg.flatten_history_dim).lower(),
                "student_history_order": "oldest_to_newest",
            }
        )
    if isinstance(policy, DepthRepresentationVelocityActorCritic):
        proprio_history_cfg = env.cfg.observations["proprio_history"]
        metadata.update(
            {
                "policy_input_names": ["proprio_history", "actor_command", "depth", "hidden_state_in"],
                "policy_output_names": ["actions", "predicted_lin_vel", "hidden_state_out"],
                "student_observation_names": env.observation_manager.active_terms["proprio_history"],
                "command_observation_names": env.observation_manager.active_terms["actor_command"],
                "student_history_length": str(proprio_history_cfg.history_length),
                "student_history_flatten_dim": str(proprio_history_cfg.flatten_history_dim).lower(),
                "student_history_order": "oldest_to_newest",
                "depth_input_dtype": "float32",
                "depth_input_unit": "m",
                "depth_input_shape": [1, *policy.depth_shape],
                "depth_input_range": [
                    policy.depth_preprocessor.depth_min_m,
                    policy.depth_preprocessor.depth_max_m,
                ],
                "depth_invalid_value": policy.depth_preprocessor.depth_max_m,
                "depth_min_m": policy.depth_preprocessor.depth_min_m,
                "depth_max_m": policy.depth_preprocessor.depth_max_m,
                "depth_preprocessing": "external:below_min_to_max,clamp",
            }
        )
    elif isinstance(policy, RepresentationVelocityActorCritic):
        proprio_history_cfg = env.cfg.observations["proprio_history"]
        metadata.update(
            {
                "policy_input_names": ["proprio_history", "actor_command"],
                "policy_output_names": ["actions", "predicted_lin_vel"],
                "student_observation_names": env.observation_manager.active_terms["proprio_history"],
                "command_observation_names": env.observation_manager.active_terms["actor_command"],
                "student_history_length": str(proprio_history_cfg.history_length),
                "student_history_flatten_dim": str(proprio_history_cfg.flatten_history_dim).lower(),
                "student_history_order": "oldest_to_newest",
            }
        )
    return metadata


class WheeledLeggedVelocityOnPolicyRunner(VelocityOnPolicyRunner):
    """Velocity runner with ONNX metadata for mixed leg/wheel actions."""

    env: RslRlVecEnvWrapper

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        """Export ONNX and attach the wheeled-legged input contract."""
        super().export_policy_to_onnx(path, filename, verbose)
        run_name = (
            wandb.run.name
            if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
            else "local"
        )
        metadata = get_wheeled_legged_metadata(
            self.env.unwrapped,
            run_name,
            self.alg.get_policy(),
        )
        attach_metadata_to_onnx(str(Path(path) / filename), metadata)

    def save(self, path: str, infos=None):
        super(VelocityOnPolicyRunner, self).save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            if (
                self.logger.logger_type in ("wandb", "WandbLogWriter")
                and self.cfg["upload_model"]
            ):
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as exc:
            print(f"[WARN] ONNX export failed (training continues): {exc}")
