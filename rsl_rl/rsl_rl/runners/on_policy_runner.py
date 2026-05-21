# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import json
import os
from pathlib import Path
import time
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[PPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self._nan_debug_dumped = False

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for rollout_step in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    try:
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    except Exception as exc:
                        self._dump_nan_debug(it, rollout_step, actions=actions, exc=exc)
                        raise
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        try:
                            check_nan(obs, rewards, dones)
                        except Exception as exc:
                            self._dump_nan_debug(
                                it,
                                rollout_step,
                                obs=obs,
                                rewards=rewards,
                                dones=dones,
                                actions=actions,
                                exc=exc,
                            )
                            raise
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def _dump_nan_debug(
        self,
        iteration: int,
        rollout_step: int,
        *,
        obs=None,
        rewards: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        exc: Exception | None = None,
    ) -> None:
        """Write a compact debug report when NaN/Inf reaches RSL-RL."""
        if self._nan_debug_dumped:
            return
        self._nan_debug_dumped = True

        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None:
            return
        out_dir = Path(log_dir) / "nan_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"nan_debug_it{iteration:06d}_step{rollout_step:03d}.json"

        env = getattr(self.env, "unwrapped", self.env)
        report: dict = {
            "iteration": iteration,
            "rollout_step": rollout_step,
            "exception": repr(exc) if exc is not None else None,
            "num_envs": getattr(self.env, "num_envs", None),
            "bad_env_ids": [],
            "obs_groups": {},
            "raw_observation_terms": {},
            "physics": {},
            "terrain": {},
            "robot": {},
            "actions": self._tensor_summary(actions),
            "rewards": self._tensor_summary(rewards),
            "dones": self._tensor_summary(dones),
        }

        bad_env_ids = self._bad_env_ids_from_outputs(obs, rewards, dones)
        raw_bad_env_ids, raw_terms = self._raw_observation_term_report(env)
        report["raw_observation_terms"] = raw_terms
        if raw_bad_env_ids.numel() > 0:
            bad_env_ids = self._merge_env_ids(bad_env_ids, raw_bad_env_ids)

        physics_bad_env_ids, physics_report = self._physics_report(env)
        report["physics"] = physics_report
        if physics_bad_env_ids.numel() > 0:
            bad_env_ids = self._merge_env_ids(bad_env_ids, physics_bad_env_ids)

        if obs is not None:
            for key, tensor in obs.items():
                report["obs_groups"][key] = self._tensor_summary(tensor)

        report["bad_env_ids"] = bad_env_ids[:50].detach().cpu().tolist()
        report["terrain"] = self._terrain_report(env, bad_env_ids)
        report["robot"] = self._robot_report(env, bad_env_ids)

        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"[NaNDebug] Wrote debug report to: {path}")

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor | None) -> dict | None:
        if tensor is None:
            return None
        with torch.no_grad():
            finite = torch.isfinite(tensor)
            safe = torch.nan_to_num(tensor.detach())
            summary = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
                "finite_count": int(finite.sum().item()),
            }
            if tensor.numel() > 0:
                summary.update(
                    {
                        "min": float(safe.min().item()),
                        "max": float(safe.max().item()),
                        "mean": float(safe.float().mean().item()),
                    }
                )
            return summary

    @staticmethod
    def _bad_env_ids_for_tensor(tensor: torch.Tensor | None) -> torch.Tensor:
        if tensor is None:
            return torch.empty(0, dtype=torch.long)
        bad = torch.isnan(tensor) | torch.isinf(tensor)
        if bad.ndim == 0:
            return torch.tensor([0], device=tensor.device, dtype=torch.long) if bool(bad.item()) else torch.empty(
                0, device=tensor.device, dtype=torch.long
            )
        bad = bad.reshape(bad.shape[0], -1).any(dim=1)
        return torch.where(bad)[0]

    def _bad_env_ids_from_outputs(self, obs, rewards, dones) -> torch.Tensor:
        ids = torch.empty(0, dtype=torch.long, device=self.device)
        if obs is not None:
            for tensor in obs.values():
                ids = self._merge_env_ids(ids, self._bad_env_ids_for_tensor(tensor))
        ids = self._merge_env_ids(ids, self._bad_env_ids_for_tensor(rewards))
        ids = self._merge_env_ids(ids, self._bad_env_ids_for_tensor(dones))
        return ids

    @staticmethod
    def _merge_env_ids(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.numel() == 0:
            return right.to(dtype=torch.long)
        if right.numel() == 0:
            return left.to(dtype=torch.long)
        return torch.unique(torch.cat((left.to(right.device), right.to(dtype=torch.long))))

    def _raw_observation_term_report(self, env) -> tuple[torch.Tensor, dict]:
        obs_manager = getattr(env, "observation_manager", None)
        if obs_manager is None:
            return torch.empty(0, dtype=torch.long, device=self.device), {}

        bad_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        report: dict = {}
        for group_name in obs_manager.active_terms:
            report[group_name] = {}
            names = obs_manager.active_terms[group_name]
            cfgs = obs_manager._group_obs_term_cfgs[group_name]
            for term_name, term_cfg in zip(names, cfgs, strict=False):
                try:
                    tensor = term_cfg.func(env, **term_cfg.params).clone()
                    term_bad_ids = self._bad_env_ids_for_tensor(tensor)
                    bad_env_ids = self._merge_env_ids(bad_env_ids, term_bad_ids)
                    item = self._tensor_summary(tensor)
                    assert item is not None
                    item["bad_env_ids"] = term_bad_ids[:50].detach().cpu().tolist()
                except Exception as exc:
                    item = {"error": repr(exc)}
                report[group_name][term_name] = item
        return bad_env_ids, report

    def _physics_report(self, env) -> tuple[torch.Tensor, dict]:
        sim = getattr(env, "sim", None)
        if sim is None:
            return torch.empty(0, dtype=torch.long, device=self.device), {}
        data = getattr(sim, "data", None)
        if data is None:
            return torch.empty(0, dtype=torch.long, device=self.device), {}

        report = {}
        bad_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata", "ctrl"):
            if hasattr(data, name):
                tensor = getattr(data, name)
                report[name] = self._tensor_summary(tensor)
                bad_env_ids = self._merge_env_ids(bad_env_ids, self._bad_env_ids_for_tensor(tensor))
        return bad_env_ids, report

    def _terrain_report(self, env, env_ids: torch.Tensor) -> dict:
        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        if terrain is None:
            return {}
        report = {}
        sub_terrains = getattr(getattr(terrain.cfg, "terrain_generator", None), "sub_terrains", None)
        names = list(sub_terrains.keys()) if sub_terrains is not None else []
        for attr in ("terrain_levels", "terrain_types", "env_origins"):
            if hasattr(terrain, attr):
                tensor = getattr(terrain, attr)
                report[attr] = self._tensor_summary(tensor)
                if env_ids.numel() > 0 and tensor.ndim > 0:
                    report[f"{attr}_bad_envs"] = tensor[env_ids[:20].to(tensor.device)].detach().cpu().tolist()
        if names and hasattr(terrain, "terrain_types") and env_ids.numel() > 0:
            type_ids = terrain.terrain_types[env_ids[:20].to(terrain.terrain_types.device)].detach().cpu().tolist()
            report["terrain_type_names_bad_envs"] = [names[i] if 0 <= i < len(names) else str(i) for i in type_ids]
        return report

    def _robot_report(self, env, env_ids: torch.Tensor) -> dict:
        scene = getattr(env, "scene", None)
        if scene is None or "robot" not in getattr(scene, "entities", {}):
            return {}
        robot = scene.entities["robot"]
        data = robot.data
        report = {}
        for name in (
            "root_link_pose_w",
            "root_link_vel_w",
            "projected_gravity_b",
            "joint_pos",
            "joint_vel",
            "joint_acc",
        ):
            try:
                tensor = getattr(data, name)
                report[name] = self._tensor_summary(tensor)
                if env_ids.numel() > 0 and tensor.ndim > 0:
                    report[f"{name}_bad_envs"] = tensor[env_ids[:20].to(tensor.device)].detach().cpu().tolist()
            except Exception as exc:
                report[name] = {"error": repr(exc)}
        return report

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
