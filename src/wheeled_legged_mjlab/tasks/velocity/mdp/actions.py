"""Task-side action terms for WF-TRON1B velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import (
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with per-env physics-frame actuator delay."""

    delay_range_s: tuple[float, float] = (0.0, 0.02)
    """Inclusive range of representable physics-step action delays, in seconds."""

    resampling_time_s: float = 5.0
    """Time between periodic per-environment delay resampling, in seconds."""

    delay_group: str = "actions"
    """Delay group shared by action terms that should use the same delay sample."""

    def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
        return DelayedJointPositionAction(self, env)


@dataclass(kw_only=True)
class DelayedJointVelocityActionCfg(JointVelocityActionCfg):
    """Joint velocity action with per-env physics-frame actuator delay."""

    delay_range_s: tuple[float, float] = (0.0, 0.02)
    """Inclusive range of representable physics-step action delays, in seconds."""

    resampling_time_s: float = 5.0
    """Time between periodic per-environment delay resampling, in seconds."""

    delay_group: str = "actions"
    """Delay group shared by action terms that should use the same delay sample."""

    def build(self, env: ManagerBasedRlEnv) -> DelayedJointVelocityAction:
        return DelayedJointVelocityAction(self, env)


class _DelayGroupState:
    """Shared delay samples for action terms in the same delay group."""

    def __init__(
        self,
        *,
        delay_range_s: tuple[float, float],
        resampling_time_s: float,
        physics_dt: float,
        step_dt: float,
        num_envs: int,
        device: str,
    ) -> None:
        delay_min_s, delay_max_s = delay_range_s
        if delay_min_s < 0.0 or delay_max_s < 0.0:
            raise ValueError(f"delay_range_s must be non-negative: {delay_range_s}")
        if delay_min_s > delay_max_s:
            raise ValueError(
                f"delay_range_s must be ordered min <= max: {delay_range_s}"
            )
        if physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be positive, got {physics_dt}")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        if resampling_time_s <= 0.0:
            raise ValueError(
                f"resampling_time_s must be positive, got {resampling_time_s}"
            )

        self.delay_range_s = (float(delay_min_s), float(delay_max_s))
        self.resampling_time_s = float(resampling_time_s)
        self.physics_dt = float(physics_dt)
        self.step_dt = float(step_dt)
        self.num_envs = int(num_envs)
        self.device = device
        self.min_delay_steps, self.max_delay_steps = _delay_step_bounds(
            self.delay_range_s, self.physics_dt
        )
        self.resampling_steps = max(1, round(resampling_time_s / step_dt))
        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._next_resample_step = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._last_prepared_step: int | None = None
        self.resample(slice(None), current_policy_step=0)

    def validate(
        self,
        *,
        delay_range_s: tuple[float, float],
        resampling_time_s: float,
        physics_dt: float,
        step_dt: float,
        num_envs: int,
        device: str,
    ) -> None:
        if (
            self.delay_range_s != (float(delay_range_s[0]), float(delay_range_s[1]))
            or self.resampling_time_s != float(resampling_time_s)
            or self.physics_dt != float(physics_dt)
            or self.step_dt != float(step_dt)
            or self.num_envs != int(num_envs)
            or self.device != device
        ):
            raise ValueError(
                "Action delay terms in the same delay_group must use identical "
                "delay_range_s, resampling_time_s, physics_dt, step_dt, num_envs, "
                "and device."
            )

    def prepare_for_policy_step(self, env: ManagerBasedRlEnv) -> None:
        current_step = _policy_step(env)
        if self._last_prepared_step == current_step:
            return
        self._last_prepared_step = current_step

        resample_env_ids = (
            (self._next_resample_step <= current_step)
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        if resample_env_ids.numel() > 0:
            self.resample(resample_env_ids, current_policy_step=current_step)

    def reset(
        self, env_ids: torch.Tensor | slice | None, *, current_policy_step: int
    ) -> None:
        self.resample(env_ids, current_policy_step=current_policy_step)

    def resample(
        self, env_ids: torch.Tensor | slice | None, *, current_policy_step: int
    ) -> None:
        env_ids_tensor = _env_ids_tensor(env_ids, self.num_envs, self.device)
        if env_ids_tensor.numel() == 0:
            return

        sampled_steps = torch.randint(
            self.min_delay_steps,
            self.max_delay_steps + 1,
            (env_ids_tensor.numel(),),
            device=self.device,
        )
        self.delay_steps[env_ids_tensor] = sampled_steps
        self._next_resample_step[env_ids_tensor] = (
            current_policy_step + self.resampling_steps
        )


class _DelayedActionMixin:
    cfg: DelayedJointPositionActionCfg | DelayedJointVelocityActionCfg

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._delay_state, self._owns_delay_state = _get_delay_group_state(
            env,
            delay_group=cfg.delay_group,
            delay_range_s=cfg.delay_range_s,
            resampling_time_s=cfg.resampling_time_s,
            physics_dt=float(env.physics_dt),
            step_dt=float(env.step_dt),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._processed_action_fifo = torch.zeros(
            self.num_envs,
            self._delay_state.max_delay_steps + 1,
            self.action_dim,
            device=self.device,
        )
        self._raw_action_fifo = torch.zeros_like(self._processed_action_fifo)
        self._delayed_processed_actions = torch.zeros_like(self._processed_actions)
        self._delayed_raw_actions = torch.zeros_like(self._raw_actions)
        self._applied_action_sum = torch.zeros_like(self._raw_actions)
        self._applied_action_substeps = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.long
        )
        self._reset_delay_buffer(slice(None))

    @property
    def delay_steps(self) -> torch.Tensor:
        """Per-environment action delay in physics frames."""
        return self._delay_state.delay_steps

    @property
    def applied_action(self) -> torch.Tensor:
        """Mean policy-space action actually applied during the current env step."""
        return self._applied_action_sum / self._applied_action_substeps.clamp_min(1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._delay_state.prepare_for_policy_step(self._env)
        super().process_actions(actions)
        self._applied_action_sum.zero_()
        self._applied_action_substeps.zero_()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._reset_processed_actions(env_ids)
        self._reset_delay_buffer(env_ids)
        self._delayed_raw_actions[env_ids] = self._raw_actions[env_ids]
        self._applied_action_sum[env_ids] = 0.0
        self._applied_action_substeps[env_ids] = 0
        if self._owns_delay_state:
            self._delay_state.reset(
                env_ids, current_policy_step=_policy_step(self._env)
            )

    def _delayed_actions_for_substep(self) -> torch.Tensor:
        self._processed_action_fifo = torch.roll(
            self._processed_action_fifo, shifts=1, dims=1
        )
        self._raw_action_fifo = torch.roll(self._raw_action_fifo, shifts=1, dims=1)
        self._processed_action_fifo[:, 0, :] = self._processed_actions
        self._raw_action_fifo[:, 0, :] = self._raw_actions
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._delayed_processed_actions = self._processed_action_fifo[
            env_ids, self._delay_state.delay_steps
        ]
        self._delayed_raw_actions = self._raw_action_fifo[
            env_ids, self._delay_state.delay_steps
        ]
        self._applied_action_sum += self._delayed_raw_actions
        self._applied_action_substeps += 1
        return self._delayed_processed_actions

    def _reset_delay_buffer(self, env_ids: torch.Tensor | slice) -> None:
        self._processed_action_fifo[env_ids] = self._processed_actions[
            env_ids
        ].unsqueeze(1)
        self._raw_action_fifo[env_ids] = self._raw_actions[env_ids].unsqueeze(1)

    def _reset_processed_actions(self, env_ids: torch.Tensor | slice) -> None:
        raw_actions = self._raw_actions[env_ids]
        scale = (
            self._scale[env_ids]
            if isinstance(self._scale, torch.Tensor)
            else self._scale
        )
        offset = (
            self._offset[env_ids]
            if isinstance(self._offset, torch.Tensor)
            else self._offset
        )
        processed_actions = raw_actions * scale + offset
        if self.cfg.clip is not None:
            processed_actions = torch.clamp(
                processed_actions,
                min=self._clip[env_ids, :, 0],
                max=self._clip[env_ids, :, 1],
            )
        self._processed_actions[env_ids] = processed_actions


class DelayedJointPositionAction(_DelayedActionMixin, JointPositionAction):
    """Position action that applies processed targets through a delay FIFO."""

    def apply_actions(self) -> None:
        delayed_actions = self._delayed_actions_for_substep()
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = delayed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


class DelayedJointVelocityAction(_DelayedActionMixin, JointVelocityAction):
    """Velocity action that applies processed targets through a delay FIFO."""

    def apply_actions(self) -> None:
        self._entity.set_joint_velocity_target(
            self._delayed_actions_for_substep(), joint_ids=self._target_ids
        )


def _get_delay_group_state(
    env: ManagerBasedRlEnv,
    *,
    delay_group: str,
    delay_range_s: tuple[float, float],
    resampling_time_s: float,
    physics_dt: float,
    step_dt: float,
    num_envs: int,
    device: str,
) -> tuple[_DelayGroupState, bool]:
    groups = getattr(env, "_wheeled_legged_action_delay_groups", None)
    if groups is None:
        groups = {}
        setattr(env, "_wheeled_legged_action_delay_groups", groups)

    if delay_group not in groups:
        groups[delay_group] = _DelayGroupState(
            delay_range_s=delay_range_s,
            resampling_time_s=resampling_time_s,
            physics_dt=physics_dt,
            step_dt=step_dt,
            num_envs=num_envs,
            device=device,
        )
        is_owner = True
    else:
        groups[delay_group].validate(
            delay_range_s=delay_range_s,
            resampling_time_s=resampling_time_s,
            physics_dt=physics_dt,
            step_dt=step_dt,
            num_envs=num_envs,
            device=device,
        )
        is_owner = False
    return groups[delay_group], is_owner


def _delay_step_bounds(
    delay_range_s: tuple[float, float], physics_dt: float
) -> tuple[int, int]:
    delay_min_s, delay_max_s = delay_range_s
    tolerance = max(abs(physics_dt) * 1.0e-9, 1.0e-12)
    min_steps = max(0, math.ceil((delay_min_s - tolerance) / physics_dt))
    max_steps = math.floor((delay_max_s + tolerance) / physics_dt)
    if min_steps > max_steps:
        raise ValueError(
            "delay_range_s must include at least one representable physics-step "
            f"delay, got delay_range_s={delay_range_s} and physics_dt={physics_dt}."
        )
    return min_steps, max_steps


def _policy_step(env: ManagerBasedRlEnv) -> int:
    return int(getattr(env, "common_step_counter", 0))


def _env_ids_tensor(
    env_ids: torch.Tensor | slice | None, num_envs: int, device: str
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return env_ids.to(device=device, dtype=torch.long)
