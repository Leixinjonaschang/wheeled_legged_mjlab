"""RSL-RL environment wrapper with applied-action transition metadata."""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper


def get_applied_actions(action_manager) -> torch.Tensor:
    """Concatenate applied policy-space actions in action-manager order."""
    applied_actions = []
    for name in action_manager.active_terms:
        term = action_manager.get_term(name)
        applied_actions.append(getattr(term, "applied_action", term.raw_action))
    return torch.cat(applied_actions, dim=-1)


class WheeledLeggedRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Expose the action actually applied during each environment step."""

    def step(self, actions: torch.Tensor):
        observations, rewards, dones, extras = super().step(actions)
        extras["applied_actions"] = get_applied_actions(self.unwrapped.action_manager)
        return observations, rewards, dones, extras
