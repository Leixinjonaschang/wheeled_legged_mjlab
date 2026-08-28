"""RSL-RL environment wrapper with applied-action transition metadata."""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper


def get_applied_actions(
    action_manager,
    dones: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate applied policy-space actions in action-manager order."""
    applied_actions = []
    for name in action_manager.active_terms:
        term = action_manager.get_term(name)
        applied_action = getattr(term, "applied_action", term.raw_action)
        if dones is not None and hasattr(term, "terminal_applied_action"):
            applied_action = applied_action.clone()
            applied_action[dones.bool()] = term.terminal_applied_action[dones.bool()]
        applied_actions.append(applied_action)
    return torch.cat(applied_actions, dim=-1)


class WheeledLeggedRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Expose the action actually applied during each environment step."""

    def step(self, actions: torch.Tensor):
        observations, rewards, dones, extras = super().step(actions)
        extras["applied_actions"] = get_applied_actions(
            self.unwrapped.action_manager,
            dones,
        )
        return observations, rewards, dones, extras
