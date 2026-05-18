# Public asset export layer.
# Re-export robot configs here so callers can import from
# `wheeled_legged_mjlab.assets` without depending on the deeper file layout.
# This is not a task registry; env cfg files still choose which robot config
# to place in `SceneCfg.entities`.

from .WF_TRON1B.wf_tron1b import WF_TRON1B_ROBOT_CFG

__all__ = [
    "WF_TRON1B_ROBOT_CFG",
]
