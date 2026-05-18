"""WF-TRON1B mjlab task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from wheeled_legged_mjlab.rl import WheeledLeggedVelocityOnPolicyRunner
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    wf_tron1b_flat_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.rl_cfg import (
    wf_tron1b_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B",
    env_cfg=wf_tron1b_rough_env_cfg(),
    play_env_cfg=wf_tron1b_rough_env_cfg(play=True),
    rl_cfg=wf_tron1b_ppo_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-WF-Tron1B",
    env_cfg=wf_tron1b_flat_env_cfg(),
    play_env_cfg=wf_tron1b_flat_env_cfg(play=True),
    rl_cfg=wf_tron1b_ppo_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)
