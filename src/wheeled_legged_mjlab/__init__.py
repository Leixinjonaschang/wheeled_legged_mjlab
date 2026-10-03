"""WF-TRON1B mjlab task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from wheeled_legged_mjlab.rl import WheeledLeggedVelocityOnPolicyRunner
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    wf_tron1b_flat_env_cfg,
    wf_tron1b_flat_rep_ts_lin_vel_env_cfg,
    wf_tron1b_rough_env_cfg,
    wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg,
    wf_tron1b_rough_rep_ts_lin_vel_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.rl_cfg import (
    wf_tron1b_ppo_runner_cfg,
    wf_tron1b_rep_ts_lin_vel_blind_predict_runner_cfg,
    wf_tron1b_rep_ts_lin_vel_blind_runner_cfg,
    wf_tron1b_rep_ts_lin_vel_depth_predict_runner_cfg,
    wf_tron1b_rep_ts_lin_vel_depth_predict_rggp_runner_cfg,
    wf_tron1b_rep_ts_lin_vel_depth_runner_cfg,
    wf_tron1b_rep_ts_lin_vel_runner_cfg,
    wf_tron1b_rep_ts_runner_cfg,
    without_dynamics_context,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_env_cfg(play=True),
    rl_cfg=wf_tron1b_rep_ts_lin_vel_blind_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Predict-BlindGP",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_env_cfg(play=True),
    rl_cfg=wf_tron1b_rep_ts_lin_vel_blind_predict_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-LPGP",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(play=True),
    rl_cfg=wf_tron1b_rep_ts_lin_vel_depth_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-OursGP",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(play=True),
    rl_cfg=wf_tron1b_rep_ts_lin_vel_depth_predict_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-RGGP",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(
        roughness_conditioned_rewards=False,
    ),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(
        play=True, roughness_conditioned_rewards=False,
    ),
    rl_cfg=wf_tron1b_rep_ts_lin_vel_depth_predict_rggp_runner_cfg(),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

# Same ablations with the critic and teacher (privileged) encoder trained without the
# dynamics context; the environments are unchanged.
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-LPGP-no_dyn_ctx",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(play=True),
    rl_cfg=without_dynamics_context(wf_tron1b_rep_ts_lin_vel_depth_runner_cfg()),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-OursGP-no_dyn_ctx",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(play=True),
    rl_cfg=without_dynamics_context(wf_tron1b_rep_ts_lin_vel_depth_predict_runner_cfg()),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-RGGP-no_dyn_ctx",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(
        roughness_conditioned_rewards=False,
    ),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(
        play=True, roughness_conditioned_rewards=False,
    ),
    rl_cfg=without_dynamics_context(
        wf_tron1b_rep_ts_lin_vel_depth_predict_rggp_runner_cfg()
    ),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP-no_dyn_ctx",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_env_cfg(play=True),
    rl_cfg=without_dynamics_context(wf_tron1b_rep_ts_lin_vel_runner_cfg()),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)

# Evaluation only: the blind policy above in the depth tasks' environment. The depth
# camera is rendered but never read by the policy; what it changes is the random stream
# (the async depth buffer draws its delays from the CUDA generator when the observation
# manager is built and at every reset), so only this environment gives the blind policy
# the same startup DR and resampled commands as the depth checkpoints' evaluations.
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP-no_dyn_ctx-DepthEnv",
    env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(),
    play_env_cfg=wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(play=True),
    rl_cfg=without_dynamics_context(wf_tron1b_rep_ts_lin_vel_runner_cfg()),
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)
