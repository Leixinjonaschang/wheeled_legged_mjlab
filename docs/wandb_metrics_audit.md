# W&B metrics 检查报告

检查日期：2026-09-05。范围：当前工作区训练代码、安装的 mjlab 1.6.0 manager 实现，以及本地 W&B summary。没有连接云端，也没有重新启动训练；summary 检查只覆盖各 run 的最终快照，不覆盖整段历史曲线。工作区检查过程中已有 action reward 拆分修改，以下配置清单按最终实际 import 结果生成。

## 主要发现

1. **命令误差受存活时长影响。** `Metrics/twist/error_vel_xy`、`error_vel_yaw` 每步加 `error × step_dt / 10s`，仅 episode reset 时清零，不在命令重采样时清零。恒定误差 1 时，存活 2 秒约记 0.2，20 秒约记 2；它们不是实际 episode 的平均速度误差。来源：`commands.py:77–90`、安装依赖 `command_manager.py:101–128`。建议按实际累计有效步数归一化。
2. **rough 指标的总体不一致。** `rough_double_contact_mean`、`rough_no_contact_mean`、`rough_single_contact_mean` 对所有环境取均值，没有使用奖励里的 rough/command mask（`rewards.py:1328`）；`rough_wheel_foot_clearance_mean/target_mean` 同样统计所有环境与轮（`rewards.py:1274`）。`rough_wheel_distance_mean/violation_mean` 则确实仅统计 rough 环境。建议统一命名或显式记录条件均值及样本数。
3. **episode 汇总按 reset 批次等权，而非 episode 等权。** Manager 先对本步结束的环境求均值，Logger 再对这些标量求均值（`logger.py:161–173`）。例如 1 个 episode 的均值 0、9 个 episode 的均值 10，记录 5，真正 episode 均值为 9。影响 `Episode_Reward/*`、`Episode_Metrics/*`、命令误差等。
4. **终止指标是每 reset 批次的平均计数。** `Episode_Termination/*` 不是终止概率，也不是本 iteration 的总次数；多个原因可以同时为真。改变环境数会改变计数量级。
5. **稀疏事件均值会被无事件步骤稀释。** 如 `landing_force_mean` 无落地时记 0，Logger 对每步标量再平均；不是整个 rollout 所有落地事件的加权平均。`rough_wheel_distance_*` 无 rough 样本同样记 0。建议累计分子和事件数后再相除。
6. **方向角包含无定义的静止情况。** `world_command_tracking_direction_angle_mean` 对全部环境平均；零命令或零速度时公式给约 90°。对应 bad_frac 已做激活、速度等过滤，两个指标不能按同一总体解释。`world_command_tracking_active` 只是全局训练步数是否到达阈值，并非有效检查环境比例（`terminations.py:199–221`）。
7. **算法诊断中的 0 可能表示未测量。** 详见算法清单的条件与聚合说明。`Learning/*` 最终实际写成 `Loss/Learning/*`，因为 Logger 只豁免 Grad/Update 前缀。

## 记录链路与公共口径

`env.extras["log"]` → `Logger.process_env_step` → `Logger.log` → `WandbSummaryWriter.add_scalar` → `wandb.log`。

- 横轴 step 为训练 iteration；每个训练 iteration 记录一次聚合值。仅 rank 0 写日志。
- 当前 mjlab 每步创建新的 log dict，未发现沿用同一字典导致整个 rollout 只保留最后一步的问题。
- 环境 key 含 `/` 时原样写入；不含时加 `Episode/`。算法 key 除 Grad/、Update/ 外均加 `Loss/`。
- `Episode_Reward/<term>` 是带权、按 reward manager dt 配置缩放的累计奖励，除以最大 episode 秒数（当前 20 秒），再按 reset 批次聚合；不是原始 reward 函数输出，也不是按实际存活秒数平均。
- `Episode_Metrics/*` 当前均采用 episode 内步均值，再对 reset 环境求均值，最后 Logger 对批次求均值。action 指标基于原始动作差分，未除 dt，因此不是物理加速度。
- 多 GPU 时环境统计和大多数 loss 来自 rank 0，不能当作所有 GPU 的全局均值；FPS 的采样数乘 world size。
- reward manager 对 reward 值做 nan_to_num；直接写 extras 的诊断值及 Logger 没有统一有限值过滤。

## 通用训练指标

| W&B key | 含义 / 条件 |
|---|---|
| Loss/learning_rate | 主 PPO optimizer 学习率 |
| Policy/mean_std | 策略动作标准差均值 |
| Perf/total_fps | 总环境采样步数 / (collection_time + learning_time) |
| Perf/collection_time | 本 iteration rollout 耗时，秒 |
| Perf/learning_time | 本 iteration 更新耗时，秒 |
| Train/mean_reward | 最近最多 100 个已完成 episode 的累计 reward 均值 |
| Train/mean_episode_length | 同一窗口的 episode 步数均值，不是秒 |
| Rnd/mean_extrinsic_reward | 开启 RND 且有已完成 episode 时，外在回报均值 |
| Rnd/mean_intrinsic_reward | 同上，内在回报均值 |
| Rnd/weight | 同上，RND 权重 |

`Train/*/time` 在 W&B 分支不记录。`video` 是视频媒体；config、模型、ONNX 与 git diff 是配置/文件，不是标量 metrics。W&B 自带 `_step`、`_timestamp`、`_runtime`、`_wandb` 等元数据。

## 当前任务 Episode_Reward 清单

下表省略统一前缀 `Episode_Reward/`。零权重 reward 不执行函数，因此该函数附带的 Metrics 日志也不会产生。

| 后缀 | flat 权重 | rough 权重 | 函数 |
|---|---:|---:|---|
| alive | 0.1 | 0.1 | is_alive_before_step |
| track_linear_velocity | 4.0 | 4.0 | track_linear_velocity |
| track_angular_velocity | 1.0 | 1.0 | track_angular_velocity |
| track_heading | 0.5 | 0.5 | track_heading |
| lin_vel_z_l2 | -0.3 | -0.3 | lin_vel_z_l2 |
| base_ang_vel_xy | -0.15 | -0.15 | base_ang_vel_xy_l2 |
| flat_orientation | -10.0 | -10.0 | flat_orientation_l2 |
| base_height | -50.0 | -50.0 | base_height_l2 |
| pose | 0.5 | 0.5 | variable_posture |
| stand_still | -3.0 | -3.0 | stand_still |
| leg_joint_pos_limits | -5.0 | -5.0 | joint_pos_limits |
| leg_joint_vel | -0.015 | -0.015 | joint_vel_l2 |
| leg_joint_torque | -1e-05 | -1e-05 | joint_torques_l2 |
| wheel_joint_vel | -0.0002 | -0.0002 | joint_vel_l2 |
| leg_joint_acc | -3e-07 | -3e-07 | joint_acc_l2 |
| wheel_joint_acc | -1e-07 | -1e-07 | joint_acc_l2 |
| joint_power | -5e-05 | -5e-05 | joint_power_l1 |
| leg_action_rate | -0.3 | -0.3 | action_term_rate_l2 |
| wheel_action_rate | -0.1 | -0.1 | action_term_rate_l2 |
| leg_action_smoothness | -0.03 | -0.03 | action_term_smoothness_l2 |
| wheel_action_smoothness | -0.01 | -0.01 | action_term_smoothness_l2 |
| self_collisions | -0.1 | -0.1 | self_collision_cost |
| illegal_ground_contact | -1.0 | -1.0 | self_collision_cost |
| soft_landing | -3e-05 | -3e-05 | soft_landing |
| wheel_air_time_balance | -4.0 | -4.0 | wheel_air_time_balance |
| non_rough_wheel_distance | 0.4 | 0.4 | non_rough_wheel_distance |
| non_rough_base_at_midpoint | 0.5 | 0.5 | non_rough_base_at_midpoint |
| rough_wheel_usage | — | -0.015 | rough_wheel_usage |
| rough_wheel_foot_clearance | — | 2.0 | rough_wheel_foot_clearance |
| rough_contact_pattern | — | 0.3 | rough_contact_pattern |
| rough_min_wheel_distance | — | -0.5 | rough_min_wheel_distance |
| non_rough_wheel_lateral_symmetry | — | 0.5 | non_rough_wheel_lateral_symmetry |
| non_rough_wheel_x_alignment | — | -50.0 | non_rough_wheel_x_alignment |
| standing_forward_wheel_air_time | — | -20.0 | standing_forward_wheel_air_time |

## 当前 Episode_Metrics 清单

- `Episode_Metrics/mean_action_acc`：`mjlab.envs.mdp.metrics.mean_action_acc`，reduce=mean。
- `Episode_Metrics/leg_action_rate`：`wheeled_legged_mjlab.tasks.velocity.mdp.rewards.action_term_rate_l2`，reduce=mean。
- `Episode_Metrics/wheel_action_rate`：`wheeled_legged_mjlab.tasks.velocity.mdp.rewards.action_term_rate_l2`，reduce=mean。
- `Episode_Metrics/leg_action_smoothness`：`wheeled_legged_mjlab.tasks.velocity.mdp.rewards.action_term_smoothness_l2`，reduce=mean。
- `Episode_Metrics/wheel_action_smoothness`：`wheeled_legged_mjlab.tasks.velocity.mdp.rewards.action_term_smoothness_l2`，reduce=mean。

`mean_action_acc` 为所有动作维二阶差分绝对值均值；leg/wheel action_rate 是一阶差分平方和；leg/wheel action_smoothness 是二阶差分平方和，并忽略 episode 最初两步。它们未乘 reward weight。

## 当前终止、命令与课程指标

- `Episode_Termination/non_finite_physics`。
- `Episode_Termination/time_out`。
- `Episode_Termination/fell_over`。
- `Episode_Termination/illegal_contact`。
- `Episode_Termination/world_command_tracking_failure`。
- `Episode_Termination/out_of_terrain_bounds`：仅 rough。

- `Metrics/twist/error_vel_xy`：世界系 xy 速度欧氏误差的归一化积分。
- `Metrics/twist/error_vel_yaw`：body z 角速度绝对误差的归一化积分。
- `Curriculum/fell_over_limit_angle/limit_angle`：弧度，默认从 65° 到 85°，120000 policy steps。
- `Curriculum/terrain_levels/mean`、`max`：全部环境地形等级的均值、最大值；仅 rough。
- `Curriculum/terrain_levels/discrete_obstacles`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/flat`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/hf_pyramid_slope`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/hf_pyramid_slope_inv`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/pyramid_stair`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/pyramid_stair_inv`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/random_rough`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/random_spread`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/random_stairs`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/stepping_stones`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。
- `Curriculum/terrain_levels/tilted_grid`：该逻辑地形类型环境的平均等级；存在该类型环境且列映射匹配时。

可选但当前未配置的 `commands_vel` 课程产生 `Curriculum/<term>/{lin_vel_x_min,lin_vel_x_max,lin_vel_y_min,lin_vel_y_max,ang_vel_z_min,ang_vel_z_max}`。play 配置会移除课程和部分终止项。

## 自定义 Metrics 完整静态清单

以下枚举当前源代码所有显式 `Metrics/` 写入点，包括未启用函数；不是承诺每个 run 都存在。表达式保留实际计算口径。统一省略前缀 `Metrics/`。

| 后缀 | 产生函数 | 计算表达式 | 位置 |
|---|---|---|---|
| roughness_gate_threshold | `_scheduled_roughness_gate_threshold` | `current` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:80 |
| roughness_gate_threshold_schedule_progress | `_scheduled_roughness_gate_threshold` | `progress` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:81 |
| angular_momentum_mean | `angular_momentum_penalty` | `torch.mean(angmom_magnitude)` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:493 |
| rough_wheel_distance_mean | `rough_min_wheel_distance` | `(distance_y * rough_active).sum() / rough_count` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:945 |
| rough_wheel_distance_violation_mean | `rough_min_wheel_distance` | `(distance_violation * rough_active).sum() / rough_count` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:948 |
| air_time_mean | `feet_air_time` | `mean_air_time` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1016 |
| standing_forward_wheel_air_time_mean | `standing_forward_wheel_air_time` | `cost.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1164 |
| standing_wheel_air_time_mean | `standing_forward_wheel_air_time` | `standing_cost.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1165 |
| non_rough_forward_wheel_air_time_mean | `standing_forward_wheel_air_time` | `forward_cost.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1166 |
| rough_wheel_foot_clearance_mean | `rough_wheel_foot_clearance` | `wheel_foot_clearance.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1274 |
| rough_wheel_foot_clearance_target_mean | `rough_wheel_foot_clearance` | `target.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1277 |
| rough_double_contact_mean | `rough_contact_pattern` | `double_contact.float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1328 |
| rough_no_contact_mean | `rough_contact_pattern` | `no_contact.float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1329 |
| rough_single_contact_mean | `rough_contact_pattern` | `(contact_count == 1).float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1330 |
| slip_velocity_mean | `feet_slip` | `mean_slip_vel` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1447 |
| landing_force_mean | `soft_landing` | `mean_landing_force` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1468 |
| roughness_mean | `_terrain_roughness_from_sensor` | `stats.foot_roughness.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:216 |
| roughness_max_mean | `_terrain_roughness_from_sensor` | `stats.robot_roughness.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:220 |
| roughness_lambda_mean | `_terrain_roughness_from_sensor` | `stats.gate.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:221 |
| roughness_jump_over_R_mean | `_terrain_roughness_from_sensor` | `(stats.jump / wheel_radius).mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:222 |
| roughness_curvature_over_R_mean | `_terrain_roughness_from_sensor` | `(stats.curvature / wheel_radius).mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:225 |
| wheel_episode_air_time_left_mean | `__call__` | `self._cumulative_air_time[:, 0].mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1079 |
| wheel_episode_air_time_right_mean | `__call__` | `self._cumulative_air_time[:, 1].mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1082 |
| wheel_episode_air_time_diff_mean | `__call__` | `air_time_diff.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1085 |
| wheel_episode_air_time_imbalance_ratio_mean | `__call__` | `imbalance_ratio.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1086 |
| wheel_air_time_balance_cost_mean | `__call__` | `cost.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1089 |
| peak_height_mean | `__call__` | `mean_peak_height` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:1412 |
| roughness_left_mean | `_terrain_roughness_from_sensor` | `stats.foot_roughness[:, 0].mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:218 |
| roughness_right_mean | `_terrain_roughness_from_sensor` | `stats.foot_roughness[:, 1].mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py:219 |
| world_command_tracking_progress_deficit_mean | `__call__` | `progress_deficit.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:199 |
| world_command_tracking_progress_bad_frac | `__call__` | `bad_progress.float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:202 |
| world_command_tracking_direction_angle_mean | `__call__` | `angle.mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:205 |
| world_command_tracking_direction_bad_frac | `__call__` | `bad_direction.float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:206 |
| world_command_tracking_heading_error_deg_mean | `__call__` | `heading_error.mean() * (180.0 / math.pi)` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:209 |
| world_command_tracking_heading_bad_frac | `__call__` | `bad_heading.float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:212 |
| world_command_tracking_airborne_frac | `__call__` | `(~grounded).float().mean()` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:215 |
| world_command_tracking_active | `__call__` | `torch.tensor(float(env.common_step_counter >= activation_step), device=env.device)` | src/wheeled_legged_mjlab/tasks/velocity/mdp/terminations.py:218 |

启用条件与单位补充：

- 当前 flat 的直接 reward 诊断为 wheel_episode_air_time 的 4 项、wheel_air_time_balance_cost_mean、landing_force_mean；两种地形均有 world_command_tracking 的 8 项。
- rough 额外启用 roughness、阈值调度、轮距、clearance、接触模式与 standing/forward air-time 诊断。当前 reward roughness 使用 terrain_scan 的单 frame，所以 roughness_mean 与 roughness_max_mean 相同；left/right 仅在至少两个 frame 的调用中写入，当前默认 reward 路径不产生。
- `angular_momentum_mean`、`air_time_mean`、`peak_height_mean`、`slip_velocity_mean` 所在 reward 未纳入当前 make_rewards，只有另行配置才会记录。
- roughness_* 为无量纲，gate lambda 为 smoothstep 后的 [0,1] 数值；不是 rough 环境比例。
- wheel_episode_air_time_* 的 left/right/diff 为当前活跃 episode 到目前为止的累计秒数，再跨环境/step 平均；不是只对已完成 episode 统计。imbalance_ratio/cost 无量纲。
- standing/forward air_time 指标是含截断、offset、mask 的成本，且对所有环境平均，不是该命令子集的原始平均腾空时长。
- clearance/distance/peak_height 为米；landing_force 为牛；slip_velocity 为米/秒；方向和 heading error 为度。

## 本地历史 summary 核查

读取 76 份 summary，解析失败 0 份；发现 251 个带命名空间的历史指标名（跨版本并集，不等于当前启用指标数）。最近文件：`wandb/run-20260829_231627-xyhb1sir/files/wandb-summary.json`。

- 非有限标量：`Loss/depth_ae_reconstruction = nan`，文件 `wandb/run-20260605_230450-y2a66o8j/files/wandb-summary.json`。这是历史运行证据，不代表当前代码仍有该问题。

完整历史名称及出现 run 数见同目录 `wandb_metrics_history_inventory.csv`。summary 只保留最终值，无法据此断言整条曲线没有 NaN、尖峰或漏点。

## 算法指标完整清单

这里使用最终 W&B 名称。`rsl_rl/rsl_rl/utils/logger.py:182` 仅保留 `Grad/`、`Update/` 前缀，其余算法 key 全部加 `Loss/`，所以算法里的 `Learning/ppo_lr` 实际为 **`Loss/Learning/ppo_lr`**。

### 基础算法

| 算法 | 最终指标 | 含义与条件 |
|---|---|---|
| PPO | `Loss/value`, `Loss/surrogate`, `Loss/entropy` | 价值误差（按配置可能 clipped）、PPO clipped surrogate、正熵。均为 minibatch 均值；不是乘系数后的优化项。 |
| PPO + RND | `Loss/rnd` | 仅启用 RND。 |
| PPO + symmetry | `Loss/symmetry` | 启用 symmetry 时输出，即使未启用 mirror loss 加入目标。 |
| Distillation | `Loss/behavior` | student/teacher 行为匹配损失。 |
| RepresentationTeacherStudentPPO | `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/representation` | PPO 加表示匹配。 |
| RepresentationVelocityTeacherStudentPPO | `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/student`, `Loss/representation`, `Loss/lin_vel`, `Loss/roughness` | student 是 representation、lin_vel、roughness 的加权和；无 roughness head 时 roughness 写 0。 |
| RepresentationVelocityPredictorTeacherStudentPPO | 同上一行七项，另加下文所有符合条件的指标 | PPO loss 平均所有 PPO minibatch；student loss 平均实际 student 更新。 |

来源：`rsl_rl/rsl_rl/algorithms/ppo.py:331`、`distillation.py:170`、`representation_teacher_student_ppo.py:213`、`representation_velocity_teacher_student_ppo.py:309`、`representation_velocity_predictor_teacher_student_ppo.py:588`。

### Predictor 固定输出

以下来源均为 `rsl_rl/rsl_rl/algorithms/representation_velocity_predictor_teacher_student_ppo.py`。

| 最终名称 | 口径 |
|---|---|
| `Grad/ppo_total_norm` | 纯 PPO 梯度 L2 norm，平均所有 PPO 步。 |
| `Grad/privileged_encoder_ppo_norm` | privileged encoder 纯 PPO 梯度 L2 norm，平均所有 PPO 步。 |
| `Grad/ppo_joint_total_norm` | 加入 dynamics 后的 PPO 参数组裁剪前 L2 norm，平均所有 PPO 步（包括纯 PPO 步）。 |
| `Grad/ppo_joint_clip_fraction` | PPO 参数组触发裁剪的更新步占全部 PPO 更新比例。 |
| `Grad/predictor_total_norm` | predictor 裁剪前 L2 norm，仅平均 dynamics 更新步。 |
| `Grad/predictor_clip_fraction` | predictor 触发裁剪占 dynamics 更新比例。 |
| `Loss/Learning/ppo_lr` | PPO optimizer 当前第一参数组学习率。 |
| `Loss/Learning/predictor_lr` | predictor optimizer 当前第一参数组学习率。 |
| `Update/privileged_encoder_norm_joint` | joint step 前后 privileged encoder 参数 L2 差的均值。 |
| `Update/privileged_encoder_norm_ppo_only` | 被采样 PPO-only step 前后 privileged encoder 参数 L2 差的均值。 |
| `Update/policy_kl_joint` | joint 单步 optimizer 更新前后策略 KL。 |
| `Update/policy_kl_ppo_only` | 被采样 PPO-only 单步 optimizer 更新前后策略 KL。 |
| `Update/joint_step_fraction` | joint step 数 / 全部 PPO 更新数。 |

记录位置 596–610；梯度计算在 439–478；更新诊断采样在 309–311，policy KL 在 1169–1194。所有 Grad/* norm 都是裁剪前指标；Update/* norm 则是优化器实际更新后的参数差。这里的单步 KL 不等同于 rollout-old policy 到当前 policy 的 adaptive PPO KL。

### 开启 latent dynamics 时

条件：`latent_dynamics_loss_coef > 0`。记录位置 617–705。

固定梯度指标：

```text
Grad/dynamics_total_norm
Grad/privileged_encoder_dynamics_norm
Grad/privileged_encoder_dynamics_to_ppo_ratio
Grad/privileged_encoder_ppo_dynamics_cosine
```

前两项是 dynamics 反传增加的梯度 L2 norm；cosine 比较 privileged encoder 的 PPO 与 dynamics 梯度方向。均平均实际 dynamics 更新。ratio 为 dynamics encoder 平均 norm 除以 PPO encoder 平均 norm。

以下九项同时输出汇总名称及 `_k{horizon}` 版本，horizon 遍历配置 `latent_dynamics_horizons`：

```text
Loss/latent_dynamics_loss
Loss/latent_dynamics_representation_loss
Loss/latent_dynamics_velocity_loss
Loss/latent_identity_loss
Loss/latent_prediction_identity_ratio
Loss/latent_shuffled_action_loss
Loss/latent_shuffled_action_ratio
Loss/latent_prediction_cosine_similarity
Loss/latent_dynamics_valid_fraction

Loss/latent_dynamics_loss_k{horizon}
Loss/latent_dynamics_representation_loss_k{horizon}
Loss/latent_dynamics_velocity_loss_k{horizon}
Loss/latent_identity_loss_k{horizon}
Loss/latent_prediction_identity_ratio_k{horizon}
Loss/latent_shuffled_action_loss_k{horizon}
Loss/latent_shuffled_action_ratio_k{horizon}
Loss/latent_prediction_cosine_similarity_k{horizon}
Loss/latent_dynamics_valid_fraction_k{horizon}
```

当 horizon > 1 时还有：

```text
Loss/latent_reversed_action_loss_k{horizon}
Loss/latent_reversed_action_ratio_k{horizon}
```

计算位置 794 起。representation 为 `1 - cosine_similarity`；velocity 为归一化速度 smooth-L1；dynamics 为 representation + velocity_coef × velocity。identity 是保持当前状态的 baseline；shuffled 是打乱不同样本的动作块；reversed 是反转动作时间顺序。prediction/identity ratio 越低表示越优于恒等 baseline；shuffled/prediction 和 reversed/prediction ratio 大于 1 表示正确动作优于扰动动作。每个 horizon 按实际样本数加权，再按配置 horizon 权重汇总。valid_fraction 是有效训练样本占比。

### 同时开启 latent rollout 时

条件：latent dynamics 已开启，且 `latent_rollout_loss_coef > 0`。位置 718–750。

```text
Loss/latent_rollout_loss
Loss/latent_rollout_representation_loss
Loss/latent_rollout_velocity_loss
Loss/latent_rollout_valid_fraction
```

step 遍历 1…`latent_rollout_horizon`：

```text
Loss/latent_rollout_loss_k{step}
Loss/latent_rollout_representation_loss_k{step}
Loss/latent_rollout_velocity_loss_k{step}
Loss/latent_rollout_identity_ratio_k{step}
Loss/latent_rollout_shuffled_action_loss_k{step}
Loss/latent_rollout_shuffled_action_ratio_k{step}
Loss/latent_rollout_cosine_similarity_k{step}
```

这些是一步 predictor 自回归展开到第 step 步的误差与 baseline 比较。representation/velocity 的定义同上；汇总 loss 为各 rollout step 的算术平均。

当最终 rollout horizon H 也属于 `latent_dynamics_horizons` 时，额外输出直接 H 步预测与递归 H 步预测的比较：

```text
Loss/latent_direct_rollout_cosine_k{H}
Loss/latent_direct_rollout_mse_k{H}
Loss/latent_direct_rollout_velocity_loss_k{H}
```

### 检查发现与解释限制

1. **没有测量时仍写 0。** Predictor 604–610 使用 `max(count, 1)`；dynamics disabled 时连 `Update/*ppo_only` 都不采样，但记录为 0。全部 PPO 步都是 joint 时，ppo_only 同样为 0。没有有效样本的 horizon 也写 0（626–705）。这不能解释为 KL/误差完美或没有参数更新。
2. **PPO-only 诊断是子集。** 309–311 仅选择 joint 步前的相邻 PPO-only 步，不是全部 PPO-only 更新的均值。
3. **梯度 ratio 的平均范围不同。** 620–622 的分子平均 dynamics 步，分母平均全部 PPO 步，不是同批次梯度比。
4. **DDP loss 未跨 rank 汇总。** 梯度和 KL 有归约，但 `.item()` 累加的 loss 没有 all_reduce，例如 predictor 505–507、588 起及基础 PPO 309–328。若仅 rank 0 输出，它代表 rank 0 的 loss。
5. **零 student 更新存在除零边界。** Predictor 592–595、velocity PPO 313–316 直接除 `student_updates`；如果允许 substeps/representation epochs 为 0，会报错。未确认当前运行配置是否触发。
6. **loss 与 gradient 的权重口径不同。** dynamics loss 日志没有乘总 `latent_dynamics_loss_coef`；dynamics Grad 来自加权目标反传，不能将二者数值直接对照。
7. **缺少部分常用诊断。** 未输出总 joint objective、student_lr、student grad norm、标准 PPO clipping fraction 和 rollout-old 到当前策略 KL。这是监控缺项，不是已证实的计算错误。
