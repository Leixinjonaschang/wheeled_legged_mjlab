# wheel_legged_mjlab

## Installation 

```shell
uv sync
```

## Usage

### Training

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Flat-WF-Tron1B  
```

<details>
<summary>Training CLI options</summary>

Training entry:

```shell
uv run python scripts/rsl_rl/train.py <TASK> [OPTIONS]
```

Available WF-Tron1B tasks:

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Flat-WF-Tron1B
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B
```

Show all task-specific options:

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B --help
```

Top-level options:

```shell
--video True|False
--video-length INT
--video-interval INT
--enable-nan-guard True|False
--gpu-ids [0]
--gpu-ids [0,1]
--gpu-ids all
--torchrunx-log-dir STR
--registry-name STR
--wandb-run-path STR
--wandb-checkpoint-name STR
```

Common agent options:

```shell
--agent.seed 42
--agent.max-iterations 30000
--agent.num-steps-per-env 24
--agent.save-interval 50
--agent.experiment-name wf_tron1b_velocity
--agent.run-name test_run
--agent.trial-message "description of this training run"
--agent.logger wandb
--agent.logger tensorboard
--agent.wandb-project mjlab
--agent.wandb-tags tag1 tag2
--agent.resume True
--agent.load-run ".*"
--agent.load-checkpoint "model_.*.pt"
--agent.clip-actions 1.0
--agent.upload-model True
```

Policy, value function, and PPO options:

```shell
--agent.actor.hidden-dims 512 256 128
--agent.actor.activation elu
--agent.actor.obs-normalization True
--agent.critic.hidden-dims 512 256 128
--agent.critic.activation elu
--agent.algorithm.learning-rate 0.001
--agent.algorithm.num-learning-epochs 5
--agent.algorithm.num-mini-batches 4
--agent.algorithm.gamma 0.99
--agent.algorithm.lam 0.95
--agent.algorithm.entropy-coef 0.01
--agent.algorithm.desired-kl 0.01
--agent.algorithm.max-grad-norm 1.0
--agent.algorithm.clip-param 0.2
--agent.algorithm.schedule adaptive
--agent.algorithm.optimizer adam
```

Common environment options:

```shell
--env.seed 0
--env.decimation 4
--env.episode-length-s 20.0
--env.scene.num-envs 2048
--env.scene.env-spacing 2.0
--env.scene.terrain.terrain-type generator
--env.scene.terrain.terrain-type plane
```

Example debug run:

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B \
  --gpu-ids [0] \
  --agent.run-name debug \
  --agent.trial-message "lateral command mode test" \
  --agent.max-iterations 1000 \
  --agent.save-interval 100 \
  --env.scene.num-envs 512 \
  --video True
```

Resume from a local checkpoint:

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B \
  --agent.resume True \
  --agent.load-run "2026-05-18_.*" \
  --agent.load-checkpoint "model_1000.pt"
```

Resume from W&B:

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B \
  --agent.resume True \
  --wandb-run-path entity/project/run_id \
  --wandb-checkpoint-name model_1000.pt
```

The training script uses `tyro`, so task config fields can be overridden with
deep CLI paths such as `--env.xxx.yyy` and `--agent.xxx.yyy`.

</details>

### Policy Evaluation

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Flat-WF-Tron1B \
  --checkpoint-file logs/rsl_rl/wf_tron1b_velocity/<RUN_DIR>/model_1000.pt
```

<details>
<summary>Policy evaluation CLI options</summary>

Policy evaluation entry:

```shell
uv run python scripts/rsl_rl/play.py <TASK> [OPTIONS]
```

Available WF-Tron1B tasks:

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Flat-WF-Tron1B
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Rough-WF-Tron1B
```

Show all task-specific options:

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Rough-WF-Tron1B --help
```

Common evaluation options:

```shell
--agent trained
--agent zero
--agent random
--checkpoint-file logs/rsl_rl/wf_tron1b_velocity/<RUN_DIR>/model_1000.pt
--wandb-run-path entity/project/run_id
--wandb-checkpoint-name model_1000.pt
--num-envs 1
--device cuda:0
--device cpu
--viewer auto
--viewer native
--viewer viser
--no-terminations True
```

Video options:

```shell
--video True
--video-length 200
--video-height 720
--video-width 1280
--camera 0
```

Evaluate a local checkpoint:

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Rough-WF-Tron1B \
  --checkpoint-file logs/rsl_rl/wf_tron1b_velocity/<RUN_DIR>/model_1000.pt \
  --num-envs 1 \
  --device cuda:0 \
  --viewer auto
```

Evaluate a W&B checkpoint:

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Rough-WF-Tron1B \
  --wandb-run-path entity/project/run_id \
  --wandb-checkpoint-name model_1000.pt \
  --num-envs 1
```

Record a rollout video:

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Rough-WF-Tron1B \
  --checkpoint-file logs/rsl_rl/wf_tron1b_velocity/<RUN_DIR>/model_1000.pt \
  --video True \
  --video-length 400 \
  --video-width 1280 \
  --video-height 720
```

Run a dummy policy for environment inspection:

```shell
uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Rough-WF-Tron1B \
  --agent zero \
  --num-envs 1 \
  --viewer auto \
  --no-terminations True
```

For trained policy evaluation, provide either `--checkpoint-file` for a local
checkpoint or `--wandb-run-path` for a W&B run. If `--wandb-checkpoint-name` is
omitted, the script resolves the checkpoint through the run path helper. With
`--viewer auto`, the script uses the native MuJoCo viewer when a display is
available and falls back to `viser` otherwise.

</details>
