# Depth student asymmetric actor–critic

The predictor tasks train the deployable depth student directly:

- `Mjlab-Velocity-Rough-WF-Tron1B-Depth-Predict`
- `Mjlab-Velocity-Rough-WF-Tron1B-Depth-Predict-NoRough`

They do not construct a teacher actor, privileged encoder, representation
alignment loss, or EMA target encoder. Other representation teacher–student
tasks are unchanged.

## Architecture

Training uses three independent modules:

1. `DepthActor` consumes five-frame proprioceptive history, the current command,
   and delayed/noisy metric depth. Its depth CNN emits 64 features, its GRU has
   128 hidden units, and its student encoder produces a normalized 64-dimensional
   latent, a linear-velocity estimate, and two wheel-roughness estimates. The
   actor head consumes the latent, estimated velocity, current proprioception,
   and command.
2. `MLPModel` is an independent critic. It consumes only `critic` and
   `dynamics_context`, with hidden widths `(512, 256, 256, 128)`.
3. `LatentDynamicsPredictor` is training-only. One network is used for each
   direct horizon `(1, 5, 10)`.

Policy and entropy gradients update the complete actor. Value gradients update
only the critic. Velocity and roughness supervision update the student CNN, GRU,
encoder, and their corresponding heads. Dynamics gradients update the CNN, GRU,
student encoder, latent head, and predictor, but not the actor head, critic,
velocity head, or roughness head.

## Dynamics objective

At time `t`, the predictor state is the student latent `z_t` concatenated with
normalized ground-truth linear velocity `v_t`. Ordered actions are the actions
actually applied by the environment, not necessarily the raw sampled actions.
For direct horizon `k`:

```text
input  = [z_t, v_t, applied_actions[t:t+k]]
target = [stop_gradient(z_{t+k}), v_{t+k}]
```

The source latent remains attached to the student encoder. Future targets are
recomputed from the same current student over the stored observation sequence
and detached. Direct horizons `(1, 5, 10)` use weights `(1.0, 0.75, 0.5)`.
The five-step autoregressive rollout repeatedly applies the one-step predictor;
intermediate predictions remain attached and are never replaced by true states.

Latent error is `1 - cosine_similarity`; velocity error is Smooth L1 with
coefficient `1.0`. The dynamics objective is:

```text
3.0 * (weighted_direct_mean + 0.75 * rollout_mean)
```

Student linear-velocity estimation uses MSE with coefficient `1.0`. Wheel
roughness uses Smooth L1 with coefficient `0.2`; `NoRough` changes only this
coefficient to zero.

## Sequence replay and optimization

Each rollout stores observations, sampled actions, applied actions, old action
distribution statistics, values, done flags, and actor hidden state. A sequence
mini-batch is the complete time axis (`24` steps by default) for a shuffled group
of environments. Time is never shuffled. Replay begins from the stored, detached
initial GRU state and zeros an environment's state after its done transition.
Predictions that cross an episode boundary or extend past the rollout are
masked; those transitions still participate in PPO.

Observation normalization is initialized once, frozen during rollout and every
replay update, then updated with that rollout after optimization.

The main Adam owns all actor and critic parameters. The predictor Adam owns only
the dynamics predictor; there is no student optimizer. Both start at `1e-3` and
clip gradients at `1.0`. Adaptive KL changes only the main learning rate.

Each iteration performs `5 × 4 = 20` PPO steps. An independent `1 × 4` sequence
pass supplies auxiliary batches at PPO steps 5, 10, 15, and 20. Shared actor
gradients are accumulated before the single main optimizer step; the predictor
steps only when a globally valid prediction sample exists. Distributed losses
are weighted by global valid-sample count, including ranks with zero local valid
samples.

## Training, resume, and deployment

Start from random initialization:

```shell
uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Rough-WF-Tron1B-Depth-Predict
```

Use the usual `--agent.resume`, `--agent.load-run`, and
`--agent.load-checkpoint` options to resume a checkpoint created by this
architecture. Checkpoints contain actor, critic, predictor, both optimizers,
normalization state, learning rates, and the runner iteration. Legacy
teacher–student predictor checkpoints are intentionally not migrated.

Evaluation returns only the actor. It can act with `proprio_history`,
`actor_command`, and `depth_camera`; privileged critic observations and training
labels are not part of actor inference. TorchScript and ONNX exclude the critic,
dynamics predictor, and roughness head. The ONNX interface is:

```text
inputs:  proprio_history, actor_command, depth, hidden_state_in
outputs: actions, predicted_lin_vel, hidden_state_out
```

The speed-preview path reads the current GRU state without advancing it.

## Diagnostics

Logs include direct losses for every horizon, rollout losses for every step,
valid-sample fractions, velocity and roughness losses, identity/shuffled/reversed
action controls, latent variance, separate main/predictor gradient norms, and the
student-backbone PPO-versus-dynamics gradient norm and cosine.

Structural, gradient-routing, replay, optimizer-count, empty-mask, checkpoint,
and export behavior are covered by the bundled CPU regression tests. Long-term
convergence and task performance still require training experiments; CPU tests
do not establish those outcomes.
