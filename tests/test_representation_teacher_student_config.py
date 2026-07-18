"""Smoke tests for WF-TRON1B representation teacher-student configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks

import wheeled_legged_mjlab  # noqa: F401
from rsl_rl.models import RepresentationActorCritic, RepresentationVelocityActorCritic
from wheeled_legged_mjlab.rl.runner import get_wheeled_legged_metadata
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    DEPTH_BUFFER_SIZE,
    DEPTH_BUFFER_UPDATE_PERIOD,
    DEPTH_CAPTURE_FREQUENCY_HZ,
    DEPTH_CAMERA_NAME,
    wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg,
    wf_tron1b_rough_rep_ts_lin_vel_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp import observations as observation_mdp


PLAY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "play.py"
PLAY_SPEC = importlib.util.spec_from_file_location("rsl_rl_play_script", PLAY_PATH)
assert PLAY_SPEC is not None
assert PLAY_SPEC.loader is not None
play = importlib.util.module_from_spec(PLAY_SPEC)
PLAY_SPEC.loader.exec_module(play)


def test_representation_teacher_student_tasks_are_registered() -> None:
    tasks = set(list_tasks())

    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS" in tasks
    assert "Mjlab-Velocity-Flat-WF-Tron1B-RepTS" in tasks

    rough_agent = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS"))
    flat_env = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B-RepTS")

    assert rough_agent["algorithm"]["class_name"] == "RepresentationTeacherStudentPPO"
    assert rough_agent["actor"]["class_name"] == "RepresentationActorCritic"
    assert rough_agent["obs_groups"] == {
        "teacher_actor": ("actor",),
        "critic": ("critic",),
        "student_history": ("actor_history",),
        "privileged_encoder": ("critic",),
    }
    assert "actor_history" in flat_env.observations


def test_representation_velocity_tasks_are_registered() -> None:
    tasks = set(list_tasks())

    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel" in tasks
    assert "Mjlab-Velocity-Flat-WF-Tron1B-RepTS-LinVel" in tasks

    rough_agent = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel"))
    flat_env = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B-RepTS-LinVel")

    assert rough_agent["algorithm"]["class_name"] == "RepresentationVelocityTeacherStudentPPO"
    assert rough_agent["algorithm"]["representation_loss_coef"] == 1.0
    assert rough_agent["algorithm"]["lin_vel_loss_coef"] == 1.0
    assert all("latent_dynamics" not in name for name in rough_agent["algorithm"])
    assert all("latent_dynamics" not in name for name in rough_agent["actor"])
    assert rough_agent["actor"]["class_name"] == "RepresentationVelocityActorCritic"
    assert rough_agent["obs_groups"] == {
        "proprio_history": ("proprio_history",),
        "actor_command": ("actor_command",),
        "lin_vel_target": ("lin_vel_target",),
        "critic": ("critic",),
        "privileged_encoder": ("privileged_encoder",),
    }
    assert "proprio_history" in flat_env.observations
    assert "actor_history" not in flat_env.observations
    assert "latent_dynamics_command_generation" not in flat_env.observations


def test_actor_history_and_rough_privileged_observations() -> None:
    cfg = wf_tron1b_rough_env_cfg()

    assert cfg.observations["actor_history"].history_length == 5
    assert cfg.observations["actor_history"].flatten_history_dim is False
    assert cfg.observations["actor_history"].enable_corruption is True
    assert cfg.observations["actor"].enable_corruption is False
    assert cfg.observations["critic"].enable_corruption is False

    play_cfg = wf_tron1b_rough_env_cfg(play=True)
    assert play_cfg.observations["actor_history"].enable_corruption is False

    actor_terms = cfg.observations["actor"].terms
    actor_history_terms = cfg.observations["actor_history"].terms
    critic_terms = cfg.observations["critic"].terms

    assert "height_scan" not in actor_terms
    assert "height_scan" not in actor_history_terms
    assert "height_scan" in critic_terms
    assert "domain_randomization_delta_quantity" not in actor_terms
    assert "domain_randomization_delta_quantity" not in actor_history_terms


def test_representation_velocity_observation_groups() -> None:
    cfg = wf_tron1b_rough_rep_ts_lin_vel_env_cfg()

    assert cfg.observations["proprio_history"].history_length == 5
    assert cfg.observations["proprio_history"].flatten_history_dim is False
    assert cfg.observations["proprio_history"].enable_corruption is True
    assert cfg.observations["actor_command"].enable_corruption is False
    assert cfg.observations["lin_vel_target"].enable_corruption is False
    assert cfg.observations["critic"].enable_corruption is False
    assert cfg.observations["privileged_encoder"].enable_corruption is False

    play_cfg = wf_tron1b_rough_rep_ts_lin_vel_env_cfg(play=True)
    assert play_cfg.observations["proprio_history"].enable_corruption is False

    proprio_terms = cfg.observations["proprio_history"].terms
    actor_command_terms = cfg.observations["actor_command"].terms
    lin_vel_target_terms = cfg.observations["lin_vel_target"].terms
    critic_terms = cfg.observations["critic"].terms
    privileged_terms = cfg.observations["privileged_encoder"].terms

    assert list(actor_command_terms) == ["command"]
    assert actor_command_terms["command"].func is mdp.generated_commands
    assert list(lin_vel_target_terms) == ["base_lin_vel"]
    assert lin_vel_target_terms["base_lin_vel"].func is mdp.base_lin_vel
    assert "command" not in proprio_terms
    assert set(proprio_terms) == {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "wheel_vel",
        "actions",
    }
    assert "base_lin_vel" in critic_terms
    assert "command" in critic_terms
    assert "base_lin_vel" not in privileged_terms
    assert "command" not in privileged_terms
    assert "height_scan" in critic_terms
    assert "height_scan" in privileged_terms


def test_depth_task_constructs_depth_buffer_without_training_input() -> None:
    cfg = load_env_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-Depth")
    agent = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-Depth"))

    depth_group = cfg.observations[DEPTH_CAMERA_NAME]
    depth_term = depth_group.terms[DEPTH_CAMERA_NAME]

    assert depth_term.func is mdp.depth_buffer
    assert depth_term.params == {
        "sensor_name": DEPTH_CAMERA_NAME,
        "buffer_size": DEPTH_BUFFER_SIZE,
        "update_period": DEPTH_BUFFER_UPDATE_PERIOD,
    }
    assert depth_group.enable_corruption is False
    assert agent["obs_groups"] == {
        "teacher_actor": ("actor",),
        "critic": ("critic",),
        "student_history": ("actor_history",),
        "privileged_encoder": ("critic",),
    }
    training_obs_groups = {
        group for groups in agent["obs_groups"].values() for group in groups
    }
    assert DEPTH_CAMERA_NAME not in training_obs_groups


def test_depth_velocity_representation_task_uses_async_depth_input() -> None:
    tasks = set(list_tasks())

    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth" in tasks
    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict" in tasks

    cfg = wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg()
    agent = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth"))
    predict_agent = asdict(
        load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict")
    )
    depth_group = cfg.observations[DEPTH_CAMERA_NAME]
    depth_term = depth_group.terms[DEPTH_CAMERA_NAME]

    assert depth_term.func is mdp.async_depth_buffer
    assert depth_term.params == {
        "sensor_name": DEPTH_CAMERA_NAME,
        "capture_frequency_hz": DEPTH_CAPTURE_FREQUENCY_HZ,
    }
    assert agent["actor"]["class_name"] == "DepthRepresentationVelocityActorCritic"
    assert agent["algorithm"]["representation_chunk_length"] == 12
    assert all("latent_dynamics" not in name for name in agent["actor"])
    assert all("latent_dynamics" not in name for name in agent["algorithm"])
    assert all("latent_rollout" not in name for name in agent["algorithm"])

    assert predict_agent["actor"]["class_name"].endswith(
        ":DepthRepresentationVelocityPredictorActorCritic"
    )
    assert predict_agent["actor"]["latent_dynamics_hidden_dims"] == (128, 256, 256, 128)
    assert predict_agent["actor"]["latent_dynamics_horizons"] == (1, 5)
    assert predict_agent["algorithm"]["class_name"].endswith(
        ":RepresentationVelocityPredictorTeacherStudentPPO"
    )
    assert predict_agent["algorithm"]["latent_dynamics_loss_coef"] == 50.0
    assert predict_agent["algorithm"]["latent_dynamics_horizons"] == (1, 5)
    assert predict_agent["algorithm"]["latent_dynamics_horizon_weights"] == (1.0, 0.5)
    assert predict_agent["algorithm"]["latent_dynamics_detach_source"] is False
    assert predict_agent["algorithm"]["latent_rollout_horizon"] == 5
    assert predict_agent["algorithm"]["latent_rollout_loss_coef"] == 0.75
    assert predict_agent["algorithm"]["num_latent_dynamics_epochs"] == 1
    assert predict_agent["algorithm"]["num_latent_dynamics_mini_batches"] == 4
    assert "latent_dynamics_command_generation" not in cfg.observations
    assert agent["obs_groups"] == {
        "proprio_history": ("proprio_history",),
        "actor_command": ("actor_command",),
        "lin_vel_target": ("lin_vel_target",),
        "critic": ("critic",),
        "privileged_encoder": ("privileged_encoder",),
        "depth_encoder": (DEPTH_CAMERA_NAME,),
    }


def test_plain_depth_task_loads_without_importing_predictor_modules() -> None:
    script = """
import builtins
from dataclasses import asdict

real_import = builtins.__import__

def reject_predictor(name, *args, **kwargs):
    if "predictor" in name:
        raise ImportError(f"blocked predictor import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_predictor

import wheeled_legged_mjlab
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.utils import resolve_callable

cfg = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth"))
assert cfg["actor"]["class_name"] == "DepthRepresentationVelocityActorCritic"
assert cfg["algorithm"]["class_name"] == "RepresentationVelocityTeacherStudentPPO"
assert not any("latent_dynamics" in key for key in cfg["actor"])
assert not any("latent_dynamics" in key for key in cfg["algorithm"])
assert resolve_callable(cfg["actor"]["class_name"]).__module__.endswith(
    "depth_representation_velocity_actor_critic"
)
assert resolve_callable(cfg["algorithm"]["class_name"]).__module__.endswith(
    "representation_velocity_teacher_student_ppo"
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_depth_buffer_updates_every_five_policy_steps(monkeypatch) -> None:
    env = SimpleNamespace(common_step_counter=0, frame=torch.ones(2, 2, 3))
    term = observation_mdp.depth_buffer(cfg=None, env=env)
    depth_calls = 0

    def get_depth(env, sensor_name):
        nonlocal depth_calls
        depth_calls += 1
        return env.frame

    monkeypatch.setattr(
        observation_mdp,
        "depth_image",
        get_depth,
    )

    obs = term(env, buffer_size=5, update_period=5)
    assert obs.shape == (2, 5, 2, 3)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 4
    env.frame = torch.full((2, 2, 3), 2.0)
    obs = term(env, buffer_size=5, update_period=5)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 5
    obs = term(env, buffer_size=5, update_period=5)
    assert torch.all(obs[:, :4] == 1.0)
    assert torch.all(obs[:, 4] == 2.0)
    assert depth_calls == 2

    env.common_step_counter = 6
    env.frame = torch.stack((torch.full((2, 3), 3.0), torch.full((2, 3), 4.0)))
    term.reset(torch.tensor([1]))
    obs = term(env, buffer_size=5, update_period=5)
    assert torch.all(obs[0, :4] == 1.0)
    assert torch.all(obs[0, 4] == 2.0)
    assert torch.all(obs[1] == 4.0)
    assert depth_calls == 3


def test_async_depth_buffer_updates_on_capture_clock(monkeypatch) -> None:
    env = SimpleNamespace(common_step_counter=0, step_dt=0.02, frame=torch.ones(2, 2, 3))
    term = observation_mdp.async_depth_buffer(cfg=None, env=env)
    depth_calls = 0

    def get_depth(env, sensor_name):
        nonlocal depth_calls
        depth_calls += 1
        return env.frame

    monkeypatch.setattr(
        observation_mdp,
        "depth_image",
        get_depth,
    )

    obs = term(env, capture_frequency_hz=25.0)
    assert obs.shape == (2, 1, 2, 3)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 1
    env.frame = torch.full((2, 2, 3), 2.0)
    obs = term(env, capture_frequency_hz=25.0)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 2
    obs = term(env, capture_frequency_hz=25.0)
    assert torch.all(obs == 2.0)
    assert depth_calls == 2

    env.common_step_counter = 3
    env.frame = torch.stack((torch.full((2, 3), 3.0), torch.full((2, 3), 4.0)))
    term.reset(torch.tensor([1]))
    obs = term(env, capture_frequency_hz=25.0)
    assert torch.all(obs[0] == 2.0)
    assert torch.all(obs[1] == 4.0)
    assert depth_calls == 3


def test_foot_contact_forces_are_rotated_to_body_frame() -> None:
    yaw_90_quat_w = torch.tensor(
        [[math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]]
    )
    contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            force=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=yaw_90_quat_w,
        )
    )
    env = SimpleNamespace(
        scene={"wheels_ground_contact": contact_sensor, "robot": robot}
    )

    obs = observation_mdp.foot_contact_forces(env, "wheels_ground_contact")

    expected = torch.tensor(
        [[0.0, -math.log1p(1.0), 0.0, math.log1p(2.0), 0.0, 0.0]]
    )
    assert torch.allclose(obs, expected, atol=1.0e-6)


def _make_dummy_metadata_env():
    action_term = SimpleNamespace(
        scale=1.0,
        action_dim=2,
        target_names=["joint_a", "joint_b"],
    )
    return SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                joint_names=["joint_a", "joint_b"],
                spec=SimpleNamespace(
                    actuators=[
                        SimpleNamespace(target="actuator/joint_a", id=0),
                        SimpleNamespace(target="actuator/joint_b", id=1),
                    ]
                ),
                data=SimpleNamespace(default_joint_pos=torch.tensor([[0.1, 0.2]])),
            )
        },
        sim=SimpleNamespace(
            mj_model=SimpleNamespace(
                actuator_gainprm=torch.tensor([[10.0], [20.0]]),
                actuator_biasprm=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]),
            )
        ),
        action_manager=SimpleNamespace(
            active_terms=["actions"],
            get_term=lambda name: action_term,
        ),
        command_manager=SimpleNamespace(active_terms=["twist"]),
        observation_manager=SimpleNamespace(
            active_terms={
                "actor": ["base_ang_vel", "projected_gravity"],
                "actor_history": ["base_ang_vel", "projected_gravity"],
            }
        ),
        cfg=SimpleNamespace(
            observations={
                "actor_history": SimpleNamespace(history_length=5, flatten_history_dim=False),
            }
        ),
    )


def _make_representation_policy() -> RepresentationActorCritic:
    obs = TensorDict(
        {
            "actor": torch.randn(2, 3),
            "actor_history": torch.randn(2, 5, 3),
            "critic": torch.randn(2, 4),
        },
        batch_size=[2],
    )
    return RepresentationActorCritic(
        obs,
        {
            "teacher_actor": ["actor"],
            "critic": ["critic"],
            "student_history": ["actor_history"],
            "privileged_encoder": ["critic"],
        },
        output_dim=2,
        hidden_dims=[8],
        encoder_hidden_dims=[8],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )


def _make_velocity_metadata_env():
    env = _make_dummy_metadata_env()
    env.observation_manager.active_terms = {
        "proprio_history": ["base_ang_vel", "projected_gravity"],
        "actor_command": ["command"],
    }
    env.cfg.observations = {
        "proprio_history": SimpleNamespace(history_length=5, flatten_history_dim=False),
    }
    return env


def _make_velocity_representation_policy() -> RepresentationVelocityActorCritic:
    obs = TensorDict(
        {
            "proprio_history": torch.randn(2, 5, 3),
            "actor_command": torch.randn(2, 3),
            "lin_vel_target": torch.randn(2, 3),
            "critic": torch.randn(2, 5),
            "privileged_encoder": torch.randn(2, 4),
        },
        batch_size=[2],
    )
    return RepresentationVelocityActorCritic(
        obs,
        {
            "proprio_history": ["proprio_history"],
            "actor_command": ["actor_command"],
            "lin_vel_target": ["lin_vel_target"],
            "critic": ["critic"],
            "privileged_encoder": ["privileged_encoder"],
        },
        output_dim=2,
        hidden_dims=[8],
        encoder_hidden_dims=[8],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )


def test_representation_metadata_describes_single_history_input() -> None:
    metadata = get_wheeled_legged_metadata(_make_dummy_metadata_env(), "local", _make_representation_policy())

    assert metadata["observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["policy_input_names"] == ["student_history"]
    assert metadata["student_observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["student_history_length"] == "5"
    assert metadata["student_history_flatten_dim"] == "false"
    assert metadata["student_history_order"] == "oldest_to_newest"


def test_velocity_representation_metadata_describes_history_and_command_inputs() -> None:
    metadata = get_wheeled_legged_metadata(
        _make_velocity_metadata_env(),
        "local",
        _make_velocity_representation_policy(),
    )

    assert metadata["observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["policy_input_names"] == ["proprio_history", "actor_command"]
    assert metadata["policy_output_names"] == ["actions", "predicted_lin_vel"]
    assert metadata["student_observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["command_observation_names"] == ["command"]
    assert metadata["student_history_length"] == "5"
    assert metadata["student_history_flatten_dim"] == "false"
    assert metadata["student_history_order"] == "oldest_to_newest"


def test_non_representation_metadata_stays_legacy_shape() -> None:
    metadata = get_wheeled_legged_metadata(_make_dummy_metadata_env(), "local")

    assert metadata["observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert "policy_input_names" not in metadata
    assert "student_observation_names" not in metadata


@dataclass
class _DummyAgentCfg:
    experiment_name: str = "dummy_experiment"
    clip_actions: float | None = None


class _DummyEnv:
    def __init__(self, cfg, device, render_mode):
        self.cfg = cfg
        self.device = device
        self.render_mode = render_mode
        self.closed = False

    def close(self):
        self.closed = True


class _DummyPolicy:
    def __init__(self):
        self.student_called = False
        self.teacher_called = False

    def __call__(self, obs):
        del obs
        self.student_called = True
        return torch.tensor([1.0])

    def act_teacher(self, obs, stochastic_output=False):
        del obs, stochastic_output
        self.teacher_called = True
        return torch.tensor([2.0])


class _DummyRunner:
    policy = _DummyPolicy()

    def __init__(self, env, cfg, device):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.loaded = None

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        self.loaded = (path, load_cfg, strict, map_location)

    def get_inference_policy(self, device=None):
        del device
        return self.policy


def test_play_initial_teacher_role_uses_teacher_policy(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"checkpoint")
    _DummyRunner.policy = _DummyPolicy()

    env_cfg = SimpleNamespace(
        commands={},
        scene=SimpleNamespace(num_envs=1),
        viewer=SimpleNamespace(height=None, width=None),
    )
    captured = {}

    class DummyViewer:
        def __init__(self, env, policy, checkpoint_manager=None):
            captured["env"] = env
            captured["policy"] = policy
            captured["checkpoint_manager"] = checkpoint_manager

        def run(self):
            captured["action"] = captured["policy"]({"obs": torch.tensor([0.0])})

    monkeypatch.setattr(play, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(play, "load_env_cfg", lambda task_id, play: env_cfg)
    monkeypatch.setattr(play, "load_rl_cfg", lambda task_id: _DummyAgentCfg())
    monkeypatch.setattr(play, "load_runner_cls", lambda task_id: _DummyRunner)
    monkeypatch.setattr(play, "WheeledLeggedVelocityEnv", _DummyEnv)
    monkeypatch.setattr(play, "RslRlVecEnvWrapper", lambda env, clip_actions: env)
    monkeypatch.setattr(play, "NativeMujocoViewer", DummyViewer)

    play.run_play(
        "dummy_task",
        play.PlayConfig(
            checkpoint_file=str(checkpoint),
            device="cpu",
            viewer="native",
            policy_role="teacher",
        ),
    )

    assert torch.equal(captured["action"], torch.tensor([2.0]))
    assert _DummyRunner.policy.teacher_called
    assert not _DummyRunner.policy.student_called


def test_representation_tests_are_not_git_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        "tests/test_representation_teacher_student_config.py",
        "rsl_rl/tests/models/test_representation_actor_critic.py",
        "rsl_rl/tests/models/test_representation_velocity_actor_critic.py",
        "rsl_rl/tests/algorithms/test_representation_teacher_student_ppo.py",
        "rsl_rl/tests/algorithms/test_representation_velocity_teacher_student_ppo.py",
    ]
    result = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout
