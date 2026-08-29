"""Smoke tests for WF-TRON1B representation teacher-student configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks

import wheeled_legged_mjlab  # noqa: F401
from rsl_rl.models import (
    DepthRepresentationVelocityActorCritic,
    RepresentationActorCritic,
    RepresentationVelocityActorCritic,
)
from wheeled_legged_mjlab.rl.runner import get_wheeled_legged_metadata
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    DEPTH_BUFFER_SIZE,
    DEPTH_BUFFER_UPDATE_PERIOD,
    DEPTH_CALIBRATION_BIAS_RANGE_M,
    DEPTH_CALIBRATION_SCALE_RANGE,
    DEPTH_CAPTURE_FREQUENCY_HZ,
    DEPTH_CAMERA_ENTITY_NAME,
    DEPTH_CAMERA_FOVY_DELTA_RANGE_DEG,
    DEPTH_CAMERA_HEIGHT,
    DEPTH_CAMERA_NAME,
    DEPTH_CAMERA_PITCH_DELTA_RANGE_RAD,
    DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
    DEPTH_CAMERA_WIDTH,
    DEPTH_DROPOUT_AREA_FRACTION_RANGE,
    DEPTH_DROPOUT_ASPECT_RATIO_RANGE,
    DEPTH_DROPOUT_ENABLED,
    DEPTH_DROPOUT_PATCH_COUNT_RANGE,
    DEPTH_DROPOUT_PROBABILITY,
    DEPTH_DISTANCE_NOISE_ENABLED,
    DEPTH_LEFT_CROP,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    DEPTH_MODEL_WIDTH,
    DEPTH_NOISE_BASE_M,
    DEPTH_NOISE_QUADRATIC_COEFF,
    DEPTH_RANDOMIZATION_ENABLED,
    DEPTH_SYSTEM_DELAY_RANGE_S,
    wf_tron1b_rough_depth_env_cfg,
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


DEPTH_RANDOMIZATION_PARAMS = {
    "enable_depth_randomization": DEPTH_RANDOMIZATION_ENABLED,
    "calibration_scale_range": DEPTH_CALIBRATION_SCALE_RANGE,
    "calibration_bias_range_m": DEPTH_CALIBRATION_BIAS_RANGE_M,
    "enable_depth_distance_noise": DEPTH_DISTANCE_NOISE_ENABLED,
    "noise_base_m": DEPTH_NOISE_BASE_M,
    "noise_quadratic_coeff": DEPTH_NOISE_QUADRATIC_COEFF,
    "enable_depth_dropout": DEPTH_DROPOUT_ENABLED,
    "dropout_probability": DEPTH_DROPOUT_PROBABILITY,
    "dropout_patch_count_range": DEPTH_DROPOUT_PATCH_COUNT_RANGE,
    "dropout_area_fraction_range": DEPTH_DROPOUT_AREA_FRACTION_RANGE,
    "dropout_aspect_ratio_range": DEPTH_DROPOUT_ASPECT_RATIO_RANGE,
}


def _assert_dynamics_context_group(cfg) -> None:
    dynamics_group = cfg.observations["dynamics_context"]
    dynamics_terms = dynamics_group.terms

    assert dynamics_group.enable_corruption is False
    assert set(dynamics_terms) == {"domain_randomization_delta_quantity"}
    assert dynamics_terms["domain_randomization_delta_quantity"].func is mdp.domain_randomization_delta_quantity


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
        "critic": ("critic", "dynamics_context"),
        "student_history": ("actor_history",),
        "privileged_encoder": ("critic", "dynamics_context"),
    }
    assert "actor_history" in flat_env.observations
    assert "dynamics_context" in flat_env.observations


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
        "critic": ("critic", "dynamics_context"),
        "privileged_encoder": ("privileged_encoder", "dynamics_context"),
    }
    assert "proprio_history" in flat_env.observations
    assert "actor_history" not in flat_env.observations
    assert "dynamics_context" in flat_env.observations
    assert "latent_dynamics_command_generation" not in flat_env.observations


def test_actor_history_and_rough_privileged_observations() -> None:
    cfg = wf_tron1b_rough_env_cfg()

    assert cfg.observations["actor_history"].history_length == 5
    assert cfg.observations["actor_history"].flatten_history_dim is False
    assert cfg.observations["actor_history"].enable_corruption is True
    assert cfg.observations["actor"].enable_corruption is False
    assert cfg.observations["critic"].enable_corruption is False
    _assert_dynamics_context_group(cfg)

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
    assert "domain_randomization_delta_quantity" not in critic_terms


def test_representation_velocity_observation_groups() -> None:
    cfg = wf_tron1b_rough_rep_ts_lin_vel_env_cfg()

    assert cfg.observations["proprio_history"].history_length == 5
    assert cfg.observations["proprio_history"].flatten_history_dim is False
    assert cfg.observations["proprio_history"].enable_corruption is True
    assert cfg.observations["actor_command"].enable_corruption is False
    assert cfg.observations["lin_vel_target"].enable_corruption is False
    assert cfg.observations["critic"].enable_corruption is False
    assert cfg.observations["privileged_encoder"].enable_corruption is False
    _assert_dynamics_context_group(cfg)

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
    assert "domain_randomization_delta_quantity" not in proprio_terms
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
    assert "domain_randomization_delta_quantity" not in critic_terms
    assert "domain_randomization_delta_quantity" not in privileged_terms
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
        "left_crop": DEPTH_LEFT_CROP,
        "depth_min_m": DEPTH_MIN_M,
        "depth_max_m": DEPTH_MAX_M,
        **DEPTH_RANDOMIZATION_PARAMS,
    }
    depth_sensor = next(
        sensor for sensor in cfg.scene.sensors if sensor.name == DEPTH_CAMERA_NAME
    )
    assert (depth_sensor.height, depth_sensor.width) == (
        DEPTH_CAMERA_HEIGHT,
        DEPTH_CAMERA_WIDTH,
    )
    assert DEPTH_MODEL_WIDTH == DEPTH_CAMERA_WIDTH - DEPTH_LEFT_CROP == 45
    assert depth_group.enable_corruption is False
    _assert_dynamics_context_group(cfg)
    assert agent["obs_groups"] == {
        "teacher_actor": ("actor",),
        "critic": ("critic", "dynamics_context"),
        "student_history": ("actor_history",),
        "privileged_encoder": ("critic", "dynamics_context"),
    }
    training_obs_groups = {
        group for groups in agent["obs_groups"].values() for group in groups
    }
    assert DEPTH_CAMERA_NAME not in training_obs_groups


@pytest.mark.parametrize(
    "factory",
    (
        wf_tron1b_rough_depth_env_cfg,
        wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg,
    ),
)
def test_depth_task_factories_expose_randomization_parameters(factory) -> None:
    cfg = factory(
        enable_depth_distance_noise=False,
        enable_depth_dropout=True,
        depth_noise_base_m=0.003,
        depth_noise_quadratic_coeff=0.007,
    )
    depth_term = cfg.observations[DEPTH_CAMERA_NAME].terms[DEPTH_CAMERA_NAME]

    assert depth_term.params["enable_depth_distance_noise"] is False
    assert depth_term.params["enable_depth_dropout"] is True
    assert depth_term.params["noise_base_m"] == 0.003
    assert depth_term.params["noise_quadratic_coeff"] == 0.007
    assert depth_term.params["dropout_probability"] == 0.30


def test_depth_velocity_representation_task_uses_async_depth_input() -> None:
    tasks = set(list_tasks())

    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth" in tasks
    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict" in tasks
    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-NoRough" in tasks

    cfg = wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg()
    agent = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth"))
    predict_agent = asdict(
        load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict")
    )
    no_rough_agent = asdict(
        load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-NoRough")
    )
    depth_group = cfg.observations[DEPTH_CAMERA_NAME]
    depth_term = depth_group.terms[DEPTH_CAMERA_NAME]

    assert depth_term.func is mdp.async_depth_buffer
    assert depth_term.params == {
        "sensor_name": DEPTH_CAMERA_NAME,
        "capture_frequency_hz": DEPTH_CAPTURE_FREQUENCY_HZ,
        "system_delay_range_s": DEPTH_SYSTEM_DELAY_RANGE_S,
        "left_crop": DEPTH_LEFT_CROP,
        "depth_min_m": DEPTH_MIN_M,
        "depth_max_m": DEPTH_MAX_M,
        **DEPTH_RANDOMIZATION_PARAMS,
    }
    assert cfg.observations["wheel_roughness"].terms["wheel_roughness"].func is mdp.wheel_roughness_gate
    assert agent["actor"]["class_name"] == "DepthRepresentationVelocityActorCritic"
    assert agent["actor"]["depth_gru_hidden_dim"] == 128
    assert agent["actor"]["depth_conv_strides"] == (2, 2, 1)
    assert agent["actor"]["depth_min_m"] == DEPTH_MIN_M
    assert agent["actor"]["depth_max_m"] == DEPTH_MAX_M
    assert agent["algorithm"]["representation_chunk_length"] == 24
    assert agent["algorithm"]["roughness_loss_coef"] == 0.2
    assert all("latent_dynamics" not in name for name in agent["actor"])
    assert all("latent_dynamics" not in name for name in agent["algorithm"])
    assert all("latent_rollout" not in name for name in agent["algorithm"])

    assert predict_agent["actor"]["class_name"].endswith(
        ":DepthRepresentationVelocityPredictorActorCritic"
    )
    assert predict_agent["actor"]["latent_dynamics_hidden_dims"] == (128, 256, 256, 128)
    assert predict_agent["actor"]["latent_dynamics_horizons"] == (1, 5, 10)
    assert predict_agent["algorithm"]["class_name"].endswith(
        ":RepresentationVelocityPredictorTeacherStudentPPO"
    )
    assert predict_agent["algorithm"]["predictor_learning_rate"] == 1.0e-3
    assert predict_agent["algorithm"]["roughness_loss_coef"] == 0.2
    assert no_rough_agent["algorithm"]["roughness_loss_coef"] == 0.0
    assert no_rough_agent["experiment_name"].endswith("depth_predict_no_rough_latent64")
    assert predict_agent["algorithm"]["latent_dynamics_loss_coef"] == 3.0
    assert predict_agent["algorithm"]["latent_dynamics_velocity_loss_coef"] == 1.0
    assert predict_agent["algorithm"]["latent_dynamics_use_ema_target"] is False
    assert predict_agent["algorithm"]["latent_dynamics_ema_decay"] == 0.99
    assert predict_agent["algorithm"]["latent_dynamics_horizons"] == (1, 5, 10)
    assert predict_agent["algorithm"]["latent_dynamics_horizon_weights"] == (
        1.0,
        0.75,
        0.5,
    )
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
        "critic": ("critic", "dynamics_context"),
        "privileged_encoder": ("privileged_encoder", "dynamics_context"),
        "depth_encoder": (DEPTH_CAMERA_NAME,),
        "wheel_roughness": ("wheel_roughness",),
    }
    assert predict_agent["obs_groups"] == agent["obs_groups"]


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
def test_depth_camera_domain_randomization_and_play_overrides() -> None:
    cfg = wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg()

    assert {"cam_pos", "cam_pitch", "cam_fovy"} <= set(cfg.events)
    assert not {"cam_pos", "cam_pitch", "cam_fovy"} & set(
        wf_tron1b_rough_env_cfg().events
    )

    cam_pos = cfg.events["cam_pos"]
    assert cam_pos.func is mdp.dr.cam_pos
    assert cam_pos.mode == "reset"
    assert cam_pos.params["asset_cfg"].camera_names == (DEPTH_CAMERA_ENTITY_NAME,)
    assert cam_pos.params["distribution"] == "uniform"
    assert cam_pos.params["operation"] == "add"
    assert cam_pos.params["ranges"] == {
        0: DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
        1: DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
        2: DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
    }
    assert cam_pos.params["shared_random"] is False

    cam_pitch = cfg.events["cam_pitch"]
    assert cam_pitch.func is mdp.dr.cam_quat
    assert cam_pitch.mode == "reset"
    assert cam_pitch.params["roll_range"] == (0.0, 0.0)
    assert cam_pitch.params["pitch_range"] == DEPTH_CAMERA_PITCH_DELTA_RANGE_RAD
    assert cam_pitch.params["yaw_range"] == (0.0, 0.0)

    cam_fovy = cfg.events["cam_fovy"]
    assert cam_fovy.func is mdp.dr.cam_fovy
    assert cam_fovy.mode == "reset"
    assert cam_fovy.params["operation"] == "add"
    assert cam_fovy.params["ranges"] == DEPTH_CAMERA_FOVY_DELTA_RANGE_DEG
    assert cam_fovy.params["shared_random"] is False

    play_cfg = wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(play=True)
    assert not {"cam_pos", "cam_pitch", "cam_fovy"} & set(play_cfg.events)
    play_depth_term = play_cfg.observations[DEPTH_CAMERA_NAME].terms[
        DEPTH_CAMERA_NAME
    ]
    assert play_depth_term.params["capture_frequency_hz"] == 30.0
    assert play_depth_term.params["system_delay_range_s"] == (0.0, 0.0)
    assert play_depth_term.params["left_crop"] == DEPTH_LEFT_CROP
    assert play_depth_term.params["depth_min_m"] == DEPTH_MIN_M
    assert play_depth_term.params["depth_max_m"] == DEPTH_MAX_M
    assert play_depth_term.params["enable_depth_randomization"] is False
    assert play_depth_term.params["enable_depth_distance_noise"] is False
    assert play_depth_term.params["enable_depth_dropout"] is False

    buffered_play_cfg = wf_tron1b_rough_depth_env_cfg(play=True)
    buffered_depth_term = buffered_play_cfg.observations[DEPTH_CAMERA_NAME].terms[
        DEPTH_CAMERA_NAME
    ]
    assert buffered_depth_term.func is mdp.depth_buffer
    assert "system_delay_range_s" not in buffered_depth_term.params
    assert buffered_depth_term.params["left_crop"] == DEPTH_LEFT_CROP
    assert buffered_depth_term.params["depth_min_m"] == DEPTH_MIN_M
    assert buffered_depth_term.params["depth_max_m"] == DEPTH_MAX_M
    assert buffered_depth_term.params["enable_depth_randomization"] is False
    assert buffered_depth_term.params["enable_depth_distance_noise"] is False
    assert buffered_depth_term.params["enable_depth_dropout"] is False


def test_depth_image_crops_and_encodes_invalid_without_modifying_camera_data() -> None:
    raw_depth = torch.full(
        (1, DEPTH_CAMERA_HEIGHT, DEPTH_CAMERA_WIDTH, 1),
        1.1,
        dtype=torch.float64,
    )
    raw_depth[0, 0, DEPTH_LEFT_CROP : DEPTH_LEFT_CROP + 10, 0] = torch.tensor(
        [-1.0, 0.0, torch.nan, -torch.inf, torch.inf, 0.1, 0.2, 1.1, 2.0, 2.5]
    )
    original_depth = raw_depth.clone()
    env = SimpleNamespace(
        scene={"depth_camera": SimpleNamespace(data=SimpleNamespace(depth=raw_depth))}
    )

    depth = observation_mdp.depth_image(
        env,
        left_crop=DEPTH_LEFT_CROP,
    )
    deployment_depth = observation_mdp.preprocess_depth_image(
        raw_depth.squeeze(-1),
        left_crop=DEPTH_LEFT_CROP,
    )

    assert depth.shape == (1, DEPTH_CAMERA_HEIGHT, DEPTH_MODEL_WIDTH)
    torch.testing.assert_close(
        depth[0, 0, :10],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 1.1, 2.0, 2.5]),
    )
    torch.testing.assert_close(deployment_depth, depth)
    torch.testing.assert_close(raw_depth, original_depth, equal_nan=True)
    assert depth.dtype is torch.float32
    assert depth.is_contiguous()
    assert torch.isfinite(depth).all()
    assert torch.all(depth >= 0.0)
    with pytest.raises(ValueError, match="left_crop must be in"):
        observation_mdp.depth_image(env, left_crop=DEPTH_CAMERA_WIDTH)


@pytest.mark.parametrize(
    ("depth_min_m", "depth_max_m", "message"),
    (
        (-0.1, 2.0, "depth_min_m must be >= 0"),
        (0.2, 0.2, "depth_max_m must be greater than depth_min_m"),
        (2.0, 1.0, "depth_max_m must be greater than depth_min_m"),
    ),
)
def test_depth_frame_processor_validates_noise_range(
    depth_min_m: float,
    depth_max_m: float,
    message: str,
) -> None:
    raw_depth = torch.ones(1, 1, 1, 1)
    with pytest.raises(ValueError, match=message):
        observation_mdp._DepthFrameProcessor()(
            raw_depth.squeeze(-1),
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )


def test_depth_calibration_is_episode_stable_and_resampled_on_reset() -> None:
    torch.manual_seed(0)
    processor = observation_mdp._DepthFrameProcessor()
    raw_depth = torch.ones(3, 4, 5)
    randomization = {
        "enable_depth_randomization": True,
        "enable_depth_distance_noise": False,
        "calibration_scale_range": (0.98, 1.02),
        "calibration_bias_range_m": (-0.01, 0.01),
    }

    first = processor(raw_depth, **randomization)
    assert processor._calibration_scale is not None
    assert processor._calibration_bias_m is not None
    initial_scale = processor._calibration_scale.clone()
    initial_bias = processor._calibration_bias_m.clone()

    second = processor(raw_depth, **randomization)
    torch.testing.assert_close(second, first)
    torch.testing.assert_close(processor._calibration_scale, initial_scale)
    torch.testing.assert_close(processor._calibration_bias_m, initial_bias)

    processor.reset(torch.tensor([1]))
    torch.testing.assert_close(processor._calibration_scale[[0, 2]], initial_scale[[0, 2]])
    torch.testing.assert_close(processor._calibration_bias_m[[0, 2]], initial_bias[[0, 2]])
    assert not (
        torch.equal(processor._calibration_scale[1], initial_scale[1])
        and torch.equal(processor._calibration_bias_m[1], initial_bias[1])
    )
    after_partial_reset = processor(raw_depth, **randomization)
    torch.testing.assert_close(after_partial_reset[[0, 2]], first[[0, 2]])
    assert not torch.equal(after_partial_reset[1], first[1])

    partial_scale = processor._calibration_scale.clone()
    partial_bias = processor._calibration_bias_m.clone()
    processor.reset()
    assert not torch.equal(processor._calibration_scale, partial_scale)
    assert not torch.equal(processor._calibration_bias_m, partial_bias)


def test_structured_depth_dropout_is_bounded_and_reproducible() -> None:
    test_dropout_probability = 0.30
    mask_kwargs = {
        "batch_size": 2048,
        "height": DEPTH_CAMERA_HEIGHT,
        "width": DEPTH_MODEL_WIDTH,
        "device": torch.device("cpu"),
        "probability": test_dropout_probability,
        "patch_count_range": DEPTH_DROPOUT_PATCH_COUNT_RANGE,
        "area_fraction_range": DEPTH_DROPOUT_AREA_FRACTION_RANGE,
        "aspect_ratio_range": DEPTH_DROPOUT_ASPECT_RATIO_RANGE,
    }
    torch.manual_seed(7)
    mask = observation_mdp._DepthFrameProcessor._structured_dropout_mask(
        **mask_kwargs
    )
    torch.manual_seed(7)
    repeated_mask = observation_mdp._DepthFrameProcessor._structured_dropout_mask(
        **mask_kwargs
    )

    torch.testing.assert_close(repeated_mask, mask)
    invalid_pixels = mask.sum(dim=(-1, -2))
    active = invalid_pixels > 0
    active_fraction = active.to(torch.float32).mean().item()
    assert active_fraction == pytest.approx(test_dropout_probability, abs=0.04)
    max_pixels = math.floor(
        DEPTH_DROPOUT_AREA_FRACTION_RANGE[1]
        * DEPTH_CAMERA_HEIGHT
        * DEPTH_MODEL_WIDTH
    )
    assert torch.all(invalid_pixels[active] <= max_pixels)

    torch.manual_seed(8)
    processed = observation_mdp._DepthFrameProcessor()(
        torch.full((4, DEPTH_CAMERA_HEIGHT, DEPTH_MODEL_WIDTH), 1.1),
        enable_depth_randomization=True,
        enable_depth_distance_noise=False,
        enable_depth_dropout=True,
        calibration_scale_range=(1.0, 1.0),
        calibration_bias_range_m=(0.0, 0.0),
        dropout_probability=1.0,
        dropout_patch_count_range=DEPTH_DROPOUT_PATCH_COUNT_RANGE,
        dropout_area_fraction_range=DEPTH_DROPOUT_AREA_FRACTION_RANGE,
        dropout_aspect_ratio_range=DEPTH_DROPOUT_ASPECT_RATIO_RANGE,
    )
    filled_invalid = processed == 0.0
    assert torch.all(filled_invalid.any(dim=(-1, -2)))
    assert torch.all(filled_invalid.sum(dim=(-1, -2)) <= max_pixels)


def test_disabled_depth_dropout_does_not_sample_rng(monkeypatch) -> None:
    processor = observation_mdp._DepthFrameProcessor()

    def unexpected_rng(*args, **kwargs):
        del args, kwargs
        raise AssertionError("depth dropout RNG must not run when disabled")

    monkeypatch.setattr(torch, "rand", unexpected_rng)
    depth = processor(
        torch.full((2, 3, 4), 1.1),
        enable_depth_randomization=True,
        calibration_scale_range=(1.0, 1.0),
        calibration_bias_range_m=(0.0, 0.0),
        enable_depth_distance_noise=False,
        enable_depth_dropout=False,
        dropout_probability=1.0,
    )

    torch.testing.assert_close(depth, torch.full_like(depth, 1.1))


def test_depth_distance_noise_uses_quadratic_sigma(monkeypatch) -> None:
    processor = observation_mdp._DepthFrameProcessor()
    raw_depth = torch.tensor([[[0.2, 1.0, 2.0, 3.0]]])

    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    processed = processor(
        raw_depth,
        depth_min_m=0.0,
        depth_max_m=4.0,
        enable_depth_randomization=True,
        calibration_scale_range=(1.0, 1.0),
        calibration_bias_range_m=(0.0, 0.0),
        enable_depth_distance_noise=True,
        noise_base_m=DEPTH_NOISE_BASE_M,
        noise_quadratic_coeff=DEPTH_NOISE_QUADRATIC_COEFF,
        dropout_probability=0.0,
    )

    measured_sigma = processed - raw_depth
    expected_sigma = (
        DEPTH_NOISE_BASE_M + DEPTH_NOISE_QUADRATIC_COEFF * raw_depth.square()
    )
    torch.testing.assert_close(
        measured_sigma,
        expected_sigma,
        rtol=0.0,
        atol=1.0e-7,
    )


@pytest.mark.parametrize(
    "callable_obj",
    (
        observation_mdp._DepthFrameProcessor.__call__,
        observation_mdp.DepthBuffer.__call__,
        observation_mdp.AsyncDepthBuffer.__call__,
    ),
)
def test_depth_noise_low_level_defaults_match_env_config(callable_obj) -> None:
    parameters = inspect.signature(callable_obj).parameters

    assert parameters["noise_base_m"].default == DEPTH_NOISE_BASE_M
    assert (
        parameters["noise_quadratic_coeff"].default
        == DEPTH_NOISE_QUADRATIC_COEFF
    )


def test_disabled_depth_distance_noise_does_not_sample_rng(monkeypatch) -> None:
    processor = observation_mdp._DepthFrameProcessor()

    def unexpected_rng(*args, **kwargs):
        del args, kwargs
        raise AssertionError("depth RNG must not run when distance noise is disabled")

    monkeypatch.setattr(torch, "rand", unexpected_rng)
    monkeypatch.setattr(torch, "randn_like", unexpected_rng)
    depth = processor(
        torch.full((2, 3, 4), 1.1),
        enable_depth_randomization=True,
        calibration_scale_range=(1.0, 1.0),
        calibration_bias_range_m=(0.0, 0.0),
        enable_depth_distance_noise=False,
        noise_base_m=DEPTH_NOISE_BASE_M,
        noise_quadratic_coeff=DEPTH_NOISE_QUADRATIC_COEFF,
        dropout_probability=0.0,
        dropout_patch_count_range=DEPTH_DROPOUT_PATCH_COUNT_RANGE,
        dropout_area_fraction_range=DEPTH_DROPOUT_AREA_FRACTION_RANGE,
        dropout_aspect_ratio_range=DEPTH_DROPOUT_ASPECT_RATIO_RANGE,
    )

    assert depth.dtype is torch.float32
    torch.testing.assert_close(depth, torch.full_like(depth, 1.1))
    processor.reset()


def test_depth_randomization_preserves_original_invalid_pixels(monkeypatch) -> None:
    processor = observation_mdp._DepthFrameProcessor()
    raw_depth = torch.tensor(
        [[[0.0, torch.nan, torch.inf, -torch.inf, 1.0]]],
        dtype=torch.float64,
    )

    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    processed = processor(
        raw_depth,
        enable_depth_randomization=True,
        calibration_scale_range=(1.0, 1.0),
        calibration_bias_range_m=(0.5, 0.5),
        enable_depth_distance_noise=True,
        noise_base_m=DEPTH_NOISE_BASE_M,
        noise_quadratic_coeff=DEPTH_NOISE_QUADRATIC_COEFF,
        dropout_probability=0.0,
    )

    torch.testing.assert_close(processed[0, 0, :4], torch.zeros(4))
    assert processed.dtype is torch.float32
    assert processed.is_contiguous()
    assert torch.isfinite(processed).all()
    assert torch.all(processed >= 0.0)


@pytest.mark.parametrize(
    ("term_factory", "term_kwargs", "stable_step", "capture_step"),
    (
        (observation_mdp.depth_buffer, {"buffer_size": 1, "update_period": 5}, 4, 5),
        (
            observation_mdp.async_depth_buffer,
            {"capture_frequency_hz": 30.0},
            1,
            2,
        ),
    ),
)
def test_depth_noise_is_sampled_only_for_new_buffered_frames(
    monkeypatch,
    term_factory,
    term_kwargs,
    stable_step: int,
    capture_step: int,
) -> None:
    env = SimpleNamespace(
        common_step_counter=0,
        step_dt=0.02,
        frame=torch.ones(2, 3, 4),
    )
    term = term_factory(cfg=None, env=env)
    noise_calls = 0

    def get_depth(env, sensor_name):
        del sensor_name
        return env.frame

    def deterministic_noise(depth):
        nonlocal noise_calls
        noise_calls += 1
        return torch.full_like(depth, float(noise_calls))

    monkeypatch.setattr(observation_mdp, "_camera_depth_image_meters", get_depth)
    monkeypatch.setattr(torch, "randn_like", deterministic_noise)
    randomization = {
        "depth_min_m": 0.0,
        "depth_max_m": 10.0,
        "enable_depth_randomization": True,
        "calibration_scale_range": (1.0, 1.0),
        "calibration_bias_range_m": (0.0, 0.0),
        "enable_depth_distance_noise": True,
        "noise_base_m": 0.1,
        "noise_quadratic_coeff": 0.0,
        "dropout_probability": 0.0,
    }

    first = term(env, **term_kwargs, **randomization).clone()
    env.common_step_counter = stable_step
    repeated = term(env, **term_kwargs, **randomization).clone()
    torch.testing.assert_close(repeated, first)
    assert noise_calls == 1

    env.common_step_counter = capture_step
    captured = term(env, **term_kwargs, **randomization)
    assert noise_calls == 2
    assert not torch.equal(captured, first)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    (
        ("calibration_scale_range", (0.0, 1.0), "calibration_scale_range"),
        ("calibration_bias_range_m", (0.1, -0.1), "calibration_bias_range_m"),
        ("noise_base_m", -0.1, "noise_base_m"),
        ("noise_base_m", math.nan, "noise_base_m"),
        ("noise_base_m", math.inf, "noise_base_m"),
        ("noise_quadratic_coeff", -0.1, "noise_quadratic_coeff"),
        ("noise_quadratic_coeff", math.nan, "noise_quadratic_coeff"),
        ("noise_quadratic_coeff", math.inf, "noise_quadratic_coeff"),
        ("dropout_probability", 1.1, "dropout_probability"),
        ("dropout_patch_count_range", (0, 3), "dropout_patch_count_range"),
        ("dropout_area_fraction_range", (0.1, 1.1), "dropout_area_fraction_range"),
        ("dropout_aspect_ratio_range", (0.0, 1.0), "dropout_aspect_ratio_range"),
    ),
)
def test_depth_randomization_validates_parameters(
    parameter: str,
    value,
    message: str,
) -> None:
    processor = observation_mdp._DepthFrameProcessor()

    with pytest.raises(ValueError, match=message):
        processor(torch.ones(1, 3, 4), **{parameter: value})


def test_depth_buffer_updates_every_five_policy_steps(monkeypatch) -> None:
    env = SimpleNamespace(common_step_counter=0, frame=torch.ones(2, 2, 3))
    term = observation_mdp.depth_buffer(cfg=None, env=env)
    depth_calls = 0

    def get_depth(env, sensor_name):
        nonlocal depth_calls
        del sensor_name
        depth_calls += 1
        return env.frame

    monkeypatch.setattr(
        observation_mdp,
        "_camera_depth_image_meters",
        get_depth,
    )
    depth_range = {"depth_min_m": 0.0, "depth_max_m": 10.0}

    obs = term(env, buffer_size=5, update_period=5, left_crop=1, **depth_range)
    assert obs.shape == (2, 5, 2, 2)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 4
    env.frame = torch.full((2, 2, 3), 2.0)
    obs = term(env, buffer_size=5, update_period=5, left_crop=1, **depth_range)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 5
    obs = term(env, buffer_size=5, update_period=5, left_crop=1, **depth_range)
    assert torch.all(obs[:, :4] == 1.0)
    assert torch.all(obs[:, 4] == 2.0)
    assert depth_calls == 2

    env.common_step_counter = 6
    env.frame = torch.stack((torch.full((2, 3), 3.0), torch.full((2, 3), 4.0)))
    term.reset(torch.tensor([1]))
    obs = term(env, buffer_size=5, update_period=5, left_crop=1, **depth_range)
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
        del sensor_name
        depth_calls += 1
        return env.frame

    monkeypatch.setattr(
        observation_mdp,
        "_camera_depth_image_meters",
        get_depth,
    )
    depth_range = {"depth_min_m": 0.0, "depth_max_m": 10.0}

    obs = term(env, capture_frequency_hz=30.0, left_crop=1, **depth_range)
    assert obs.shape == (2, 1, 2, 2)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 1
    env.frame = torch.full((2, 2, 3), 2.0)
    obs = term(env, capture_frequency_hz=30.0, left_crop=1, **depth_range)
    assert torch.all(obs == 1.0)
    assert depth_calls == 1

    env.common_step_counter = 2
    obs = term(env, capture_frequency_hz=30.0, left_crop=1, **depth_range)
    assert torch.all(obs == 2.0)
    assert depth_calls == 2

    env.common_step_counter = 3
    env.frame = torch.full((2, 2, 3), 3.0)
    obs = term(env, capture_frequency_hz=30.0, left_crop=1, **depth_range)
    assert torch.all(obs == 2.0)
    assert depth_calls == 2

    env.common_step_counter = 4
    obs = term(env, capture_frequency_hz=30.0, left_crop=1, **depth_range)
    assert torch.all(obs == 3.0)
    assert depth_calls == 3


def test_wheel_roughness_preserves_left_right_order(monkeypatch) -> None:
    height_samples = torch.zeros(1, 2, 25)
    height_samples[0, 1, 0] = 0.20
    env = SimpleNamespace(scene={"wheel_height_scan": object()})

    monkeypatch.setattr(
        observation_mdp,
        "_terrain_clearance_samples",
        lambda sensor: height_samples,
    )
    gate = observation_mdp.wheel_roughness_gate(
        env,
        wheel_radius=0.127,
        gate_min=0.10,
        gate_max=0.40,
        grid_shape=(5, 5),
    )

    assert gate.shape == (1, 2)
    assert gate[0, 0].item() == pytest.approx(0.0)
    assert gate[0, 1].item() == pytest.approx(1.0)


def test_async_depth_buffer_applies_per_env_delay_and_reset(monkeypatch) -> None:
    env = SimpleNamespace(common_step_counter=0, step_dt=0.02, frame=torch.ones(2, 2, 3))
    term = observation_mdp.async_depth_buffer(cfg=None, env=env)
    depth_calls = 0

    def get_depth(env, sensor_name):
        nonlocal depth_calls
        del sensor_name
        depth_calls += 1
        return env.frame

    monkeypatch.setattr(observation_mdp, "_camera_depth_image_meters", get_depth)
    depth_range = {"depth_min_m": 0.0, "depth_max_m": 20.0}

    obs = term(
        env,
        capture_frequency_hz=30.0,
        system_delay_range_s=(0.0, 0.020),
        **depth_range,
    )
    assert obs.shape == (2, 1, 2, 3)
    assert torch.all(obs == 1.0)
    assert term._delay_s is not None
    term._delay_s.copy_(torch.tensor([0.005, 0.015], device=term._delay_s.device))

    env.common_step_counter = 2
    env.frame = torch.full((2, 2, 3), 2.0)
    obs = term(
        env,
        capture_frequency_hz=30.0,
        system_delay_range_s=(0.0, 0.020),
        **depth_range,
    )
    assert torch.all(obs[0] == 2.0)
    assert torch.all(obs[1] == 1.0)

    env.common_step_counter = 3
    obs = term(
        env,
        capture_frequency_hz=30.0,
        system_delay_range_s=(0.0, 0.020),
        **depth_range,
    )
    assert torch.all(obs == 2.0)

    env.common_step_counter = 4
    env.frame = torch.full((2, 2, 3), 3.0)
    obs = term(
        env,
        capture_frequency_hz=30.0,
        system_delay_range_s=(0.0, 0.020),
        **depth_range,
    )
    assert torch.all(obs[0] == 3.0)
    assert torch.all(obs[1] == 2.0)

    def fixed_delay(count, device):
        return torch.full((count,), 0.012, device=device, dtype=torch.float64)

    monkeypatch.setattr(term, "_sample_delay", fixed_delay)
    env.common_step_counter = 5
    env.frame = torch.stack((torch.full((2, 3), 4.0), torch.full((2, 3), 9.0)))
    term.reset(torch.tensor([1]))
    obs = term(
        env,
        capture_frequency_hz=30.0,
        system_delay_range_s=(0.0, 0.020),
        **depth_range,
    )
    assert torch.all(obs[0] == 3.0)
    assert torch.all(obs[1] == 9.0)
    assert term._delay_s is not None
    assert term._delay_s[0].item() == pytest.approx(0.005)
    assert term._delay_s[1].item() == pytest.approx(0.012)

    env.common_step_counter = 6
    env.frame = torch.stack((torch.full((2, 3), 5.0), torch.full((2, 3), 11.0)))
    term.reset(torch.tensor([1]))
    obs = term(
        env,
        capture_frequency_hz=30.0,
        system_delay_range_s=(0.0, 0.020),
        **depth_range,
    )
    assert torch.all(obs[0] == 4.0)
    assert torch.all(obs[1] == 11.0)
    assert depth_calls == 5


def test_async_depth_buffer_validates_clock_and_delay_ranges() -> None:
    env = SimpleNamespace(common_step_counter=0, step_dt=0.02)
    term = observation_mdp.async_depth_buffer(cfg=None, env=env)

    with pytest.raises(ValueError, match="capture_frequency_hz must be positive"):
        term(env, capture_frequency_hz=0.0)
    with pytest.raises(ValueError, match="system_delay_range_s must be non-negative"):
        term(env, system_delay_range_s=(-0.001, 0.020))
    with pytest.raises(ValueError, match="must be ordered"):
        term(env, system_delay_range_s=(0.020, 0.010))

    env.step_dt = 0.0
    with pytest.raises(ValueError, match="env.step_dt must be positive"):
        term(env)


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
            },
            get_term_cfg=lambda group, name: SimpleNamespace(
                scale=None,
                clip=None,
                flatten_history_dim=True,
                history_length=0,
            ),
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
            "dynamics_context": torch.randn(2, 13),
        },
        batch_size=[2],
    )
    return RepresentationActorCritic(
        obs,
        {
            "teacher_actor": ["actor"],
            "critic": ["critic"],
            "student_history": ["actor_history"],
            "privileged_encoder": ["critic", "dynamics_context"],
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
            "dynamics_context": torch.randn(2, 13),
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
            "privileged_encoder": ["privileged_encoder", "dynamics_context"],
        },
        output_dim=2,
        hidden_dims=[8],
        encoder_hidden_dims=[8],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )


def _make_depth_velocity_representation_policy() -> DepthRepresentationVelocityActorCritic:
    obs = TensorDict(
        {
            "proprio_history": torch.randn(2, 5, 3),
            "actor_command": torch.randn(2, 3),
            "lin_vel_target": torch.randn(2, 3),
            "critic": torch.randn(2, 5),
            "privileged_encoder": torch.randn(2, 4),
            "dynamics_context": torch.randn(2, 13),
            "depth_camera": torch.randn(2, 1, 32, 24),
        },
        batch_size=[2],
    )
    return DepthRepresentationVelocityActorCritic(
        obs,
        {
            "proprio_history": ["proprio_history"],
            "actor_command": ["actor_command"],
            "lin_vel_target": ["lin_vel_target"],
            "critic": ["critic"],
            "privileged_encoder": ["privileged_encoder", "dynamics_context"],
            "depth_encoder": ["depth_camera"],
        },
        output_dim=2,
        hidden_dims=[8],
        encoder_hidden_dims=[8],
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        distribution_cfg={"class_name": "GaussianDistribution"},
    )


def test_representation_metadata_describes_single_history_input() -> None:
    metadata = get_wheeled_legged_metadata(_make_dummy_metadata_env(), "local", _make_representation_policy())

    assert metadata["observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["policy_input_names"] == ["student_history"]
    assert metadata["student_observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert "dynamics_context" not in metadata["policy_input_names"]
    assert "domain_randomization_delta_quantity" not in metadata["student_observation_names"]
    assert metadata["student_history_length"] == "5"
    assert metadata["student_history_flatten_dim"] == "false"
    assert metadata["student_history_order"] == "oldest_to_newest"
    assert metadata["observation_terms_scale"] == [1.0, 1.0]
    assert metadata["observation_terms_flatten_history_dim"] == [True, True]
    assert metadata["observation_terms_history_length"] == [0, 0]
    assert metadata["observation_terms_clip"] == [
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
    ]


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
    assert "dynamics_context" not in metadata["policy_input_names"]
    assert "domain_randomization_delta_quantity" not in metadata["student_observation_names"]
    assert metadata["student_history_length"] == "5"
    assert metadata["student_history_flatten_dim"] == "false"
    assert metadata["student_history_order"] == "oldest_to_newest"


def test_depth_velocity_representation_metadata_matches_onnx_io() -> None:
    policy = _make_depth_velocity_representation_policy()
    metadata = get_wheeled_legged_metadata(_make_velocity_metadata_env(), "local", policy)
    onnx_policy = policy.as_onnx(verbose=False)

    assert metadata["policy_input_names"] == onnx_policy.input_names
    assert metadata["policy_output_names"] == onnx_policy.output_names
    assert metadata["depth_input_dtype"] == "float32"
    assert metadata["depth_input_unit"] == "m"
    assert metadata["depth_input_shape"] == [1, 1, 32, 24]
    assert metadata["depth_invalid_value"] == 0.0
    assert metadata["depth_min_m"] == DEPTH_MIN_M
    assert metadata["depth_max_m"] == DEPTH_MAX_M
    assert metadata["depth_preprocessing"] == "below_min_to_max,clamp,normalize"


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

    @property
    def unwrapped(self):
        return self

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
