"""Smoke tests for WF-TRON1B representation teacher-student configuration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import subprocess

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks

import wheeled_legged_mjlab  # noqa: F401
from rsl_rl.models import RepresentationActorCritic
from wheeled_legged_mjlab.rl.runner import get_wheeled_legged_metadata
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import wf_tron1b_rough_env_cfg


def test_representation_teacher_student_tasks_are_registered() -> None:
    tasks = set(list_tasks())

    assert "Mjlab-Velocity-Rough-WF-Tron1B-RepTS" in tasks
    assert "Mjlab-Velocity-Flat-WF-Tron1B-RepTS" in tasks

    rough_agent = asdict(load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS"))
    flat_env = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B-RepTS")

    assert rough_agent["algorithm"]["class_name"] == "RepresentationTeacherStudentPPO"
    assert rough_agent["actor"]["class_name"] == "RepresentationActorCritic"
    assert rough_agent["obs_groups"] == {
        "actor": ("actor",),
        "critic": ("critic",),
        "proprio_encoder": ("actor_history",),
        "privileged_encoder": ("critic",),
    }
    assert "actor_history" in flat_env.observations


def test_actor_history_and_rough_privileged_observations() -> None:
    cfg = wf_tron1b_rough_env_cfg()

    assert cfg.observations["actor_history"].history_length == 5
    assert cfg.observations["actor_history"].flatten_history_dim is True

    actor_terms = cfg.observations["actor"].terms
    actor_history_terms = cfg.observations["actor_history"].terms
    critic_terms = cfg.observations["critic"].terms

    assert "height_scan" not in actor_terms
    assert "height_scan" not in actor_history_terms
    assert "height_scan" in critic_terms
    assert "domain_randomization_delta_quantity" not in actor_terms
    assert "domain_randomization_delta_quantity" not in actor_history_terms


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
                "actor_history": SimpleNamespace(history_length=5, flatten_history_dim=True),
            }
        ),
    )


def _make_representation_policy() -> RepresentationActorCritic:
    obs = TensorDict(
        {
            "actor": torch.randn(2, 3),
            "actor_history": torch.randn(2, 15),
            "critic": torch.randn(2, 4),
        },
        batch_size=[2],
    )
    return RepresentationActorCritic(
        obs,
        {
            "actor": ["actor"],
            "critic": ["critic"],
            "proprio_encoder": ["actor_history"],
            "privileged_encoder": ["critic"],
        },
        output_dim=2,
        hidden_dims=[8],
        encoder_hidden_dims=[8],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )


def test_representation_metadata_describes_two_policy_inputs() -> None:
    metadata = get_wheeled_legged_metadata(_make_dummy_metadata_env(), "local", _make_representation_policy())

    assert metadata["observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["policy_input_names"] == ["actor_obs", "proprio_obs"]
    assert metadata["actor_observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["proprio_observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert metadata["proprio_history_length"] == "5"
    assert metadata["proprio_flatten_history_dim"] == "true"


def test_non_representation_metadata_stays_legacy_shape() -> None:
    metadata = get_wheeled_legged_metadata(_make_dummy_metadata_env(), "local")

    assert metadata["observation_names"] == ["base_ang_vel", "projected_gravity"]
    assert "policy_input_names" not in metadata
    assert "proprio_observation_names" not in metadata


def test_representation_tests_are_not_git_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        "tests/test_representation_teacher_student_config.py",
        "rsl_rl/tests/models/test_representation_actor_critic.py",
        "rsl_rl/tests/algorithms/test_representation_teacher_student_ppo.py",
    ]
    result = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout
