"""RL configuration for WF-TRON1B velocity task."""

from dataclasses import dataclass
from typing import Tuple

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

class WFTRON1BRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    trial_message: str = ""


@dataclass
class RslRlRepresentationModelCfg(RslRlModelCfg):
    """Config for representation-level teacher-student actor-critic."""

    encoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    latent_dim: int = 64
    normalize_latent: bool = True
    class_name: str = "RepresentationActorCritic"


@dataclass
class RslRlRepresentationTeacherStudentPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config for representation-level teacher-student PPO."""

    proprio_encoder_learning_rate: float = 1.0e-3
    num_proprio_encoder_substeps: int = 1
    class_name: str = "RepresentationTeacherStudentPPO"


@dataclass
class RslRlRepresentationVelocityModelCfg(RslRlModelCfg):
    """Config for velocity representation teacher-student actor-critic."""

    encoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    latent_dim: int = 64
    normalize_latent: bool = True
    class_name: str = "RepresentationVelocityActorCritic"


@dataclass
class RslRlDepthRepresentationVelocityModelCfg(RslRlRepresentationVelocityModelCfg):
    """Config for depth velocity representation teacher-student actor-critic."""

    depth_feature_dim: int = 64
    depth_gru_hidden_dim: int = 64
    depth_channels: Tuple[int, ...] = (16, 32, 32)
    latent_dynamics_hidden_dims: Tuple[int, ...] = (128, 128)
    class_name: str = "DepthRepresentationVelocityActorCritic"


@dataclass
class RslRlRepresentationVelocityTeacherStudentPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config for velocity representation teacher-student PPO."""

    student_learning_rate: float = 1.0e-3
    num_student_substeps: int = 1
    num_representation_epochs: int | None = None
    num_representation_mini_batches: int | None = None
    representation_chunk_length: int = 12
    representation_loss_coef: float = 1.0
    lin_vel_loss_coef: float = 1.0
    latent_dynamics_loss_coef: float = 0.0
    latent_dynamics_horizons: Tuple[int, ...] = (1,)
    latent_dynamics_horizon_weights: Tuple[float, ...] = (1.0,)
    latent_dynamics_detach_source: bool = False
    num_latent_dynamics_epochs: int = 1
    num_latent_dynamics_mini_batches: int = 4
    class_name: str = "RepresentationVelocityTeacherStudentPPO"


def wf_tron1b_ppo_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON1B velocity task."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="wf_tron1b_velocity",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_rep_ts_lin_vel_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create velocity representation teacher-student runner configuration."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlRepresentationVelocityModelCfg(
            hidden_dims=(512, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=64,
            normalize_latent=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlRepresentationVelocityTeacherStudentPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            student_learning_rate=1.0e-3,
            num_student_substeps=1,
            representation_loss_coef=1.0,
            lin_vel_loss_coef=1.0,
        ),
        obs_groups={
            "proprio_history": ("proprio_history",),
            "actor_command": ("actor_command",),
            "lin_vel_target": ("lin_vel_target",),
            "critic": ("critic",),
            "privileged_encoder": ("privileged_encoder",),
        },
        experiment_name="wf_tron1b_velocity_rep_ts_lin_vel_latent64",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_rep_ts_lin_vel_depth_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create velocity representation teacher-student runner configuration with depth."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlDepthRepresentationVelocityModelCfg(
            hidden_dims=(512, 256, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=64,
            normalize_latent=True,
            depth_feature_dim=64,
            depth_gru_hidden_dim=64,
            depth_channels=(16, 32, 32),
            latent_dynamics_hidden_dims=(128, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlRepresentationVelocityTeacherStudentPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            student_learning_rate=1.0e-3,
            num_student_substeps=1,
            num_representation_epochs=1,
            num_representation_mini_batches=4,
            representation_chunk_length=12,
            representation_loss_coef=1.0,
            lin_vel_loss_coef=1.0,
            latent_dynamics_loss_coef=1.0,
            latent_dynamics_horizons=(1, 5),
            latent_dynamics_horizon_weights=(1.0, 0.5),
            latent_dynamics_detach_source=False,
            num_latent_dynamics_epochs=1,
            num_latent_dynamics_mini_batches=4,
        ),
        obs_groups={
            "proprio_history": ("proprio_history",),
            "actor_command": ("actor_command",),
            "lin_vel_target": ("lin_vel_target",),
            "critic": ("critic",),
            "privileged_encoder": ("privileged_encoder",),
            "depth_encoder": ("depth_camera",),
        },
        experiment_name="wf_tron1b_velocity_rep_ts_lin_vel_depth_latent64",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_rep_ts_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create representation-level teacher-student runner configuration."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlRepresentationModelCfg(
            hidden_dims=(512, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=64,
            normalize_latent=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlRepresentationTeacherStudentPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            proprio_encoder_learning_rate=1.0e-3,
            num_proprio_encoder_substeps=1,
        ),
        obs_groups={
            "teacher_actor": ("actor",),
            "critic": ("critic",),
            "student_history": ("actor_history",),
            "privileged_encoder": ("critic",),
        },
        experiment_name="wf_tron1b_velocity_rep_ts_latent64",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )
