from __future__ import annotations

import pytest
import torch

from rsl_rl.algorithms import (
    PPO,
    RepresentationTeacherStudentPPO,
    RepresentationVelocityTeacherStudentPPO,
)
from rsl_rl.algorithms.representation_velocity_predictor_teacher_student_ppo import (
    RepresentationVelocityPredictorTeacherStudentPPO,
)


def _optimizer(learning_rate: float) -> torch.optim.Adam:
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=learning_rate)


@pytest.mark.parametrize(
    ("algorithm_class", "optimizer_fields"),
    [
        (PPO, (("optimizer", "learning_rate", "optimizer_state_dict"),)),
        (
            RepresentationTeacherStudentPPO,
            (
                ("optimizer", "learning_rate", "optimizer_state_dict"),
                (
                    "proprio_optimizer",
                    "proprio_encoder_learning_rate",
                    "proprio_optimizer_state_dict",
                ),
            ),
        ),
        (
            RepresentationVelocityTeacherStudentPPO,
            (
                ("optimizer", "learning_rate", "optimizer_state_dict"),
                (
                    "student_optimizer",
                    "student_learning_rate",
                    "student_optimizer_state_dict",
                ),
            ),
        ),
        (
            RepresentationVelocityPredictorTeacherStudentPPO,
            (
                ("optimizer", "learning_rate", "optimizer_state_dict"),
                (
                    "predictor_optimizer",
                    "predictor_learning_rate",
                    "predictor_optimizer_state_dict",
                ),
                (
                    "student_optimizer",
                    "student_learning_rate",
                    "student_optimizer_state_dict",
                ),
            ),
        ),
    ],
)
def test_load_restores_optimizer_learning_rate_fields(
    algorithm_class,
    optimizer_fields,
) -> None:
    algorithm = algorithm_class.__new__(algorithm_class)
    checkpoint = {}

    for index, (optimizer_name, field_name, checkpoint_name) in enumerate(
        optimizer_fields,
        start=1,
    ):
        restored_learning_rate = index * 1.0e-4
        setattr(algorithm, optimizer_name, _optimizer(9.0e-3))
        setattr(algorithm, field_name, 9.0e-3)
        checkpoint[checkpoint_name] = _optimizer(restored_learning_rate).state_dict()

    algorithm.load(
        checkpoint,
        load_cfg={"optimizer": True, "iteration": False},
        strict=True,
    )

    for optimizer_name, field_name, _ in optimizer_fields:
        optimizer = getattr(algorithm, optimizer_name)
        assert getattr(algorithm, field_name) == pytest.approx(
            optimizer.param_groups[0]["lr"]
        )
