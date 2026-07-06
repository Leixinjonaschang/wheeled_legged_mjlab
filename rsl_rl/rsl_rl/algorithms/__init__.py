# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .distillation import Distillation
from .ppo import PPO
from .representation_teacher_student_ppo import RepresentationTeacherStudentPPO
from .representation_velocity_teacher_student_ppo import RepresentationVelocityTeacherStudentPPO

__all__ = [
    "PPO",
    "Distillation",
    "RepresentationTeacherStudentPPO",
    "RepresentationVelocityTeacherStudentPPO",
]
