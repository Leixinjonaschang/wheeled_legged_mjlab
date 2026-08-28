from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


def test_observation_storage_preserves_leaf_dtypes() -> None:
    observations = TensorDict(
        {
            "float64": torch.zeros(2, 3, dtype=torch.float64),
            "int64": torch.zeros(2, 1, dtype=torch.int64),
            "bool": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )

    storage = RolloutStorage("rl", 2, 4, observations, [1])

    for key, value in observations.items():
        assert storage.observations[key].dtype == value.dtype
