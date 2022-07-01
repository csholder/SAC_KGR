import warnings
from typing import Any, Dict, Generator, List, Optional, Union, Tuple
import numpy as np
import torch as th

from src.common.common_class import BufferSample
from src.common.common_class import Observation


class ReplayBuffer():
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        max_path_length: int,
        device: Union[th.device, str] = "cpu"
    ):
        # Adjust buffer size
        self.buffer_size = buffer_size
        self.max_path_length = max_path_length

        self.query_relations = np.zeros(self.buffer_size, dtype=np.int)
        self.target_entities = np.zeros(self.buffer_size, dtype=np.int)

        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        # self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.paths_length = np.zeros(self.buffer_size, dtype=np.int)

        self.paths_entity = np.zeros((self.buffer_size, max_path_length + 1), dtype=np.int)
        self.paths_relation = np.zeros((self.buffer_size, max_path_length + 1), dtype=np.int)

        self.pos = 0
        self.full = False
        self.device = device

    def add(
        self,
        entity_path: th.Tensor,
        relation_path: th.Tensor,
        query_relation: th.Tensor,
        target_entity: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
        path_length: th.Tensor,
    ) -> None:
        batch_size = entity_path.shape[0]
        mid_batch_size = min(batch_size, self.buffer_size - self.pos - 1)
        self.paths_entity[self.pos: self.pos + mid_batch_size] = entity_path[:mid_batch_size].detach().cpu().numpy().copy()
        self.paths_relation[self.pos: self.pos + mid_batch_size] = relation_path[:mid_batch_size].detach().cpu().numpy().copy()
        self.query_relations[self.pos: self.pos + mid_batch_size] = query_relation[:mid_batch_size].detach().cpu().numpy().copy()
        self.target_entities[self.pos: self.pos + mid_batch_size] = target_entity[:mid_batch_size].detach().cpu().numpy().copy()
        self.rewards[self.pos: self.pos + mid_batch_size] = reward[:mid_batch_size].detach().cpu().numpy().copy()
        # self.dones[self.pos: self.pos + mid_batch_size] = done[:mid_batch_size].detach().cpu().numpy().copy()
        self.paths_length[self.pos: self.pos + mid_batch_size] = path_length[:mid_batch_size].detach().cpu().numpy().copy()

        self.pos += mid_batch_size

        if mid_batch_size < batch_size:
            self.pos = 0
            remain_batch_size = batch_size - mid_batch_size
            self.paths_entity[self.pos: self.pos + remain_batch_size] = entity_path[mid_batch_size:].detach().cpu().numpy().copy()
            self.paths_relation[self.pos: self.pos + remain_batch_size] = relation_path[mid_batch_size:].detach().cpu().numpy().copy()
            self.query_relations[self.pos: self.pos + remain_batch_size] = query_relation[mid_batch_size:].detach().cpu().numpy().copy()
            self.target_entities[self.pos: self.pos + remain_batch_size] = target_entity[mid_batch_size:].detach().cpu().numpy().copy()
            self.rewards[self.pos: self.pos + remain_batch_size] = reward[mid_batch_size:].detach().cpu().numpy().copy()
            # self.dones[self.pos: self.pos + remain_batch_size] = done[mid_batch_size:].detach().cpu().numpy().copy()
            self.paths_length[self.pos: self.pos + remain_batch_size] = path_length[mid_batch_size:].detach().cpu().numpy().copy()
            self.pos += remain_batch_size

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int,):
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = np.random.randint(1, self.buffer_size, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> BufferSample:
        data = (
            self.max_path_length,
            self.query_relations[batch_inds],
            self.target_entities[batch_inds],
            # self.dones[batch_inds],
            self.paths_entity[batch_inds],
            self.paths_relation[batch_inds],
            self.rewards[batch_inds],
            self.paths_length[batch_inds],
        )
        return BufferSample(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    def __len__(self):
        if self.full:
            return self.buffer_size
        return self.pos + 1