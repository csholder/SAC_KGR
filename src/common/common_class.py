import torch
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union


class Observation():
    def __init__(
        self,
        num_rollout_steps,
        query_relation,
        target_entity,
        path=None,
        path_length=None,
    ):
        self.num_rollout_steps = num_rollout_steps
        self.query_relation = query_relation
        self.target_entity = target_entity
        self.path_r = path[0]
        self.path_e = path[1]
        self.path_length = path_length

    @property
    def start_entity(self):
        return self.path_e[:, 0]

    @property
    def last_relation(self):
        return th.gather(self.path_r, dim=1, index=self.path_length.unsqueeze(dim=-1).long()).squeeze(dim=-1).long()

    @property
    def current_entity(self):
        return th.gather(self.path_e, dim=1, index=self.path_length.unsqueeze(dim=-1).long()).squeeze(dim=-1).long()

    @property
    def done(self):
        return (self.path_length == self.num_rollout_steps).float()

    @property
    def last_step(self):
        return self.path_length == (self.num_rollout_steps - 1)


class BufferSample():
    def __init__(
        self,
        max_path_length,
        query_relation,
        target_entity,
        # done,
        entity_path,
        relation_path,
        reward,
        path_length,
    ):
        self.reward = reward
        # start_entity = entity_path[:, 0]
        self.action = (relation_path.gather(dim=1, index=path_length.unsqueeze(dim=-1).long()).squeeze(dim=-1),
                       entity_path.gather(dim=1, index=path_length.unsqueeze(dim=-1).long()).squeeze(dim=-1))
        last_relation_path = relation_path.scatter(1, path_length.unsqueeze(dim=-1).long(), 0)
        last_entity_path = entity_path.scatter(1, path_length.unsqueeze(dim=-1).long(), 0)
        # last_done = torch.zeros_like(query_relation, dtype=torch.bool)
        self.observation = Observation(
            num_rollout_steps=max_path_length,
            query_relation=query_relation,
            target_entity=target_entity,
            path=(last_relation_path, last_entity_path),
            path_length=path_length - 1
        )
        self.next_observation = Observation(
            num_rollout_steps=max_path_length,
            query_relation=query_relation,
            target_entity=target_entity,
            path=(relation_path, entity_path),
            path_length=path_length
        )


class MLPFeaturesExtractor(nn.Module):
    """
    Class that represents a features extractor.

    :param state_dim:
    :param features_dim: Number of features extracted.
    """

    def __init__(
        self,
        action_dim,
        history_dim,
        state_dim,
        features_dim: int,
        history_num_layers: int,
        ff_dropout_rate: float,
        xavier_initialization: bool,
        relation_only: bool,
    ):
        super(MLPFeaturesExtractor, self).__init__()
        assert action_dim > 0
        assert history_dim > 0
        assert state_dim > 0
        assert features_dim > 0
        assert history_num_layers > 0
        self._action_dim = action_dim
        self._history_dim = history_dim
        self._features_dim = features_dim
        self._state_dim = state_dim
        self._history_num_layers = history_num_layers
        self._xavier_initialization = xavier_initialization
        self._relation_only = relation_only

        self.W1 = nn.Linear(state_dim, action_dim)
        self.W2 = nn.Linear(action_dim, features_dim)
        self.W1Dropout = nn.Dropout(p=ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=ff_dropout_rate)

        self.path_encoder = nn.LSTM(input_size=self.action_dim,
                                    hidden_size=self.history_dim,
                                    num_layers=self.history_num_layers,
                                    batch_first=True)
        self.initialize_modules()

        print('========================== MLPFeaturesExtractor ==========================')
        print('_action_dim: ', self._action_dim)
        print('_history_dim: ', self._history_dim)
        print('_features_dim: ', self._features_dim)
        print('_state_dim: ', self._state_dim)
        print('_history_num_layers: ', self._history_num_layers)
        print('_xavier_initialization: ', self._xavier_initialization)
        print('ff_dropout_rate: ', ff_dropout_rate)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def history_dim(self) -> int:
        return self._history_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def history_num_layers(self):
        return self._history_num_layers

    def extractor(self, state: th.Tensor) -> th.Tensor:
        X = self.W1(state)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)
        return X2

    def get_action_embedding(self, action, kg):
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        # if self.relation_only:                                      # 搜寻下一步时只使用关系，不考虑实体信息
        #     action_embedding = relation_embedding
        # else:
        entity_embedding = kg.get_entity_embeddings(e)
        action_embedding = th.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def path_encoding(self, path: tuple, path_length: list, kg):
        path_embeddings = self.get_action_embedding(path, kg)
        packed_path_embeddings = nn.utils.rnn.pack_padded_sequence(path_embeddings, path_length,
                                                                   batch_first=True, enforce_sorted=False)
        _, (h, _) = self.path_encoder(packed_path_embeddings)  # action_dim -> history_dim
        return h[-1]

    def forward(self, obs: Observation, kg) -> th.Tensor:
        q, path_r, path_e, path_length = obs.query_relation, obs.path_r, obs.path_e, obs.path_length
        current_e = obs.current_entity
        path_length = (path_length.long() + 1).detach().cpu().tolist()
        H = self.path_encoding((path_r, path_e), path_length, kg)  # history_dim
        Q = kg.get_relation_embeddings(q)
        assert H.shape[:-1] == Q.shape[:-1]
        if not self._relation_only:
            E = kg.get_entity_embeddings(current_e)
            assert H.shape[:-1] == E.shape[:-1]
            state = th.cat([E, H, Q], dim=-1)  # history_dim + relation_dim + entity_dim
        else:
            state = th.cat([H, Q], dim=-1)  # history_dim + relation_dim
        state = self.extractor(state)  # features_dim
        return state

    def initialize_modules(self):
        if self._xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            for name, param in self.path_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                    # nn.init.constant_(param, 1.414)



class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool