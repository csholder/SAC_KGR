import torch as th
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import src.common.utils as utils
from src.common.common_class import Observation
from src.common.common_class import MLPFeaturesExtractor


class BaseCritic(nn.Module):
    def __init__(
        self,
        net_arch: List[int],
        action_dim: int,
        history_dim: int,
        state_dim: int,
        ff_dropout_rate: float,
        history_num_layers: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 1,
        feature_extractor: nn.Module = None,
        share_feature_extractor: bool = False,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(BaseCritic, self).__init__()

        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_dim = history_dim
        self.activation_fn = activation_fn
        self.feature_dim = action_dim
        self.n_critics = n_critics

        self.share_feature_extractor = False if feature_extractor is None else share_feature_extractor
        if feature_extractor is None:
            self.feature_extractor = MLPFeaturesExtractor(action_dim, history_dim, state_dim, self.feature_dim,
                                                          history_num_layers, ff_dropout_rate=ff_dropout_rate,
                                                          xavier_initialization=xavier_initialization,
                                                          relation_only=relation_only,)
        else:
            self.feature_extractor = feature_extractor

        self.q_networks = []
        for idx in range(n_critics):
            q_net = utils.create_mlp(action_dim + self.feature_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: Observation, action: th.Tensor, kg):
        with th.set_grad_enabled(not self.share_feature_extractor):
            feature = self.feature_extractor(obs, kg)
        assert feature.shape[:-1] == action.shape[:-1]
        qvalue_input = th.cat([feature, action], dim=-1)
        qvalue = th.cat(tuple(q_net(qvalue_input) for q_net in self.q_networks), dim=-1)
        qvalue, _ = th.min(qvalue, dim=-1)
        return qvalue

    def predict_q_value(self, obs: Observation, action: Tuple[th.Tensor, th.Tensor], kg) -> th.Tensor:
        action_r, action_e = action
        action_embedding = utils.get_action_embedding((action_r, action_e), kg)
        q_value = self(obs, action_embedding, kg)
        return q_value

    def q1_forward(self, obs, actions, kg, use_action_space_bucketing=False) -> th.Tensor:
        with th.no_grad():
            features = self.feature_extractor(obs, actions, kg, use_action_space_bucketing)
        return self.q_networks[0](th.cat([features, actions], dim=1))

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)
