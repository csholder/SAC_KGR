import torch
import torch as th
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import src.common.utils as utils
from src.common.policy_utils import get_action_space, get_action_space_in_buckets
from src.common.common_class import Observation
from src.common.common_class import MLPFeaturesExtractor
from src.common.base_critic import BaseCritic
from src.common.base_actor import BaseActor
from src.common.policy import BasePolicy


class Actor(BaseActor):
    def __init__(
        self,
        net_arch: List[int],
        action_dim: int,
        history_dim: int,
        state_dim: int,
        ff_dropout_rate: float,
        history_num_layers: int,
        activation_fn: Union[str, nn.Module] = nn.ReLU,
        action_dropout_rate: float = 0.5,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(Actor, self).__init__(
            net_arch,
            action_dim,
            history_dim,
            state_dim,
            ff_dropout_rate,
            history_num_layers,
            activation_fn,
            action_dropout_rate,
            xavier_initialization,
            relation_only,
        )


class Critic(BaseCritic):
    def __init__(
        self,
        net_arch: List[int],
        action_dim: int,
        history_dim: int,
        state_dim: int,
        ff_dropout_rate: float,
        history_num_layers: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_dropout_rate: float = 0.0,
        feature_extractor: nn.Module = None,
        n_critics: int = 1,
        share_feature_extractor: bool = False,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(Critic, self).__init__(
            net_arch=net_arch,
            action_dim=action_dim,
            history_dim=history_dim,
            state_dim=state_dim,
            ff_dropout_rate=ff_dropout_rate,
            history_num_layers=history_num_layers,
            activation_fn=activation_fn,
            n_critics=n_critics,
            share_feature_extractor=share_feature_extractor,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
        )
        self.action_dropout_rate = action_dropout_rate

    def forward_batch(self, obs: Observation, actions: th.Tensor, kg) -> th.Tensor:
        feature = self.feature_extractor(obs, kg)
        qvalue_input = th.cat([feature.unsqueeze(dim=1).expand_as(actions), actions], dim=-1)
        qvalues = th.cat(tuple(q_net(qvalue_input) for q_net in self.q_networks), dim=-1)
        q_values, _ = th.min(qvalues, dim=-1)
        return q_values

    def predict_q_value_batch(self, observation: Observation, action_space, kg):
        (r_space, e_space), action_mask = action_space
        action_embeddings = utils.get_action_embedding((r_space, e_space), kg)
        q_values = self.forward_batch(observation, action_embeddings, kg)
        q_values = q_values - (1 - action_mask) * utils.HUGE_INT
        return q_values, action_mask

    def predict_action_dist_batch(self, observation: Observation, action_space, kg):
        q_values, action_mask = self.predict_q_value_batch(observation, action_space, kg)
        action_dist = F.softmax(q_values / self.temperature, dim=-1)
        return action_dist, q_values, action_mask

    def calculate_q_values(self, obs: Observation, kg, use_action_space_bucketing=False):
        if use_action_space_bucketing:
            references, next_r, next_e, q_values = [], [], [], []
            db_action_spaces, db_references, db_observations = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b, obs_b in zip(db_action_spaces, db_references, db_observations):
                q_values_b, _ = self.predict_q_value_batch(obs_b, action_space_b, kg)
                references.extend(reference_b)
                q_values.append(q_values_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]

            action_space = utils.pad_and_cat_action_space(kg, db_action_spaces, inv_offset)
            q_values = utils.pad_and_cat(q_values, padding_value=-utils.HUGE_INT)[inv_offset]
        else:
            action_space = get_action_space(obs, kg)
            q_values, _ = self.predict_q_value_batch(obs, action_space, kg)
        q_values *= action_space[1]
        return action_space, q_values


class SACPolicy(BasePolicy):
    def __init__(
        self,
        entity_dim,
        relation_dim,
        history_dim,
        history_num_layers,
        activation_fn,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]],
        n_critics: int = 1,
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.5,
        share_features_extractor: bool = True,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(SACPolicy, self).__init__(
            entity_dim,
            relation_dim,
            history_dim,
            history_num_layers,
            activation_fn,
            net_arch,
            ff_dropout_rate,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
        )
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.history_dim = history_dim
        self.state_dim = entity_dim + relation_dim + history_dim
        self.action_dim = entity_dim + relation_dim

        self.net_arch = net_arch
        actor_arch, critic_arch = self._get_actor_critic_arch(net_arch)

        base_kwargs = {
            'net_arch': actor_arch,
            'action_dim': self.action_dim,
            'history_dim': self.history_dim,
            'state_dim': self.state_dim,
            'ff_dropout_rate': ff_dropout_rate,
            'history_num_layers': history_num_layers,
            'activation_fn': activation_fn,
            'xavier_initialization': xavier_initialization,
            'relation_only': relation_only,
        }

        self.actor_kwargs = base_kwargs.copy()
        self.actor_kwargs.update({
            'action_dropout_rate': action_dropout_rate,
        })

        self.critic_kwargs = base_kwargs.copy()
        self.critic_kwargs.update(
            {
                'n_critics': n_critics,
                'share_feature_extractor': share_features_extractor
            }
        )

        # self.exploration_rate = exploration_rate
        # self.exploration_fraction = exploration_fraction
        # self.exploration_initial_eps = exploration_initial_eps
        # self.exploration_final_eps = exploration_final_eps

        self.share_features_extractor = share_features_extractor

        self._build()

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return utils.get_device("cpu")

    def _get_actor_critic_arch(self, net_arch):
        if isinstance(net_arch, list):
            actor_arch, critic_arch = net_arch, net_arch
        else:
            assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
            assert 'pi' in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
            assert 'qf' in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
            actor_arch, critic_arch = net_arch['pi'], net_arch['qf']
        return actor_arch, critic_arch

    def _build(self,):
        self.actor = self.make_actor()

        if self.share_features_extractor:
            self.critic = self.make_critic(feature_extractor=self.actor.feature_extractor)
        else:
            self.critic = self.make_critic(feature_extractor=None)

        self.critic_target = self.make_critic(feature_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.train(False)

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self, feature_extractor=None) -> Critic:
        critic_kwargs = self.critic_kwargs.copy()
        if feature_extractor is not None:
            critic_kwargs['feature_extractor'] = feature_extractor
        return Critic(**critic_kwargs).to(self.device)

    def predict(self, obs: Observation, kg, use_action_space_bucketing=False, deterministic=True):
        with th.no_grad():
            return self.actor.action_prob(obs, kg, use_action_space_bucketing)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.train(mode)
        self.training = mode


