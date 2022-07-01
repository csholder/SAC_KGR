import torch
import torch as th
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import src.common.utils as utils
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


class SACPolicy(BasePolicy):
    def __init__(
        self,
        entity_dim,
        relation_dim,
        history_dim,
        history_num_layers,
        activation_fn,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]],
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 1,
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.5,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
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
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
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

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class

        # self.exploration_rate = exploration_rate
        # self.exploration_fraction = exploration_fraction
        # self.exploration_initial_eps = exploration_initial_eps
        # self.exploration_final_eps = exploration_final_eps

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
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
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=self.actor_learning_rate, **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(feature_extractor=self.actor.feature_extractor)
            critic_parameters = [param for name, param in self.critic.named_parameters() if 'feature_extractor' not in name]
        else:
            self.critic = self.make_critic(feature_extractor=None)
            critic_parameters = self.critic.parameters()
        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=self.critic_learning_rate, **self.optimizer_kwargs)

        self.critic_target = self.make_critic(feature_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.train(False)

        # self.exploration_schedule = utils.get_linear_fn(
        #     self.exploration_initial_eps,
        #     self.exploration_final_eps,
        #     self.exploration_fraction,
        # )

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


