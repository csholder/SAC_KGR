import torch
import torch as th
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# from src.common.utils import create_mlp, batch_lookup, pad_and_cat, get_device, HUGE_INT, pad_and_cat_action_space
import src.common.utils as utils
from src.common.common_class import Observation
from src.common.common_class import MLPFeaturesExtractor
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
        print('========================== Actor ==========================')
        print('action dim: ', self.action_dim)
        print('state_dim: ', state_dim)
        print('history dim: ', self.history_dim)
        print('activate fn: ', self.activation_fn)
        print('ff_dropout_rate: ', ff_dropout_rate)
        print('history_num_layers: ', self.history_num_layers)
        print('action_dropout_rate: ', self.action_dropout_rate)


class OnPolicy(BasePolicy):
    def __init__(
        self,
        entity_dim,
        relation_dim,
        history_dim,
        history_num_layers,
        activation_fn,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]],
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.5,
        actor_learning_rate: float = 0.001,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(OnPolicy, self).__init__(
            entity_dim,
            relation_dim,
            history_dim,
            history_num_layers,
            activation_fn,
            net_arch=net_arch,
            ff_dropout_rate=ff_dropout_rate,
            actor_learning_rate=actor_learning_rate,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
        )
        actor_arch = self._get_actor_arch(net_arch)

        self.actor_kwargs = {
            'net_arch': actor_arch,
            'action_dim': self.action_dim,
            'history_dim': self.history_dim,
            'state_dim': self.state_dim,
            'ff_dropout_rate': ff_dropout_rate,
            'history_num_layers': history_num_layers,
            'activation_fn': activation_fn,
            'action_dropout_rate': action_dropout_rate,
            'xavier_initialization': xavier_initialization,
            'relation_only': relation_only,
        }

        self.actor_learning_rate = actor_learning_rate
        self._build()

        print('========================== Policy ==========================')
        print('entity_dim: ', entity_dim)
        print('relation_dim: ', relation_dim)
        print('history dim: ', history_dim)
        print('activate fn: ', activation_fn)
        print('ff_dropout_rate: ', ff_dropout_rate)
        print('history_num_layers: ', history_num_layers)
        print('action_dropout_rate: ', action_dropout_rate)
        print('actor_learning_rate: ', actor_learning_rate)

    def _build(self,):
        self.actor = self.make_actor()

    def make_actor(self) -> BaseActor:
        return Actor(**self.actor_kwargs).to(self.device)

    def action_distribution(self, obs, kg, use_action_space_bucketing=False, merge_aspace_batching_outcome=False):
        with th.no_grad():
            return self.actor.action_distribution(obs, kg, use_action_space_bucketing, merge_aspace_batching_outcome)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.training = mode
