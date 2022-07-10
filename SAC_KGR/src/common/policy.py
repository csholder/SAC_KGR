import torch as th
import torch.nn as nn

from typing import Type, List, Tuple, Optional, Dict, Any

from src.common.base_actor import BaseActor
from src.common.base_critic import BaseCritic
from src.common.common_class import Observation
from src.common import utils


class BasePolicy(nn.Module):
    def __init__(
        self,
        entity_dim,
        relation_dim,
        history_dim,
        history_num_layers,
        activation_fn,
        net_arch,
        ff_dropout_rate,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(BasePolicy, self).__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.history_dim = history_dim
        if relation_only:
            self.state_dim = relation_dim + history_dim
        else:
            self.state_dim = entity_dim + relation_dim + history_dim
        self.action_dim = entity_dim + relation_dim
        self.history_num_layers = history_num_layers
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        self.base_kwargs = {
            'net_arch': net_arch,
            'action_dim': self.action_dim,
            'history_dim': self.history_dim,
            'state_dim': self.state_dim,
            'ff_dropout_rate': ff_dropout_rate,
            'history_num_layers': history_num_layers,
            'activation_fn': activation_fn,
            'xavier_initialization': xavier_initialization,
            'relation_only': relation_only,
        }

        self.actor_kwargs, self.critic_kwargs = None, None

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return utils.get_device("cpu")

    def _get_actor_arch(self, net_arch):
        if isinstance(net_arch, list):
            actor_arch = net_arch
        else:
            assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
            assert 'pi' in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
            actor_arch = net_arch['pi']
        return actor_arch

    def _get_critic_arch(self, net_arch):
        if isinstance(net_arch, list):
            critic_arch = net_arch
        else:
            assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
            assert 'qf' in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
            critic_arch = net_arch['qf']
        return critic_arch

    def _build(self):
        raise NotImplementedError

    def make_actor(self) -> BaseActor:
        raise BaseActor(**self.actor_kwargs).to(self.device)

    def make_critic(self) -> BaseCritic:
        raise BaseCritic(**self.critic_kwargs).to(self.device)

    # def evaluate_action(self, obs: Observation, action: th.Tensor, kg):
    #     raise NotImplementedError

    # def sample_action(self, obs: Observation, kg, use_action_space_bucketing=False):
    #     raise NotImplementedError

    def predict(self, obs: Observation, kg, use_action_space_bucketing=False, deterministic=True):
        raise NotImplementedError
