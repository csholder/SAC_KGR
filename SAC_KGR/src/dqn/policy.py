import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type, Dict, Any, Optional, Union, Tuple
import numpy as np

from src.common import utils
from src.common.common_class import MLPFeaturesExtractor, Observation
from src.common.policy_utils import *
from src.common.base_critic import BaseCritic
from src.common.policy import BasePolicy


class QNetwork(BaseCritic):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

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
        temperature: float = 1.0,
        n_critics: int = 1,
        xavier_initialization: bool = True,
        relation_only: bool = True,
    ):
        super(QNetwork, self).__init__(
            net_arch=net_arch,
            action_dim=action_dim,
            history_dim=history_dim,
            state_dim=state_dim,
            ff_dropout_rate=ff_dropout_rate,
            history_num_layers=history_num_layers,
            activation_fn=activation_fn,
            n_critics=n_critics, 
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
        )
        self.action_dropout_rate = action_dropout_rate
        self.temperature = temperature

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


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        entity_dim,
        relation_dim,
        history_dim,
        history_num_layers,
        activation_fn,
        n_critics,
        ff_dropout_rate,
        critic_learning_rate: float = 0.001,
        action_dropout_rate: float = 0.1,
        exploration_rate: float = 0.1,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        boltzmann_exploration: bool = False,
        temperature: float = 1.0,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = [64, 64],
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        xavier_initialization: bool = True,
        relation_only: bool = True,
    ):
        super(DQNPolicy, self).__init__(
            entity_dim,
            relation_dim,
            history_dim,
            history_num_layers,
            activation_fn,
            net_arch,
            ff_dropout_rate,
            critic_learning_rate=critic_learning_rate,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
        )
        self.action_dropout_rate = action_dropout_rate
        self.exploration_rate = exploration_rate
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.boltzmann_exploration = boltzmann_exploration
        self.temperature = temperature

        self.optimizer_class = optimizer_class

        self.critic_kwargs = self.base_kwargs.copy()
        self.critic_kwargs.update({
            "action_dropout_rate": action_dropout_rate,
            "n_critics": n_critics,
            'temperature': temperature,
        })

        self.q_net, self.q_net_target = None, None
        self._build()

    def _build(self) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.critic_learning_rate, **self.optimizer_kwargs)

        self.exploration_schedule = utils.get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        return QNetwork(**self.critic_kwargs).to(self.device)

    def _update_exploration(self, current_progress_remaining):
        self.exploration_rate = self.exploration_schedule(current_progress_remaining)

    def evaluate_action(self, obs: Observation, action: th.Tensor, kg):
        q_value = self.q_net.predict_q_value(obs, action, kg)
        return q_value

    def sample_action(self, obs: Observation, kg, use_action_space_bucketing=False, apply_action_dropout=True):
        def apply_dist_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = th.rand(action_dist.size(), device=action_dist.device)
                action_keep_mask = (rand > self.action_dropout_rate).float()
                sample_action_dist = \
                    action_dist * action_keep_mask + utils.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def apply_q_value_action_dropout_mask(q_values, action_mask):
            if self.action_dropout_rate > 0:
                rand = th.rand(q_values.size(), device=q_values.device)
                action_keep_mask = (rand > self.action_dropout_rate).float()
                keep_mask = action_keep_mask * action_mask
                if th.any(keep_mask.sum(dim=-1) == 0):
                    zero_row_flag = (keep_mask.sum(dim=-1) == 0).unsqueeze(dim=-1).expand_as(action_keep_mask)
                    keep_mask = th.where(zero_row_flag, action_mask, keep_mask)
                sample_q_values = q_values - utils.HUGE_INT * (1 - keep_mask)
                return sample_q_values
            else:
                return q_values

        if use_action_space_bucketing:
            references, next_r, next_e, q_values = [], [], [], []
            db_action_spaces, db_references, db_observations = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b, obs_b in zip(db_action_spaces, db_references, db_observations):
                action_dist_b, q_values_b, action_mask_b = self.q_net.predict_action_dist_batch(obs_b, action_space_b, kg)
                if apply_action_dropout:
                    q_values_b = apply_q_value_action_dropout_mask(q_values_b, action_mask_b)
                # Greedy action
                q_value_b, action_indice_b = q_values_b.max(dim=1, keepdim=True)
                action_r_b = utils.batch_lookup(action_space_b[0][0], action_indice_b)
                action_e_b = utils.batch_lookup(action_space_b[0][1], action_indice_b)
                references.extend(reference_b)
                q_values.append(q_value_b)
                next_r.append(action_r_b)
                next_e.append(action_e_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            next_r = th.cat(next_r, dim=-1)[inv_offset]
            next_e = th.cat(next_e, dim=-1)[inv_offset]
            q_values_ = th.cat(q_values, dim=0)[inv_offset]
        else:
            action_space = get_action_space(obs, kg)
            action_dist, q_values, action_mask = self.q_net.predict_action_dist_batch(obs, action_space, kg)
            if not self.boltzmann_exploration:
                if apply_action_dropout:
                    q_values = apply_q_value_action_dropout_mask(q_values, action_mask)
                q_values_, action_indice = q_values.max(dim=1, keepdim=True)
            else:
                if apply_action_dropout:
                    action_dist = apply_dist_action_dropout_mask(action_dist, action_mask)
                action_indice = th.multinomial(action_dist, num_samples=1)
                q_values_ = utils.batch_lookup(q_values, action_indice)
            next_r = utils.batch_lookup(action_space[0][0], action_indice)
            next_e = utils.batch_lookup(action_space[0][1], action_indice)
        return {'action_sample': (next_r, next_e), 'q_values': q_values_.reshape(-1)}

    def random_sample_action(self, obs: Observation, kg, use_action_space_bucketing=False):
        def apply_sample_with_mask(action_dim, action_mask):
            actions_n = action_mask.sum(dim=-1)
            mean_values = 1. / actions_n
            sample_prob = mean_values.unsqueeze(dim=-1).repeat(1, action_dim)
            sample_prob *= action_mask
            sample_prob = th.distributions.Categorical(sample_prob)
            return sample_prob.sample().reshape(-1, 1)

        if use_action_space_bucketing:
            references, next_r, next_e = [], [], []
            db_action_spaces, db_references, _ = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                (action_r, action_e), action_mask = action_space_b
                action_indice = apply_sample_with_mask(action_r.shape[-1], action_mask)
                references.extend(reference_b)
                next_r.append(utils.batch_lookup(action_r, idx=action_indice))
                next_e.append(utils.batch_lookup(action_e, idx=action_indice))
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            next_r = th.cat(next_r, dim=-1)[inv_offset]
            next_e = th.cat(next_e, dim=-1)[inv_offset]
        else:
            action_space = get_action_space(obs, kg)
            (action_r, action_e), action_mask = action_space
            action_indice = apply_sample_with_mask(action_r.shape[-1], action_mask)
            next_r = utils.batch_lookup(action_r, idx=action_indice)
            next_e = utils.batch_lookup(action_e, idx=action_indice)
        return {'action_sample': (next_r, next_e)}

    def predict(self, obs: Observation, kg, use_action_space_bucketing=False, deterministic=True):
        with th.no_grad():
            if (not deterministic) and (not self.boltzmann_exploration) and np.random.rand() < self.exploration_rate:
                sample_outcome = self.random_sample_action(obs, kg, use_action_space_bucketing)
            else:
                sample_outcome = self.sample_action(obs, kg, use_action_space_bucketing)
        return sample_outcome

    def calculate_q_values(self, obs: Observation, kg, use_action_space_bucketing=False):
        if use_action_space_bucketing:
            references, next_r, next_e, q_values = [], [], [], []
            db_action_spaces, db_references, db_observations = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b, obs_b in zip(db_action_spaces, db_references, db_observations):
                q_values_b, _ = self.q_net.predict_q_value_batch(obs_b, action_space_b, kg)
                references.extend(reference_b)
                q_values.append(q_values_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]

            action_space = utils.pad_and_cat_action_space(kg, db_action_spaces, inv_offset)
            q_values = utils.pad_and_cat(q_values, padding_value=-utils.HUGE_INT)[inv_offset]
        else:
            action_space = get_action_space(obs, kg)
            q_values, _ = self.q_net.predict_q_value_batch(obs, action_space, kg)
        return action_space, q_values

    def calculate_action_dist(self, obs: Observation, kg, use_action_space_bucketing=False,
                              merge_aspace_batching_outcome=False):
        if use_action_space_bucketing:
            references, next_r, next_e, action_dist = [], [], [], []
            db_action_spaces, db_references, db_observations = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b, obs_b in zip(db_action_spaces, db_references, db_observations):
                action_dist_b, _, _ = self.q_net.predict_action_dist_batch(obs_b, action_space_b, kg)
                references.extend(reference_b)
                action_dist.append(action_dist_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            if merge_aspace_batching_outcome:
                action_space = utils.pad_and_cat_action_space(kg, db_action_spaces, inv_offset)
                action_dist = utils.pad_and_cat(action_dist, padding_value=0)[inv_offset]
            else:
                action_space = db_action_spaces
        else:
            action_space = get_action_space(obs, kg)
            action_dist, _, _ = self.q_net.predict_action_dist_batch(obs, action_space, kg)
        return action_space, action_dist

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.train(mode)