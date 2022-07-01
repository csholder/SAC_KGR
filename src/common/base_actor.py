import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import src.common.utils as utils
from src.common.common_class import Observation
from src.common.common_class import MLPFeaturesExtractor
from src.common.policy_utils import *


class BaseActor(nn.Module):
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
        super(BaseActor, self).__init__()
        self.net_arch = net_arch
        self.action_dim = action_dim
        self.history_dim = history_dim
        if isinstance(activation_fn, str):
            if activation_fn.lower() == 'relu':
                activation_fn = nn.ReLU
            elif activation_fn.lower() == 'tanh':
                activation_fn = nn.Tanh
            elif activation_fn.lower() == 'sigmoid':
                activation_fn = nn.Sigmoid
            else:
                raise NotImplementedError
        self.activation_fn = activation_fn
        self.history_num_layers = history_num_layers

        self.feature_extractor = MLPFeaturesExtractor(action_dim, history_dim, state_dim, action_dim,
                                                      history_num_layers, ff_dropout_rate=ff_dropout_rate,
                                                      xavier_initialization=xavier_initialization,
                                                      relation_only=relation_only)

        self.action_dropout_rate = action_dropout_rate

    def policy_fun(self, obs, kg, action_space):
        X2 = self.feature_extractor(obs, kg)
        (r_space, e_space), action_mask = action_space
        A = utils.get_action_embedding((r_space, e_space), kg)
        assert A.shape[-1] == X2.shape[-1]
        action_dist = F.softmax(th.squeeze(A @ th.unsqueeze(X2, 2), 2) - (1 - action_mask) * utils.HUGE_INT, dim=-1)
        return action_dist

    def sample_action(self, db_action_space, db_action_dist):
        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = th.rand(action_dist.size(), device=action_dist.device)
                action_keep_mask = (rand > self.action_dropout_rate).float()
                sample_action_dist = \
                    action_dist * action_keep_mask + utils.EPSILON * (1 - action_keep_mask) * action_mask
                keep_mask = action_keep_mask * action_mask
                if th.any(keep_mask.sum(dim=-1) == 0):
                    zero_row_flag = (keep_mask.sum(dim=-1) == 0).unsqueeze(dim=-1).expand_as(action_keep_mask)
                    sample_action_dist = th.where(zero_row_flag, action_dist, sample_action_dist)
                return sample_action_dist
            else:
                return action_dist

        sample_outcome = {}
        ((r_space, e_space), action_mask) = db_action_space
        sample_action_dist = apply_action_dropout_mask(db_action_dist, action_mask)
        idx = th.multinomial(sample_action_dist, 1, replacement=True)
        next_r = utils.batch_lookup(r_space, idx)
        next_e = utils.batch_lookup(e_space, idx)
        action_prob = utils.batch_lookup(db_action_dist, idx)
        sample_outcome['action_sample'] = (next_r, next_e)
        sample_outcome['action_prob'] = action_prob
        return sample_outcome

    def action_prob(self, obs, kg, use_action_space_bucketing=False):
        if use_action_space_bucketing:
            references, next_r, next_e, action_prob, entropy_list = [], [], [], [], []
            db_action_spaces, db_references, db_observations = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b, obs_b in zip(db_action_spaces, db_references, db_observations):
                action_dist_b = self.policy_fun(obs_b, kg, action_space_b)
                references.extend(reference_b)
                entropy_list.append(utils.entropy(action_dist_b))
                sample_outcome_b = self.sample_action(action_space_b, action_dist_b)
                next_r_b = sample_outcome_b['action_sample'][0]
                next_e_b = sample_outcome_b['action_sample'][1]
                action_prob_b = sample_outcome_b['action_prob']
                next_r.append(next_r_b)
                next_e.append(next_e_b)
                action_prob.append(action_prob_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            next_r = th.cat(next_r, dim=-1)[inv_offset]
            next_e = th.cat(next_e, dim=-1)[inv_offset]
            action_prob = th.cat(action_prob, dim=-1)[inv_offset]
            entropy = th.cat(entropy_list, dim=-1)[inv_offset]
            sample_outcome = {'action_sample': (next_r, next_e), 'action_prob': action_prob, 'entropy': entropy}
        else:
            action_space = get_action_space(obs, kg)
            action_dist = self.policy_fun(obs, kg, action_space)
            entropy = utils.entropy(action_dist)
            sample_outcome = self.sample_action(action_space, action_dist)
            sample_outcome['entropy'] = entropy
        return sample_outcome

    def action_distribution(self, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=True):
        if use_action_space_bucketing:
            references, action_dist, action_space = [], [], None
            db_action_spaces, db_references, db_observations = get_action_space_in_buckets(obs, kg)
            for action_space_b, reference_b, obs_b in zip(db_action_spaces, db_references, db_observations):
                action_dist_b = self.policy_fun(obs_b, kg, action_space_b)
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
            action_dist = self.policy_fun(obs, kg, action_space)
        return action_space, action_dist