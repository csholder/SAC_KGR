"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Policy gradient (REINFORCE algorithm) training and inference.
"""

import torch
from typing import List, Union

from src.learn_framework import LFramework
from src.pg.policy import OnPolicy
from src.common import utils
from src.common.common_class import Observation
from src.emb.fact_network import get_conve_kg_state_dict, get_conve_nn_state_dict
from src.common.knowledge_graph import KnowledgeGraph
from src.pg.pg import PolicyGradient


class RewardShapingPolicyGradient(PolicyGradient):
    def __init__(
        self,
        args,
        kg: KnowledgeGraph,
        fn_kg,
        fn,
        entity_dim: int,
        relation_dim: int,
        history_dim: int,
        history_num_layers: int = 3,
        actor_learning_rate: float = 3e-4,
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.1,
        net_arch: List[int] = [64, 64],
        policy_class: Union[str, OnPolicy] = None,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(RewardShapingPolicyGradient, self).__init__(
            args,
            kg,
            entity_dim,
            relation_dim,
            history_dim,
            history_num_layers,
            actor_learning_rate,
            ff_dropout_rate,
            action_dropout_rate,
            net_arch,
            policy_class=policy_class,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
        )

        self.reward_shaping_threshold = args.reward_shaping_threshold

        self.fn_kg = fn_kg
        self.fn = fn
        self.mu = args.mu

        fn_state_dict = torch.load(args.conve_state_dict_path)
        fn_nn_state_dict = get_conve_nn_state_dict(fn_state_dict)
        fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
        self.fn.load_state_dict(fn_nn_state_dict)
        self.fn_kg.load_state_dict(fn_kg_state_dict)

        self.fn.eval()
        self.fn_kg.eval()
        utils.detach_module(self.fn)
        utils.detach_module(self.fn_kg)

    def reward_fun(self, e1, r, e2, pred_e2, path_trace=None):
        real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg).squeeze(1)
        real_reward_mask = (real_reward > self.reward_shaping_threshold).float()
        real_reward *= real_reward_mask
        # print('real reward: ', real_reward.sum().detach().cpu().item())
        binary_reward = (pred_e2 == e2).float()
        return binary_reward + self.mu * (1 - binary_reward) * real_reward, binary_reward
