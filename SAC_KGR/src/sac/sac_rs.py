from typing import Any, Dict, List, Optional, Tuple, Type, Union
from logging import Logger
import os
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import defaultdict as ddict

# from src.common.off_policy_algorithm import OffPolicyAlgorithm
from src.sac.sac import SAC
from src.common.buffers import ReplayBuffer
from src.sac.policy import SACPolicy
from src.common.knowledge_graph import KnowledgeGraph
import src.common.utils as utils
from src.emb.fact_network import get_conve_nn_state_dict, get_conve_kg_state_dict


class SACrs(SAC):
    def __init__(
        self,
        args,
        kg: KnowledgeGraph,
        entity_dim: int,
        relation_dim: int,
        history_dim: int,
        history_num_layers: int = 3,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        eof_learning_rate: float = 3e-4,
        buffer_size: int = 100,
        batch_size: int = 256,
        num_rollout_steps: int = 3,
        learning_starts: int = 100,
        action_dropout_rate: float = 0.1,
        net_arch: List[int] = [64, 64],
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = 'auto',
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = 'auto',
        replay_buffer_class: Union[str, ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        policy_class: Union[str, SACPolicy] = None,
        mu: float = 1.0,
        fn_kg: KnowledgeGraph = None,
        fn=None,
        verbose: int = 0,
    ):
        super(SACrs, self).__init__(
            args,
            kg=kg,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            history_dim=history_dim,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            eof_learning_rate=eof_learning_rate,
            buffer_size=buffer_size,
            history_num_layers=history_num_layers,
            num_rollout_steps=num_rollout_steps,
            batch_size=batch_size,
            learning_starts=learning_starts,
            action_dropout_rate=action_dropout_rate,
            net_arch=net_arch,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_class=policy_class,
            verbose=verbose,
        )
        self.mu = mu
        self.fn_kg = fn_kg
        self.fn = fn
        if self.fn_model == 'conve':
            fn_state_dict = th.load(args.conve_state_dict_path)
            fn_nn_state_dict = get_conve_nn_state_dict(fn_state_dict)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
            self.fn.load_state_dict(fn_nn_state_dict)
        else:
            raise NotImplementedError
        self.fn_kg.load_state_dict(fn_kg_state_dict)
        self.logger.info('load fn and fn_kg from {}...'.format(args.conve_state_dict_path))

        self.fn.eval()
        self.fn_kg.eval()
        utils.detach_module(self.fn)
        utils.detach_module(self.fn_kg)

    def reward_fun(self, e1, r, e2, pred_e2):
        real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg).squeeze(1)
        binary_reward = (pred_e2 == e2).float()
        return binary_reward + self.mu * (1 - binary_reward) * real_reward

    @property
    def fn_model(self):
        return self.model_name.split('.')[3]
