import torch as th
import torch.nn.functional as F
from typing import List, Tuple, Union, Type, Optional, Any, Dict
import numpy as np
import wandb

from src.dqn.policy import DQNPolicy
from src.common.off_policy_algorithm import OffPolicyAlgorithm
from src.common.knowledge_graph import KnowledgeGraph
from src.common.buffers import ReplayBuffer
from src.common import utils
from src.common.common_class import Observation
from src.dqn.dqn import DQN
from src.emb.fact_network import get_conve_kg_state_dict, get_conve_nn_state_dict


class RewardShapingDQN(DQN):
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
        learning_rate: float = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        buffer_batch_size: int = 32,
        magnification: int = 1,
        learning_starts: int = 100,
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.1,
        reward_shaping_threshold: float = 0.0,
        net_arch: List[int] = [64, 64],
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        n_critics: int = 1,
        target_update_interval: int = 1,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 0.5,
        exploration_final_eps: float = 0.05,
        boltzmann_exploration: bool = False,
        temperature: float = 1.0,
        max_grad_norm: float = 0.,
        replay_buffer_class: Union[str, ReplayBuffer] = None,
        policy_class: Union[str, DQNPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        xavier_initialization: bool = True,
        relation_only: bool = True,
        deterministic: bool = False,
        verbose: int = 0,
        _init_setup_model: bool = True,
        beam_search_with_q_value: bool = True,
        target_net_dropout: bool = False,
    ):
        super(RewardShapingDQN, self).__init__(
            args,
            kg,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            history_dim=history_dim,
            history_num_layers=history_num_layers,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            buffer_batch_size=buffer_batch_size,
            magnification=magnification,
            learning_starts=learning_starts,
            ff_dropout_rate=ff_dropout_rate,
            action_dropout_rate=action_dropout_rate,
            net_arch=net_arch,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            n_critics=n_critics,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            boltzmann_exploration=boltzmann_exploration,
            temperature=temperature,
            max_grad_norm=max_grad_norm,
            replay_buffer_class=replay_buffer_class,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
            deterministic=deterministic,
            verbose=verbose,
            _init_setup_model=_init_setup_model,
            beam_search_with_q_value=beam_search_with_q_value,
            target_net_dropout=target_net_dropout,
        )
        self.fn = fn
        self.fn_kg = fn_kg
        self.mu = args.mu

        self.reward_shaping_threshold = reward_shaping_threshold

        fn_state_dict = th.load(args.conve_state_dict_path)
        fn_nn_state_dict = get_conve_nn_state_dict(fn_state_dict)
        fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
        self.fn.load_state_dict(fn_nn_state_dict)
        self.fn_kg.load_state_dict(fn_kg_state_dict)

        self.fn.eval()
        self.fn_kg.eval()
        utils.detach_module(self.fn)
        utils.detach_module(self.fn_kg)

    def reward_fun(self, e1, r, e2, pred_e2):
        real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg).squeeze(1)
        real_reward_mask = (real_reward > self.reward_shaping_threshold).float()
        real_reward *= real_reward_mask
        binary_reward = (pred_e2 == e2).float()
        return binary_reward + self.mu * (1 - binary_reward) * real_reward