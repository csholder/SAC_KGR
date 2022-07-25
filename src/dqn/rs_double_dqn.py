import torch as th
import torch.nn.functional as F
from typing import List, Tuple, Union, Type, Optional, Any, Dict
import numpy as np
import wandb

from src.dqn.policy import DQNPolicy
from src.dqn.rs_dqn import RewardShapingDQN
from src.common.knowledge_graph import KnowledgeGraph
from src.common.buffers import ReplayBuffer
from src.common import utils
from src.common.common_class import Observation


class DoubleDQN(RewardShapingDQN):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

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
        lr_scheduler_step: int = 5,
        lr_decay_gamma: float = 0.75,
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
        super(DoubleDQN, self).__init__(
            args,
            kg,
            fn_kg,
            fn,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            history_dim=history_dim,
            history_num_layers=history_num_layers,
            learning_rate=learning_rate,
            lr_scheduler_step=lr_scheduler_step,
            lr_decay_gamma=lr_decay_gamma,
            buffer_size=buffer_size,
            buffer_batch_size=buffer_batch_size,
            magnification=magnification,
            learning_starts=learning_starts,
            ff_dropout_rate=ff_dropout_rate,
            action_dropout_rate=action_dropout_rate,
            reward_shaping_threshold=reward_shaping_threshold,
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

    def do_train(self, gradient_steps: int, batch_size: int = 100):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        # self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            with th.no_grad():
                # Compute the next Q-values using the target network
                sample_outcome = self.q_net.sample_action(replay_data.next_observation, self.kg,
                                                          use_action_space_bucketing=self.use_action_space_bucketing,
                                                          apply_action_dropout=self.target_net_dropout)
                sample_action = sample_outcome['action_sample']
                next_q_values = self.policy.evaluate_action(replay_data.next_observation, action=sample_action,
                                                            kg=self.kg)

                next_q_values = next_q_values.reshape(-1)
                # 1-step TD target
                target_q_values = replay_data.reward + (
                        1 - replay_data.next_observation.done) * self.gamma * next_q_values

            # Get current Q-values estimates
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = self.policy.evaluate_action(replay_data.observation, action=replay_data.action,
                                                           kg=self.kg)

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm > 0:
                th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self._on_step()

        # Increase update counter
        self._n_updates += gradient_steps

        loss_dict = {}
        loss_dict['n_updates'] = self._n_updates
        loss_dict['print_loss'] = np.mean(losses)
        if self.args.use_wandb:
            wandb.log({'loss': np.mean(losses)})

        return loss_dict
