import io
import pathlib
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch as th

from src.sac.policy import SACPolicy
from src.dqn.policy import DQNPolicy, DuelDQNPolicy
from src.common.buffers import ReplayBuffer
from src.common.common_class import TrainFreq, TrainFrequencyUnit, RolloutReturn, Observation
from src.common.save_utils import save_to_pkl, load_from_pkl
import src.common.utils as utils
from src.common.knowledge_graph import KnowledgeGraph
from src.learn_framework import LFramework


class OffPolicyAlgorithm(LFramework):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        args,
        kg: KnowledgeGraph,
        entity_dim: int,
        relation_dim: int,
        history_dim: int,
        history_num_layers: int,
        ff_dropout_rate: float,
        critic_learning_rate: float = 0.001,
        xavier_initialization: bool = True,
        relation_only: bool = False,
        buffer_size: int = 1_000_000,  # 1e6
        buffer_batch_size: int = 256,
        magnification: int = 1,
        learning_starts: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        n_critics: int = 1,
        replay_buffer_class: Union[str, ReplayBuffer] = None,
        policy_class: Union[str, SACPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        deterministic=True,
        verbose: int = 0,
    ):
        super(OffPolicyAlgorithm, self).__init__(args, kg)
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.history_dim = history_dim
        self.history_num_layers = history_num_layers
        self.ff_dropout_rate = ff_dropout_rate
        self.critic_learning_rate = critic_learning_rate
        self.xavier_initialization = xavier_initialization
        self.relation_only = relation_only

        self.buffer_size = buffer_size
        self.buffer_batch_size = buffer_batch_size
        self.magnification = magnification
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps

        self._episode_storage = None

        self.n_critics = n_critics
        self.replay_buffer_class = replay_buffer_class
        self.policy_class = policy_class
        self.deterministic = deterministic
        self.verbose = verbose

        # Save train freq parameter, will be converted later to TrainFreq object
        if train_freq[1] == 'episode':
            self.num_rollouts = train_freq[0]
            train_freq = (1, train_freq[1])
        self.train_freq = train_freq
        assert buffer_size % (self.num_rollouts * self.buffer_batch_size) == 0

        self.num_timesteps = 0
        self._episode_num = 0
        self._n_updates = 0

        # self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]

        # Update policy keyword arguments
        self.policy_kwargs = policy_kwargs
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                'entity_dim': entity_dim,
                'relation_dim': relation_dim,
                'history_dim': self.history_dim,
                'history_num_layers': self.history_num_layers,
                'activation_fn': th.nn.Tanh,
                'n_critics': self.n_critics,
                'ff_dropout_rate': self.ff_dropout_rate,
                'xavier_initialization': self.xavier_initialization,
                'relation_only': self.relation_only,
            }

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        # Use DictReplayBuffer if needed
        # self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            if self.replay_buffer_class == 'ReplayBuffer':
                self.replay_buffer = ReplayBuffer(
                    self.buffer_size,
                    self.num_rollout_steps,
                    self.device,
                )
            else:
                raise NotImplementedError

        if self.policy_class == 'SACPolicy':
            self.policy = SACPolicy(  # pytype:disable=not-instantiable
                **self.policy_kwargs,  # pytype:disable=not-instantiable
            )
        elif self.policy_class == 'DQNPolicy':
            self.policy = DQNPolicy(
                **self.policy_kwargs,
            )
        elif self.policy_class == 'DuelDQNPolicy':
            self.policy = DuelDQNPolicy(
                **self.policy_kwargs,
            )
        else:
            raise NotImplementedError
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def _setup_learn(
        self,
        mini_batch,
    ):
        start_e, query_r, target_e = mini_batch
        start_r = th.full(start_e.size(), self.kg.dummy_start_r, device=self.device, dtype=th.long)
        path_r = th.full((len(start_e), self.num_rollout_steps + 1), self.kg.dummy_r, device=self.device, dtype=th.long)
        path_r[:, 0] = start_r
        path_e = th.full((len(start_e), self.num_rollout_steps + 1), self.kg.dummy_e, device=self.device, dtype=th.long)
        path_e[:, 0] = start_e
        init_obs = Observation(
            num_rollout_steps=self.num_rollout_steps,
            query_relation=query_r,
            target_entity=target_e,
            path=(path_r, path_e),
            path_length=torch.zeros_like(start_r).float()
        )

        return init_obs

    def learn(
        self,
        mini_batch,
    ):
        init_obs = self._setup_learn(
            mini_batch,
        )

        rollout = self.collect_rollouts(
            start_observation=init_obs,
            train_freq=self.train_freq,
            replay_buffer=self.replay_buffer,
        )

        loss_dict = {}
        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            magnification = self.magnification * len(self.replay_buffer) / self.buffer_size
            gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
            gradient_steps = int((1 + magnification) * gradient_steps)
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                buffer_batch_size = int((1 + magnification) * self.buffer_batch_size)
                loss_dict = self.do_train(batch_size=buffer_batch_size, gradient_steps=gradient_steps)

        return loss_dict

    def do_train(self, gradient_steps: int, batch_size: int):
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def sample_action(
        self,
        observation: Observation,
        deterministic=True,
    ):
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, self.kg, self.use_action_space_bucketing, deterministic=deterministic)

    def _sample_next_step(
        self,
    ):
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action according to policy
        # Note: when using continuous actions,
        # we assume that the policy uses tanh to scale the action
        # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        # sample_outcome = self.policy.predict(self._last_obs, self.kg, self.use_action_space_bucketing,
        #                                      self.deterministic)
        sample_outcome = self.sample_action(self._last_obs, deterministic=self.deterministic)
        return sample_outcome['action_sample']

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        new_obs: Observation,
        reward: th.Tensor,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode

        replay_buffer.add(
            new_obs.path_e,
            new_obs.path_r,
            new_obs.query_relation,
            new_obs.target_entity,
            reward,
            new_obs.done,
            new_obs.path_length,
        )

        self._last_obs = new_obs

    def _step(
        self,
        last_obs: Observation,
        action,
    ):
        new_obs_path_length = last_obs.path_length + 1
        path_r, path_e = last_obs.path_r, last_obs.path_e
        # path_r[:, new_obs_path_length] = action[0]
        path_r = path_r.scatter(1, new_obs_path_length.unsqueeze(dim=-1).long(), action[0].unsqueeze(dim=-1))
        # path_e[:, new_obs_path_length] = action[1]
        path_e = path_e.scatter(1, new_obs_path_length.unsqueeze(dim=-1).long(), action[1].unsqueeze(dim=-1))
        new_obs = Observation(
            num_rollout_steps=self.num_rollout_steps,
            query_relation=last_obs.query_relation,
            target_entity=last_obs.target_entity,
            path=(path_r, path_e),
            path_length=new_obs_path_length,
        )
        done_mask = new_obs.done
        reward = self.reward_fun(new_obs.start_entity, new_obs.query_relation, new_obs.target_entity, action[1])
        reward *= done_mask
        return new_obs, reward

    def collect_rollouts(
        self,
        start_observation: Observation,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        # Vectorize action noise if needed
        continue_training = True
        self._last_obs = start_observation

        while utils.should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions = self._sample_next_step()
            # Rescale and perform action
            new_obs, rewards = self._step(self._last_obs, actions)

            self.num_timesteps += 1
            num_collected_steps += 1

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, new_obs, rewards)

            # self._on_step()

            if torch.all(new_obs.done.bool()):
                if self.run_analysis:
                    current_rewards = (rewards == 1.).float()
                    if self.rewards is None:
                        self.rewards = current_rewards
                    else:
                        self.rewards = torch.cat([self.rewards, current_rewards])

                    self.record_path_trace((self._last_obs.path_r, self._last_obs.path_e))
                # Update stats
                num_collected_episodes += 1
                self._episode_num += 1
                self._last_obs = start_observation

        return RolloutReturn(num_collected_steps, num_collected_episodes, continue_training)

    def beam_search(
        self,
        mini_batch,
        beam_size: int,
        save_beam_search_paths=False,
    ):
        raise NotImplementedError

    def record_path_trace(self, path_trace):
        path_length = path_trace[0].size(1) * 2
        flattened_path_trace = [x.unsqueeze(dim=-1) for x in path_trace]
        path_trace_mat = torch.cat(flattened_path_trace, dim=-1).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]