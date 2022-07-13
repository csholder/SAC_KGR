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


class DQN(OffPolicyAlgorithm):
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
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        xavier_initialization: bool = True,
        relation_only: bool = True,
        deterministic: bool = False,
        verbose: int = 0,
        _init_setup_model: bool = True,
        beam_search_with_q_value: bool = True,
        target_net_dropout: bool = False,
    ):
        super(DQN, self).__init__(
            args,
            kg,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            history_dim=history_dim,
            history_num_layers=history_num_layers,
            ff_dropout_rate=ff_dropout_rate,
            critic_learning_rate=learning_rate,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
            buffer_size=buffer_size,
            buffer_batch_size=buffer_batch_size,
            magnification=magnification,
            learning_starts=learning_starts,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            n_critics=n_critics,
            replay_buffer_class=replay_buffer_class,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.history_dim = history_dim
        self.history_num_layers = history_num_layers
        self.action_dropout_rate = action_dropout_rate
        self.net_arch = net_arch

        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = exploration_initial_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.boltzmann_exploration = boltzmann_exploration
        self.temperature = temperature

        self.optimizer_class = optimizer_class
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs

        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        self.beam_search_with_q_value = beam_search_with_q_value
        self.target_net_dropout = target_net_dropout

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.policy_kwargs.update({
            'action_dropout_rate': self.action_dropout_rate,
            'net_arch': self.net_arch,
            'exploration_rate': self.exploration_rate,
            'exploration_fraction': self.exploration_fraction,
            'exploration_initial_eps': self.exploration_initial_eps,
            'exploration_final_eps': self.exploration_final_eps,
            'boltzmann_exploration': self.boltzmann_exploration,
            'temperature': self.temperature,
        })
        super(DQN, self)._setup_model()
        self._create_aliases()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.parameters()),
                                              lr=self.critic_learning_rate, **self.optimizer_kwargs)

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

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
                sample_outcome = self.q_net_target.sample_action(replay_data.next_observation, self.kg,
                                                                 use_action_space_bucketing=self.use_action_space_bucketing,
                                                                 apply_action_dropout=self.target_net_dropout)
                next_q_values = sample_outcome['q_values']
                
                # _, q_values = self.policy.calculate_q_values(replay_data.next_observation, self.kg,
                #                                              use_action_space_bucketing=self.use_action_space_bucketing)
                # next_q_values, _ = q_values.max(dim=-1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1)
                # 1-step TD target
                target_q_values = replay_data.reward + (1 - replay_data.next_observation.done) * self.gamma * next_q_values

            # Get current Q-values estimates
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = self.policy.evaluate_action(replay_data.observation, action=replay_data.action, kg=self.kg)

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

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            utils.polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

    def learn(
        self,
        mini_batch,
    ):
        if self.args.use_wandb:
            wandb.log({'exploration rate': self.policy.exploration_rate})
        return super(DQN, self).learn(
            mini_batch,
        )

    def predict(
        self,
        mini_batch,
        beam_size,
        verbose=False,
        query_path_dict=None
    ):
        self.eval()
        with th.no_grad():
            e1, e2, r = mini_batch
            if self.beam_search_with_q_value:
                beam_search_output = self.beam_search_q_value(mini_batch, beam_size)
            else:
                beam_search_output = self.beam_search_probability(mini_batch, beam_size)
            pred_e2s = beam_search_output['pred_e2s']
            pred_e2_scores = beam_search_output['pred_e2_scores']
            if verbose:
                # print inference paths
                search_traces = beam_search_output['search_traces']
                output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
                for i in range(len(e1)):
                    for j in range(output_beam_size):
                        ind = i * output_beam_size + j
                        if pred_e2s[i][j] == self.kg.dummy_e:
                            break
                        search_trace = []
                        for k in range(len(search_traces)):
                            search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                        path_str = utils.format_path(search_trace, self.kg)
                        if query_path_dict is not None:
                            query_path_dict[(int(e1[i]), int(e2[i]), int(r[i]))].append(
                                [float(pred_e2_scores[i][j]), path_str])

            pred_scores = th.zeros([len(e1), self.kg.num_entities], device=self.device)
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = pred_e2_scores[i]
        return pred_scores, query_path_dict

    def beam_search_q_value(
        self,
        mini_batch,
        beam_size: int,
        save_beam_search_paths=False,
    ):
        e_s, q, e_t = mini_batch
        batch_size = len(e_s)

        def top_k_action(q_values, action_space):
            full_size = len(q_values)
            assert (full_size % batch_size == 0)
            last_k = int(full_size / batch_size)

            (r_space, e_space), _ = action_space
            action_space_size = r_space.size()[1]
            # => [batch_size, k'*action_space_size]
            q_values = q_values.view(batch_size, -1)
            beam_action_space_size = q_values.size()[1]
            k = min(beam_size, beam_action_space_size)
            # [batch_size, k]
            q_value, action_ind = th.topk(q_values, k)
            next_r = utils.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
            next_e = utils.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
            # [batch_size, k] => [batch_size*k]
            q_value = q_value.view(-1)
            # *** compute parent offset
            # [batch_size, k]
            action_beam_offset = action_ind // action_space_size
            # [batch_size, 1]
            action_batch_offset = th.arange(batch_size, device=self.device).unsqueeze(1) * last_k
            # [batch_size, k] => [batch_size*k]
            action_offset = (action_batch_offset + action_beam_offset).view(-1)
            return (next_r, next_e), q_value, action_offset

        def top_k_answer_unique(q_values, action_space):
            full_size = len(q_values)
            assert (full_size % batch_size == 0)
            last_k = int(full_size / batch_size)
            (r_space, e_space), _ = action_space
            action_space_size = r_space.size()[1]

            r_space = r_space.view(batch_size, -1)
            e_space = e_space.view(batch_size, -1)
            q_values = q_values.view(batch_size, -1)
            beam_action_space_size = q_values.size()[1]
            assert (beam_action_space_size % action_space_size == 0)
            k = min(beam_size, beam_action_space_size)
            next_r_list, next_e_list = [], []
            q_value_list = []
            action_offset_list = []
            for i in range(batch_size):
                q_values_b = q_values[i]
                r_space_b = r_space[i]
                e_space_b = e_space[i]
                unique_e_space_b = th.unique(e_space_b.data.cpu()).to(device=self.device)
                unique_q_value_dist, unique_idx = utils.unique_max(unique_e_space_b, e_space_b, q_values_b)
                k_prime = min(len(unique_e_space_b), k)
                top_unique_q_value_dist, top_unique_idx2 = th.topk(unique_q_value_dist, k_prime)
                top_unique_idx = unique_idx[top_unique_idx2]
                top_unique_beam_offset = top_unique_idx // action_space_size
                top_r = r_space_b[top_unique_idx]
                top_e = e_space_b[top_unique_idx]
                next_r_list.append(top_r.unsqueeze(0))
                next_e_list.append(top_e.unsqueeze(0))
                q_value_list.append(top_unique_q_value_dist.unsqueeze(0))
                top_unique_batch_offset = i * last_k
                top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
                action_offset_list.append(top_unique_action_offset.unsqueeze(0))
            next_r = utils.pad_and_cat(next_r_list, padding_value=self.kg.dummy_r).view(-1)
            next_e = utils.pad_and_cat(next_e_list, padding_value=self.kg.dummy_e).view(-1)
            q_value = utils.pad_and_cat(q_value_list, padding_value=-utils.HUGE_INT)
            action_offset = utils.pad_and_cat(action_offset_list, padding_value=-1)
            return (next_r, next_e), q_value.view(-1), action_offset.view(-1)

        def adjust_search_trace(search_trace, action_offset):
            for i, (r, e) in enumerate(search_trace):
                new_r = r[action_offset]
                new_e = e[action_offset]
                search_trace[i] = (new_r, new_e)

        r_s = th.full_like(e_s, self.kg.dummy_start_r)
        start_r = th.full(e_s.size(), self.kg.dummy_start_r, device=self.device, dtype=th.long)
        path_r = th.full((len(e_s), self.num_rollout_steps + 1), self.kg.dummy_r, device=self.device, dtype=th.long)
        path_r[:, 0] = start_r
        path_e = th.full((len(e_s), self.num_rollout_steps + 1), self.kg.dummy_e, device=self.device, dtype=th.long)
        path_e[:, 0] = e_s

        start_obs = Observation(
            num_rollout_steps=self.num_rollout_steps,
            query_relation=q,
            target_entity=e_t,
            path=(path_r, path_e),
            path_length=th.zeros_like(e_s).float(),
        )
        action = (r_s, e_s)
        if save_beam_search_paths:
            search_trace = [(r_s, e_s)]
        self._last_obs = start_obs
        for t in range(self.num_rollout_steps):
            action_space, q_values = self.policy.calculate_q_values(self._last_obs, self.kg, self.use_action_space_bucketing)
            if t == self.num_rollout_steps - 1:
                action, q_value, action_offset = top_k_answer_unique(q_values, action_space)
            else:
                action, q_value, action_offset = top_k_action(q_values, action_space)

            last_r, current_e = action
            k = int(current_e.size()[0] / batch_size)
            # => [batch_size*k]
            q = utils.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
            e_s = utils.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
            e_t = utils.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
            path_r, path_e = self._last_obs.path_r, self._last_obs.path_e
            path_r = utils.tile_along_beam(path_r.view(batch_size, -1), k)
            path_e = utils.tile_along_beam(path_e.view(batch_size, -1), k)
            path_r[:, t + 1] = last_r
            path_e[:, t + 1] = current_e

            new_obs = Observation(
                num_rollout_steps=self.num_rollout_steps,
                query_relation=q,
                target_entity=e_t,
                path=(path_r, path_e),
                path_length=th.ones_like(e_s).float() * (t + 1),
            )
            self._last_obs = new_obs

            if save_beam_search_paths:
                adjust_search_trace(search_trace, action_offset)
                search_trace.append(action)

        output_beam_size = int(action[0].size()[0] / batch_size)
        # [batch_size*beam_size] => [batch_size, beam_size]
        beam_search_output = dict()
        beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
        beam_search_output['pred_e2_scores'] = q_value.view(batch_size, -1)
        if save_beam_search_paths:
            beam_search_output['search_traces'] = search_trace

        return beam_search_output

    def beam_search_probability(
        self,
        mini_batch,
        beam_size: int,
        save_beam_search_paths=False,
    ):
        e_s, q, e_t = mini_batch
        batch_size = len(e_s)

        def top_k_action(log_action_dist, action_space):
            full_size = len(log_action_dist)
            assert (full_size % batch_size == 0)
            last_k = int(full_size / batch_size)

            (r_space, e_space), _ = action_space
            action_space_size = r_space.size()[1]
            # => [batch_size, k'*action_space_size]
            log_action_dist = log_action_dist.view(batch_size, -1)
            beam_action_space_size = log_action_dist.size()[1]
            k = min(beam_size, beam_action_space_size)
            # [batch_size, k]
            log_action_prob, action_ind = th.topk(log_action_dist, k)
            next_r = utils.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
            next_e = utils.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
            # [batch_size, k] => [batch_size*k]
            log_action_prob = log_action_prob.view(-1)
            # *** compute parent offset
            # [batch_size, k]
            action_beam_offset = action_ind // action_space_size
            # [batch_size, 1]
            action_batch_offset = th.arange(batch_size, device=self.device).unsqueeze(1) * last_k
            # [batch_size, k] => [batch_size*k]
            action_offset = (action_batch_offset + action_beam_offset).view(-1)
            return (next_r, next_e), log_action_prob, action_offset

        def top_k_answer_unique(log_action_dist, action_space):
            full_size = len(log_action_dist)
            assert (full_size % batch_size == 0)
            last_k = int(full_size / batch_size)
            (r_space, e_space), _ = action_space
            action_space_size = r_space.size()[1]

            r_space = r_space.view(batch_size, -1)
            e_space = e_space.view(batch_size, -1)
            log_action_dist = log_action_dist.view(batch_size, -1)
            beam_action_space_size = log_action_dist.size()[1]
            assert (beam_action_space_size % action_space_size == 0)
            k = min(beam_size, beam_action_space_size)
            next_r_list, next_e_list = [], []
            log_action_prob_list = []
            action_offset_list = []
            for i in range(batch_size):
                log_action_dist_b = log_action_dist[i]
                r_space_b = r_space[i]
                e_space_b = e_space[i]
                unique_e_space_b = th.unique(e_space_b.data.cpu()).to(device=self.device)
                unique_q_value_dist, unique_idx = utils.unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
                k_prime = min(len(unique_e_space_b), k)
                top_unique_q_value_dist, top_unique_idx2 = th.topk(unique_q_value_dist, k_prime)
                top_unique_idx = unique_idx[top_unique_idx2]
                top_unique_beam_offset = top_unique_idx // action_space_size
                top_r = r_space_b[top_unique_idx]
                top_e = e_space_b[top_unique_idx]
                next_r_list.append(top_r.unsqueeze(0))
                next_e_list.append(top_e.unsqueeze(0))
                log_action_prob_list.append(top_unique_q_value_dist.unsqueeze(0))
                top_unique_batch_offset = i * last_k
                top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
                action_offset_list.append(top_unique_action_offset.unsqueeze(0))
            next_r = utils.pad_and_cat(next_r_list, padding_value=self.kg.dummy_r).view(-1)
            next_e = utils.pad_and_cat(next_e_list, padding_value=self.kg.dummy_e).view(-1)
            log_action_prob = utils.pad_and_cat(log_action_prob_list, padding_value=-utils.HUGE_INT)
            action_offset = utils.pad_and_cat(action_offset_list, padding_value=-1)
            return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

        def adjust_search_trace(search_trace, action_offset):
            for i, (r, e) in enumerate(search_trace):
                new_r = r[action_offset]
                new_e = e[action_offset]
                search_trace[i] = (new_r, new_e)

        r_s = th.full_like(e_s, self.kg.dummy_start_r)
        start_r = th.full(e_s.size(), self.kg.dummy_start_r, device=self.device, dtype=th.long)
        path_r = th.full((len(e_s), self.num_rollout_steps + 1), self.kg.dummy_r, device=self.device, dtype=th.long)
        path_r[:, 0] = start_r
        path_e = th.full((len(e_s), self.num_rollout_steps + 1), self.kg.dummy_e, device=self.device, dtype=th.long)
        path_e[:, 0] = e_s
        log_action_prob = th.zeros(batch_size, device=self.device, dtype=th.float)

        start_obs = Observation(
            num_rollout_steps=self.num_rollout_steps,
            query_relation=q,
            target_entity=e_t,
            path=(path_r, path_e),
            path_length=th.zeros_like(e_s).float(),
        )
        action = (r_s, e_s)
        if save_beam_search_paths:
            search_trace = [(r_s, e_s)]
        self._last_obs = start_obs
        for t in range(self.num_rollout_steps):
            action_space, action_dist = self.policy.calculate_action_dist(self._last_obs, self.kg, self.use_action_space_bucketing,
                                                                          merge_aspace_batching_outcome=True)
            log_action_dist = log_action_prob.view(-1, 1) + utils.safe_log(action_dist)

            if t == self.num_rollout_steps - 1:
                action, log_action_prob, action_offset = top_k_answer_unique(log_action_dist, action_space)
            else:
                action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space)

            last_r, current_e = action
            k = int(current_e.size()[0] / batch_size)
            # => [batch_size*k]
            q = utils.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
            e_s = utils.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
            e_t = utils.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
            path_r, path_e = self._last_obs.path_r, self._last_obs.path_e
            path_r = utils.tile_along_beam(path_r.view(batch_size, -1), k)
            path_e = utils.tile_along_beam(path_e.view(batch_size, -1), k)
            path_r[:, t + 1] = last_r
            path_e[:, t + 1] = current_e

            new_obs = Observation(
                num_rollout_steps=self.num_rollout_steps,
                query_relation=q,
                target_entity=e_t,
                path=(path_r, path_e),
                path_length=th.ones_like(e_s).float() * (t + 1),
            )
            self._last_obs = new_obs

            if save_beam_search_paths:
                adjust_search_trace(search_trace, action_offset)
                search_trace.append(action)

        output_beam_size = int(action[0].size()[0] / batch_size)
        # [batch_size*beam_size] => [batch_size, beam_size]
        beam_search_output = dict()
        beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
        beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)
        if save_beam_search_paths:
            beam_search_output['search_traces'] = search_trace

        return beam_search_output


    def update_progress_state(self, epoch_id):
        self._update_current_progress_remaining(epoch_id, self.num_epochs)
        self.policy._update_exploration(self._current_progress_remaining)
        self.logger.info("train/current_progress_remaining:   {}".format(self._current_progress_remaining))
        self.logger.info("train/exploration_rate:   {}".format(self.policy.exploration_rate))

    def _update_current_progress_remaining(self, num_epochs: int, total_epochs: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_epochs + 1) / float(total_epochs)

    def _excluded_save_params(self) -> List[str]:
        return super(DQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []