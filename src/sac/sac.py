from typing import Any, Dict, List, Optional, Tuple, Type, Union
from logging import Logger
import os
import wandb
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import defaultdict as ddict

from src.common.off_policy_algorithm import OffPolicyAlgorithm
from src.common.buffers import ReplayBuffer
from src.sac.policy import SACPolicy
from src.common.knowledge_graph import KnowledgeGraph
import src.common.utils as utils
from src.common.common_class import Observation


class SAC(OffPolicyAlgorithm):
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
        xavier_initialization: bool = True,
        relation_only: bool = False,
        buffer_size: int = 100,
        batch_size: int = 256,
        learning_starts: int = 100,
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.5,
        net_arch: List[int] = [64, 64],
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = 'auto',
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = 'auto',
        action_entropy_ratio: float = 0.8,
        max_grad_norm: float = 0.,
        replay_buffer_class: Union[str, ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        policy_class: Union[str, SACPolicy] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ):
        super(SAC, self).__init__(
            args,
            kg=kg,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            history_dim=history_dim,
            history_num_layers=history_num_layers,
            ff_dropout_rate=ff_dropout_rate,
            critic_learning_rate=critic_learning_rate,
            xavier_initialization=xavier_initialization,
            relation_only=relation_only,
            buffer_size=buffer_size,
            buffer_batch_size=batch_size,
            learning_starts=learning_starts,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            policy_class=policy_class,
            verbose=verbose,
        )
        self.history_dim = history_dim
        self.history_num_layers = history_num_layers

        self.action_dropout_rate = action_dropout_rate
        self.actor_learning_rate = actor_learning_rate
        self.target_entropy = target_entropy
        self.action_entropy_ratio = action_entropy_ratio
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        self.eof_learning_rate = eof_learning_rate
        self.ent_coef_optimizer = None
        self.ent_coef = ent_coef

        self.optimizer_class = optimizer_class
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs

        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs

        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self.max_grad_norm = max_grad_norm

        self.net_arch = net_arch
        self.policy_class = policy_class
        self._setup_model()

    def _setup_model(self) -> None:
        self.policy_kwargs.update({
            'action_dropout_rate': self.action_dropout_rate,
            'net_arch': self.net_arch,
            'share_features_extractor': True,
        })
        super(SAC, self)._setup_model()
        self._create_aliases()

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            # self.target_entropy = -np.prod(self.kg.max_num_actions).astype(np.float32)
            # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
            self.target_entropy = np.log(self.kg.max_num_actions).astype(np.float32) * self.action_entropy_ratio
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        self.logger.info('Entropy low threshold: {}'.format(self.target_entropy))
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = self.optimizer_class([self.log_ent_coef], lr=self.eof_learning_rate)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

            # Setup optimizer with initial learning rate
        self.critic_optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                                     lr=self.critic_learning_rate, **self.optimizer_kwargs)

        # Setup optimizer with initial learning rate
        self.actor_optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.actor.parameters()),
                                                    lr=self.actor_learning_rate, **self.optimizer_kwargs)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _predict_action_dist(
        self,
        obs: Observation,
        merge_aspace_batching_outcome=True,
    ):
        return self.actor.action_distribution(obs, self.kg, self.use_action_space_bucketing, merge_aspace_batching_outcome)

    def do_train(self, gradient_steps: int, batch_size: int = 64):
        self.train()
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor_optimizer, self.critic_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        # self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs, action_entropies = [], [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Action by the current actor for the sampled state
            # sample_outcome = self.actor.action_prob(replay_data.observation, kg=self.kg,
            #                                         use_action_space_bucketing=self.use_action_space_bucketing)
            action_space, action_dist = self.actor.action_distribution(replay_data.observation, kg=self.kg, 
                                                                       use_action_space_bucketing=self.use_action_space_bucketing,
                                                                       merge_aspace_batching_outcome=True)

            # action_pi, prob = sample_outcome['action_sample'], sample_outcome['action_prob']
            # log_prob = utils.safe_log(prob)
            action_mask = action_space[1]
            log_action_dist = utils.safe_log(action_dist)
            log_action_dist *= action_mask
            
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef)
                # ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_loss = th.mul(action_dist.detach(), -(ent_coef * (log_action_dist.detach() + self.target_entropy))).sum(dim=-1).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            entropy = -th.mul(action_dist.detach(), log_action_dist.detach()).sum(dim=-1).mean()
            action_entropies.append(entropy.item())
            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            ent_coef = th.exp(self.log_ent_coef.detach())
            with th.no_grad():
                # Select action according to policy
                next_sample_outcome = self.actor.action_prob(replay_data.next_observation, kg=self.kg,
                                                             use_action_space_bucketing=self.use_action_space_bucketing)
                next_action, next_prob = next_sample_outcome['action_sample'], next_sample_outcome['action_prob']
                next_log_prob = utils.safe_log(next_prob)
                # Compute the next Q values: min over all critics targets
                next_action_embedding = utils.get_action_embedding(next_action, self.kg)
                next_q_values = self.critic_target.forward(replay_data.next_observation, next_action_embedding, self.kg)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob
                # td error + entropy term
                target_q_values = replay_data.reward + (1 - replay_data.next_observation.done) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            store_action_embedding = utils.get_action_embedding(replay_data.action, self.kg)
            current_q_values = self.critic.forward(replay_data.observation, store_action_embedding, self.kg)

            # Compute critic loss
            # critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_loss = 0.5 * F.mse_loss(current_q_values, target_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            # current_action_embedding = utils.get_action_embedding(action_pi, self.kg)
            # min_qf_pi = self.critic.forward(replay_data.observation, current_action_embedding, self.kg)
            # actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            _, current_q_values_for_action_space = self.critic.calculate_q_values(replay_data.observation, self.kg, self.use_action_space_bucketing)
            actor_loss = th.matmul(action_dist, (ent_coef.detach() * log_action_dist - current_q_values_for_action_space).t()).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        loss_dict = ddict(list)
        loss_dict["n_updates"] = self._n_updates
        loss_dict["ent_coef"] = np.mean(ent_coefs)
        loss_dict["actor_loss"] = np.mean(actor_losses)
        loss_dict["critic_loss"] = np.mean(critic_losses)
        loss_dict["action_entropy"] = np.mean(action_entropies)
        if len(ent_coef_losses) > 0:
            loss_dict["ent_coef_loss"] = np.mean(ent_coef_losses)

        if self.args.use_wandb:
            wandb.log({'ent_coef': loss_dict["ent_coef"]})
            wandb.log({'actor_loss': loss_dict["actor_loss"]})
            wandb.log({'critic_loss': loss_dict["critic_loss"]})
            wandb.log({'ent_coef_losses': loss_dict["ent_coef_loss"]})
            wandb.log({'action_entropy': loss_dict["action_entropy"]})
        return loss_dict

    def learn(
        self,
        mini_batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
        log_interval: int = 4,
    ):
        return super(SAC, self).learn(
            mini_batch=mini_batch,
        )

    def predict(
        self,
        mini_batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
        beam_size: int,
        verbose=False,
        query_path_dict=None,
    ):
        self.eval()
        with th.no_grad():
            e1, r, e2 = mini_batch
            beam_search_output = self.beam_search(mini_batch, beam_size)

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
                        # self.logger.info('beam {}: score = {} \n<PATH> {}\n'.format(
                        #     j, float(pred_e2_scores[i][j]), path_str))
                        if query_path_dict is not None:
                            query_path_dict[(int(e1[i]), int(e2[i]), int(r[i]))].append(
                                [float(pred_e2_scores[i][j]), path_str])

            pred_scores = th.zeros([len(e1), self.kg.num_entities], device=self.device)
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = th.exp(pred_e2_scores[i])
        return pred_scores, query_path_dict

    def beam_search(
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
                unique_log_action_dist, unique_idx = utils.unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
                k_prime = min(len(unique_e_space_b), k)
                top_unique_log_action_dist, top_unique_idx2 = th.topk(unique_log_action_dist, k_prime)
                top_unique_idx = unique_idx[top_unique_idx2]
                top_unique_beam_offset = top_unique_idx // action_space_size
                top_r = r_space_b[top_unique_idx]
                top_e = e_space_b[top_unique_idx]
                next_r_list.append(top_r.unsqueeze(0))
                next_e_list.append(top_e.unsqueeze(0))
                log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
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
            action_space, action_dist = self._predict_action_dist(self._last_obs,
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
        # self.policy._update_exploration(self._current_progress_remaining)
        self.logger.info("train/current_progress_remaining:   {}".format(self._current_progress_remaining))
        # self.logger.info("train/exploration_rate:   {}".format(self.policy.exploration_rate))

    def _update_current_progress_remaining(self, num_epochs: int, total_epochs: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_epochs + 1) / float(total_epochs)