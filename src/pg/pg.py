"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Policy gradient (REINFORCE algorithm) training and inference.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from src.learn_framework import LFramework
from src.pg.policy import OnPolicy
from src.common import utils
from src.common.common_class import Observation
from src.common.knowledge_graph import KnowledgeGraph


class PolicyGradient(LFramework):
    def __init__(
        self,
        args,
        kg: KnowledgeGraph,
        entity_dim: int,
        relation_dim: int,
        history_dim: int,
        history_num_layers: int = 3,
        actor_learning_rate: float = 3e-4,
        ff_dropout_rate: float = 0.1,
        action_dropout_rate: float = 0.1,
        net_arch: List[int] = [64, 64],
        policy_class: Union[str, OnPolicy] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        xavier_initialization: bool = True,
        relation_only: bool = False,
    ):
        super(PolicyGradient, self).__init__(args, kg)

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.history_dim = history_dim
        self.history_num_layers = history_num_layers
        self.actor_learning_rate = actor_learning_rate
        self.ff_dropout_rate = ff_dropout_rate
        self.action_dropout_rate = action_dropout_rate
        self.net_arch = net_arch
        self.policy_class = policy_class
        self.xavier_initialization = xavier_initialization
        self.relation_only = relation_only

        # Training hyperparameters
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class

        self._setup_model()

        print('========================== PolicyGradient ==========================')
        print('entity_dim: ', entity_dim)
        print('relation_dim: ', relation_dim)
        print('history dim: ', history_dim)
        print('history_num_layers: ', history_num_layers)
        print('actor_learning_rate: ', actor_learning_rate)
        print('num_rollout_steps: ', self.num_rollout_steps)
        print('ff_dropout_rate: ', ff_dropout_rate)
        print('action_dropout_rate: ', action_dropout_rate)
        print('policy_class: ', policy_class)

    def _setup_model(self):
        self.policy_kwargs = {
            'entity_dim': self.entity_dim,
            'relation_dim': self.relation_dim,
            'history_dim': self.history_dim,
            'history_num_layers': self.history_num_layers,
            'activation_fn': torch.nn.Tanh,
            'net_arch': self.net_arch,
            'ff_dropout_rate': self.ff_dropout_rate,
            'action_dropout_rate': self.action_dropout_rate,
            'xavier_initialization': self.xavier_initialization,
            'relation_only': self.relation_only,
        }

        if self.policy_class == 'OnPolicy':
            self.policy = OnPolicy(  # pytype:disable=not-instantiable
                **self.policy_kwargs,  # pytype:disable=not-instantiable
            )
        else:
            raise NotImplementedError
        self.policy = self.policy.to(self.device)
        self._last_obs = None

        self._create_aliases()
        self.optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.parameters()),
                                              lr=self.actor_learning_rate, **self.optimizer_kwargs)
        for param_group in self.optimizer.param_groups:  #
            print(param_group)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor

    def learn(
        self,
        mini_batch
    ):
        loss_dict = self.loss(mini_batch)
        loss = loss_dict['model_loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_dict

    def reward_fun(self, e1, r, e2, pred_e2):
        reward = (pred_e2 == e2).float()
        return reward, reward

    def loss(self, mini_batch):
        e1, r, e2 = mini_batch
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']

        # Compute discounted reward
        final_reward, binary_reward = self.reward_fun(e1, r, e2, pred_e2)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = binary_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            # fn = torch.zeros(final_reward.size())
            # for i in range(len(final_reward)):
            #     if not final_reward[i]:
            #         if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
            #             fn[i] = 1
            # if self.fns is None:
            #     self.fns = fn
            # else:
            #     self.fns = torch.cat([self.fns, fn])

            current_rewards = (binary_reward == 1.).float()
            if self.rewards is None:
                self.rewards = current_rewards
            else:
                self.rewards = torch.cat([self.rewards, current_rewards])

        return loss_dict

    def _setup_learn(
        self,
        mini_batch,
    ):
        start_e, query_r, target_e = mini_batch
        start_r = torch.full(start_e.size(), self.kg.dummy_start_r, device=self.device, dtype=torch.long)
        path_r = torch.full((len(start_e), self.num_rollout_steps + 1), self.kg.dummy_r, device=self.device, dtype=torch.long)
        path_r[:, 0] = start_r
        path_e = torch.full((len(start_e), self.num_rollout_steps + 1), self.kg.dummy_e, device=self.device, dtype=torch.long)
        path_e[:, 0] = start_e
        init_obs = Observation(
            num_rollout_steps=self.num_rollout_steps,
            query_relation=query_r,
            target_entity=target_e,
            path=(path_r, path_e),
            path_length=torch.zeros_like(start_r).float()
        )

        return init_obs

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
        return new_obs

    def rollout(self, e_s, q, e_t, num_steps):
        assert (num_steps > 0)
        # Initialization
        log_action_probs = []
        action_entropy = []
        start_observation = self._setup_learn(mini_batch=(e_s, q, e_t))
        self._last_obs = start_observation

        for t in range(num_steps):
            sample_outcome = self.actor.action_prob(self._last_obs, self.kg,
                                                    use_action_space_bucketing=self.use_action_space_bucketing)

            action = sample_outcome['action_sample']
            new_obs = self._step(self._last_obs, action)

            action_prob = sample_outcome['action_prob']
            log_action_probs.append(utils.safe_log(action_prob))
            policy_entropy = sample_outcome['entropy']
            action_entropy.append(policy_entropy)
            self._last_obs = new_obs

        pred_e2 = self._last_obs.path_e[:, -1]
        if self.run_analysis:
            self.record_path_trace((self._last_obs.path_r, self._last_obs.path_e))

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
        }

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = torch.tensor(rand > self.action_dropout_rate, device=self.device).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + utils.EPSILON * (1 - action_keep_mask) * action_mask
                if (sample_action_dist.sum(dim=-1) == 0.).any():
                    nonzero_mask = (sample_action_dist.sum(dim=-1) != 0.).unsqueeze(dim=-1).repeat(
                        1, sample_action_dist.shape[-1])
                    epsilon_tensor = torch.ones_like(sample_action_dist) * (1.0 / sample_action_dist.shape[-1])
                    sample_action_dist = torch.where(nonzero_mask, sample_action_dist, epsilon_tensor)
                    self.logger.info('Distribution occurs all zero lines in hr_pg.')
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = utils.batch_lookup(r_space, idx)
            next_e = utils.batch_lookup(e_space, idx)
            action_prob = utils.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(
        self,
        mini_batch,
        beam_size,
        verbose=False,
        query_path_dict=None
    ):
        self.eval()
        with torch.no_grad():
            e1, e2, r = mini_batch
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
                        if query_path_dict is not None:
                            query_path_dict[(int(e1[i]), int(e2[i]), int(r[i]))].append(
                                [float(pred_e2_scores[i][j]), path_str])

            pred_scores = torch.zeros([len(e1), self.kg.num_entities], device=self.device)
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores, query_path_dict

    def _predict_action_dist(
        self,
        obs: Observation,
        merge_aspace_batching_outcome=False,
    ):
        return self.policy.action_distribution(obs, self.kg, self.use_action_space_bucketing, merge_aspace_batching_outcome)

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
            log_action_prob, action_ind = torch.topk(log_action_dist, k)
            next_r = utils.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
            next_e = utils.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
            # [batch_size, k] => [batch_size*k]
            log_action_prob = log_action_prob.view(-1)
            # *** compute parent offset
            # [batch_size, k]
            action_beam_offset = action_ind // action_space_size
            # [batch_size, 1]
            action_batch_offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * last_k
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
                unique_e_space_b = torch.unique(e_space_b.data.cpu()).to(device=self.device)
                unique_log_action_dist, unique_idx = utils.unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
                k_prime = min(len(unique_e_space_b), k)
                top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
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

        r_s = torch.full_like(e_s, self.kg.dummy_start_r)
        start_r = torch.full(e_s.size(), self.kg.dummy_start_r, device=self.device, dtype=torch.long)
        path_r = torch.full((len(e_s), self.num_rollout_steps + 1), self.kg.dummy_r, device=self.device, dtype=torch.long)
        path_r[:, 0] = start_r
        path_e = torch.full((len(e_s), self.num_rollout_steps + 1), self.kg.dummy_e, device=self.device, dtype=torch.long)
        path_e[:, 0] = e_s
        log_action_prob = torch.zeros(batch_size, device=self.device, dtype=torch.float)

        start_obs = Observation(
            num_rollout_steps=self.num_rollout_steps,
            query_relation=q,
            target_entity=e_t,
            path=(path_r, path_e),
            path_length=torch.zeros_like(e_s).float(),
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
                path_length=torch.ones_like(e_s).float() * (t + 1),
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
        pass

    def record_path_trace(self, path_trace):
        path_length = path_trace[0].size(1) * 2
        flattened_path_trace = [t.unsqueeze(dim=-1) for t in path_trace]
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
