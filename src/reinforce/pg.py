"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Policy gradient (REINFORCE algorithm) training and inference.
"""

import torch

from src.learn_framework import LFramework
from src.common import utils
import src.reinforce.beam_search as search


class REINFORCE(LFramework):
    def __init__(self, args, kg, pn):
        super(REINFORCE, self).__init__(args, kg)

        # Training hyperparameters
        self.model = args.model_name
        self.relation_only = args.relation_only
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.actor_learning_rate = args.actor_learning_rate

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0
        self.policy = pn

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.actor_learning_rate)
        # for param_group in self.optimizer.param_groups:  #
        #     print(param_group)

    def reward_fun(self, e1, r, e2, pred_e2):
        reward = (pred_e2 == e2).float()
        return reward, reward

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
            # loss_dict['fn'] = fn

            current_rewards = (binary_reward == 1.).float()
            if self.rewards is None:
                self.rewards = current_rewards
            else:
                self.rewards = torch.cat([self.rewards, current_rewards])

        return loss_dict

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.policy

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = utils.int_fill_var_cuda(e_s.size(), kg.dummy_start_r, device=self.device)
        seen_nodes = utils.int_fill_var_cuda(e_s.size(), kg.dummy_e, device=self.device).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(utils.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        if self.run_analysis:
            self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
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
                action_keep_mask = utils.var_cuda(rand > self.action_dropout_rate, device=self.device).float()
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

    def predict(self, mini_batch, beam_size, use_action_space_bucketing, verbose=False, query_path_dict=None):
        kg, pn = self.kg, self.policy
        with torch.no_grad():
            e1, r, e2 = mini_batch
            beam_search_output = search.beam_search(pn, e1, r, e2, kg, self.num_rollout_steps, beam_size)
            pred_e2s = beam_search_output['pred_e2s']
            pred_e2_scores = beam_search_output['pred_e2_scores']
            if verbose:
                # print inference paths
                search_traces = beam_search_output['search_traces']
                output_beam_size = min(beam_size, pred_e2_scores.shape[1])
                for i in range(len(e1)):
                    for j in range(output_beam_size):
                        ind = i * output_beam_size + j
                        if pred_e2s[i][j] == kg.dummy_e:
                            break
                        search_trace = []
                        for k in range(len(search_traces)):
                            search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                        path_str = utils.format_path(search_trace, kg)
                        # self.logger.info('beam {}: score = {} \n<PATH> {}\n'.format(
                        #     j, float(pred_e2_scores[i][j]), path_str))
                        if query_path_dict is not None:
                            query_path_dict[(int(e1[i]), int(e2[i]), int(r[i]))].append(
                                [float(pred_e2_scores[i][j]), path_str])

            pred_scores = utils.zeros_var_cuda([len(e1), kg.num_entities], device=self.device)
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores, query_path_dict

    def record_path_trace(self, path_trace):
        path_length = len(path_trace) * 2
        flattened_path_trace = [x.unsqueeze(dim=-1) for t in path_trace for x in t]
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
