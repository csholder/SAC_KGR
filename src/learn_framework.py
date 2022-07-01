import os
import random
import shutil

import wandb
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# from src.sac.sac import SAC
from src.common.common_class import Observation
from src.common.utils import format_batch
from src.eval import *


class LFramework(nn.Module):

    def __init__(self, args, kg):
        super(LFramework, self).__init__()
        self.args = args
        self.logger = args.logger
        self.device = args.device
        self.model_name = args.model_name
        self.model_dir = args.model_dir

        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps=args.num_rollout_steps
        self.beam_size = args.beam_size

        self.kg = kg

        self.run_analysis = args.run_analysis
        self.num_peek_epochs = args.num_peek_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self._current_progress_remaining = 1.

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0
        self.rewards = None

        print('========================== LFramework ==========================')
        print('model_name: ', self.model_name)
        print('start_epoch: ', self.start_epoch)
        print('num_epochs: ', self.num_epochs)
        print('train_batch_size: ', self.train_batch_size)
        print('eval_batch_size: ', self.eval_batch_size)
        print('num_rollouts: ', self.num_rollouts)
        print('beam_size: ', self.beam_size)
        print('run_analysis: ', self.run_analysis)
        print('num_peek_epochs: ', self.num_peek_epochs)
        print('num_wait_epochs: ', self.num_wait_epochs)
        print('use_action_space_bucketing: ', self.use_action_space_bucketing)

    def print_all_model_parameters(self):
        self.logger.info('\nModel Parameters')
        self.logger.info('--------------------------')
        for name, param in self.named_parameters():
            self.logger.info('{}\t{}\trequires_grad={}'.format(name, param.numel(), param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        self.logger.info('Total # parameters = {}'.format(sum(param_sizes)))
        self.logger.info('--------------------------')

    def reward_fun(self, e1, r, e2, pred_e2):
        return (pred_e2 == e2).float()

    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()

        best_dev_metrics = self.args.best_dev_metrics
        dev_metrics_history = []

        for epoch_id in range(self.start_epoch, self.num_epochs):
            self.logger.info('Epoch {}: '.format(epoch_id))
            self.train()

            # random.shuffle(train_data)

            if self.run_analysis:
                self.rewards = None
                self.fns = None
                self.path_types = dict()
                self.num_path_types = 0

            batch_losses, entropies, batch_actor_losses, batch_critic_losses, batch_ent_coef_losses = [], [], [], [], []
            for example_id in tqdm(range(0, len(train_data), self.train_batch_size), desc='Training: '):
                mini_batch = train_data[example_id: example_id + self.train_batch_size]
                mini_batch = format_batch(mini_batch, num_labels=self.kg.num_entities,
                                          num_tiles=self.num_rollouts, device=self.device)
                loss_dict = self.learn(mini_batch)

                if 'sac' in self.model_name:
                    if 'actor_loss' in loss_dict:
                        batch_actor_losses.append(loss_dict['actor_loss'])
                    if 'critic_loss' in loss_dict:
                        batch_critic_losses.append(loss_dict['critic_loss'])
                    if 'ent_coef_loss' in loss_dict:
                        batch_ent_coef_losses.append(loss_dict['ent_coef_loss'])
                else:
                    if 'print_loss' in loss_dict:
                        batch_losses.append(loss_dict['print_loss'])

                if 'entropy' in loss_dict:
                    entropies.append(loss_dict['entropy'])

            self.update_progress_state(epoch_id)

            if 'sac' in self.model_name:
                stdout_msg = 'Epoch {}: average actor loss = {}\tcritic loss = {}\tent_coef_loss: {}'.format(epoch_id, np.mean(batch_actor_losses),
                                                                                                             np.mean(batch_critic_losses),
                                                                                                             np.mean(batch_ent_coef_losses))
                if self.args.use_wandb:
                    wandb.log({'epoch actor loss': np.mean(batch_actor_losses)})
                    wandb.log({'epoch critic loss': np.mean(batch_critic_losses)})
                    wandb.log({'epoch ent coef loss': np.mean(batch_ent_coef_losses)})
            else:
                stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
                if self.args.use_wandb:
                    wandb.log({'epoch loss': np.mean(batch_losses)})
            if entropies:
                stdout_msg += ' entropy = {}'.format(np.mean(entropies))
            self.logger.info(stdout_msg)

            if self.run_analysis and self.model_name.startswith('rl'):
                self.logger.info('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(self.rewards.sum())
                hit_ratio = num_hits / len(self.rewards)
                self.logger.info('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                if self.args.use_wandb:
                    wandb.log({'path_type_seen': self.num_path_types})
                    wandb.log({'hit num': num_hits})
                    wandb.log({'hit ratio': hit_ratio})

                num_fns = float(self.fns.sum())
                fn_ratio = num_fns / len(self.fns)
                self.logger.info('* Analysis: false negative ratio = {}'.format(fn_ratio))

            if self.run_analysis or (epoch_id % self.num_peek_epochs == 0):
                tail_dev_data = [dev_data[idx * 2] for idx in range(len(dev_data) // 2)]
                head_dev_data = [dev_data[idx * 2 + 1] for idx in range(len(dev_data) // 2)]
                tail_dev_scores, _ = self.forward(tail_dev_data, verbose=False)
                self.logger.info('Dev set performance for tail prediction: (correct evaluation)')
                h1_t, h3_t, h5_t, h10_t, mrr_t, _, _, _, _, _ = hits_and_ranks(tail_dev_data, tail_dev_scores,
                                                                               self.kg.dev_objects,
                                                                               self.logger, verbose=True)
                head_dev_scores, _ = self.forward(head_dev_data, verbose=False)
                self.logger.info('Dev set performance for head prediction: (correct evaluation)')
                h1_h, h3_h, h5_h, h10_h, mrr_h, _, _, _, _, _ = hits_and_ranks(head_dev_data, head_dev_scores,
                                                                               self.kg.dev_objects,
                                                                               self.logger, verbose=True)
                mrr = (mrr_t + mrr_h) / 2
                metrics = mrr
                self.logger.info('Dev set performance: (correct evaluation)')
                self.logger.info('Hits@1 = {}'.format((h1_t + h1_h) / 2))
                self.logger.info('Hits@3 = {}'.format((h3_t + h3_h) / 2))
                self.logger.info('Hits@5 = {}'.format((h5_t + h5_h) / 2))
                self.logger.info('Hits@10 = {}'.format((h10_t + h10_h) / 2))
                self.logger.info('MRR = {}'.format(mrr))

                if self.args.use_wandb:
                    wandb.log({'hits@1': (h1_t + h1_h) / 2})
                    wandb.log({'hits@3': (h3_t + h3_h) / 2})
                    wandb.log({'hits@5': (h5_t + h5_h) / 2})
                    wandb.log({'hits@10': (h10_t + h10_h) / 2})
                    wandb.log({'mrr': mrr})
                # Action dropout anneaking
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(save_dir=self.model_dir, checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)

    def learn(self, mini_batch) -> dict:
        raise NotImplementedError

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def forward(self, examples, verbose=False, query_path_dict=None):
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.eval_batch_size)):
            mini_batch = examples[example_id:example_id + self.eval_batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.eval_batch_size:
                self.make_full_batch(mini_batch, self.eval_batch_size)
            mini_batch = format_batch(mini_batch, num_labels=self.kg.num_entities, num_tiles=1, device=self.device)
            if self.model_name.startswith('embed'):
                pred_score = self.predict(mini_batch)
            else:
                pred_score, query_path_dict = self.predict(mini_batch, beam_size=self.beam_size,
                                                           use_action_space_bucketing=self.use_action_space_bucketing,
                                                           verbose=verbose, query_path_dict=query_path_dict)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores, query_path_dict

    def update_progress_state(self, epoch_id):
        raise NotImplementedError

    def save_checkpoint(self, save_dir, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(save_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(save_dir, 'model_best.tar')
            # shutil.copyfile(out_tar, best_path)
            torch.save(checkpoint_dict, best_path)
            self.logger.info('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            self.logger.info('=> saving checkpoint to \'{}\''.format(out_tar))

    def load_checkpoint(self, model_path, do_train=False):
        if os.path.isfile(model_path):
            self.logger.info('=> loading checkpoint from \'{}\''.format(model_path))
            checkpoint = torch.load(model_path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
            if do_train:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            self.logger.info('=> not checkpoint found at \'{}\''.format(model_path))
