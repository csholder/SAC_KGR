#!/usr/bin/env bash

data_dir="data/FB15K-237_20"
model_name="rl.sac"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=100
num_wait_epochs=10
num_peek_epochs=2
train_batch_size=128
eval_batch_size=8
baseline="na"
policy_class='SACPolicy'
replay_buffer_class='ReplayBuffer'
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
beta=0.02
relation_only="False"
beam_size=128
#num_paths_per_entity=-1
#margin=-1

actor_learning_rate=0.001
critic_learning_rate=0.0001
eof_learning_rate=0.0001
buffer_size=614400
buffer_batch_size=128
train_freq_value=20
train_freq_unit='episode'
learning_starts=3
gradient_steps=20
ent_coef='auto_0.01'
mu=1.0
tau=0.1  # 0.005
gamma=1.0
target_update_interval=1
target_entropy='auto'
activation_fn='relu'
exploration_initial_eps=1.0
exploration_final_eps=0.05
exploration_fraction=0.1