#!/usr/bin/env bash

data_dir="data/FB15K-237_20"
model_name="embed.conve"

add_reversed_training_edges="True"
group_examples_by_query="True"
entity_dim=200
relation_dim=200
num_rollouts=1
num_epochs=1000
num_wait_epochs=100
train_batch_size=512
eval_batch_size=128
learning_rate=0.001
grad_norm=0
emb_dropout_rate=0.3
beam_size=128

num_negative_samples=100
margin=0.5
