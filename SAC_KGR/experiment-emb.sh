#!/bin/bash

source ~/.bashrc
conda activate py37_torch151

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

add_reversed_training_edges_flag=''
if [[ $add_reversed_training_edges = *"True"* ]]; then
    add_reversed_training_edges_flag="--add_reversed_training_edges"
fi
group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model_name $model_name \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --num_rollouts $num_rollouts \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --train_batch_size $train_batch_size \
    --eval_batch_size $eval_batch_size \
    --learning_rate $learning_rate \
    --num_negative_samples $num_negative_samples \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --beam_size $beam_size \
    $group_examples_by_query_flag \
    $add_reversed_training_edges_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
