#!/bin/bash

source ~/.bashrc
conda activate py37_torch151

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model_name $model_name \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --train_batch_size $train_batch_size \
    --eval_batch_size $eval_batch_size \
    --baseline $baseline \
    --policy_class $policy_class \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    $relation_only_flag \
    --beta $beta \
    --beam_size $beam_size \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    --gpu $gpu \
    --actor_learning_rate $actor_learning_rate \
    --critic_learning_rate $critic_learning_rate \
    --eof_learning_rate $eof_learning_rate \
    --buffer_size $buffer_size \
    --buffer_batch_size $buffer_batch_size \
    --magnification $magnification \
    --train_freq_value $train_freq_value \
    --train_freq_unit $train_freq_unit \
    --learning_starts $learning_starts \
    --gradient_steps $gradient_steps \
    --ent_coef $ent_coef \
    --mu $mu \
    --tau $tau \
    --gamma $gamma \
    --target_update_interval $target_update_interval \
    --target_entropy $target_entropy \
    --activation_fn $activation_fn \
    --exploration_initial_eps $exploration_initial_eps \
    --exploration_final_eps $exploration_final_eps \
    --exploration_fraction $exploration_fraction \
    --conve_state_dict_path $conve_state_dict_path \
    $ARGS"

echo "Executing $cmd"

$cmd
