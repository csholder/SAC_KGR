import argparse
import os



parser = argparse.ArgumentParser(description='SAC Algorithm on Knowledge Graph Reasoning with Reward Shaping')

# Experimental control
parser.add_argument('--process_data', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--run_analysis', action='store_true')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'))
parser.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'))
parser.add_argument('--log_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log'))
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--gpu', type=int, default=-1, help='gpu device (default: -1)')
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')
parser.add_argument('--use_wandb', action='store_true')


# Environmental Setups
parser.add_argument('--best_dev_metrics', type=float, default=0., help='path to a pretrained checkpoint')
parser.add_argument('--add_reverse_relations', type=bool, default=True,
                    help='add reverse relations to KB (default: True)')
parser.add_argument('--add_reversed_training_edges', action='store_true',
                    help='add reversed edges to extend training set (default: False)')
parser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate (default: 0.3)')
parser.add_argument('--group_examples_by_query', action='store_true',
                    help='group examples by topic entity + query relation (default: False)')
parser.add_argument('--entity_dim', type=int, default=200, metavar='E',
                    help='entity embedding dimension (default: 200)')
parser.add_argument('--relation_dim', type=int, default=200, metavar='R',
                    help='relation embedding dimension (default: 200)')
parser.add_argument('--history_dim', type=int, default=200, metavar='H',
                    help='action history encoding LSTM hidden states dimension (default: 400)')
parser.add_argument('--history_num_layers', type=int, default=3, metavar='L',
                    help='action history encoding LSTM number of layers (default: 1)')
parser.add_argument('--use_action_space_bucketing', action='store_true',
                    help='bucket adjacency list by outgoing degree to avoid memory blow-up (default: False)')
parser.add_argument('--bucket_interval', type=int, default=10,
                    help='adjacency list bucket size (default: 32)')
parser.add_argument('--test', action='store_true',
                    help='perform inference on the test set (default: False)')

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--actor_learning_rate', type=float, default=0.001)
parser.add_argument('--critic_learning_rate', type=float, default=0.001)
parser.add_argument('--eof_learning_rate', type=float, default=0.001)
parser.add_argument('--lr_scheduler_step', type=int, default=-1)
parser.add_argument('--lr_decay_gamma', type=float, default=0.75)
parser.add_argument('--num_epochs', type=int, default=200,
                    help='maximum number of pass over the entire training set (default: 20)')
parser.add_argument('--num_wait_epochs', type=int, default=5,
                    help='number of epochs to wait before stopping training if dev set performance drops')
parser.add_argument('--num_peek_epochs', type=int, default=1,
                    help='number of epochs to wait for next dev set result check (default: 1)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch from which the training should start (default: 0)')
parser.add_argument('--train_batch_size', type=int, default=10,
                    help='mini-batch size during training (default: 256)')
parser.add_argument('--eval_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
parser.add_argument('--buffer_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
parser.add_argument('--train_freq_value', type=int, default=1)
parser.add_argument('--train_freq_unit', type=str, default='episode')
parser.add_argument('--gradient_steps', type=int, default=1)
parser.add_argument('--grad_norm', type=float, default=10000,
                    help='norm threshold for gradient clipping (default 10000)')
parser.add_argument('--xavier_initialization', type=bool, default=True,
                    help='Initialize all model parameters using xavier initialization (default: True)')


# Embedding Model
parser.add_argument('--hidden_dropout_rate', type=float, default=0.3,
                    help='ConvE hidden layer dropout rate (default: 0.3)')
parser.add_argument('--feat_dropout_rate', type=float, default=0.2,
                    help='ConvE feature dropout rate (default: 0.2)')
parser.add_argument('--emb_2D_d1', type=int, default=10,
                    help='ConvE embedding 2D shape dimension 1 (default: 10)')
parser.add_argument('--emb_2D_d2', type=int, default=20,
                    help='ConvE embedding 2D shape dimension 2 (default: 20)')
parser.add_argument('--num_out_channels', type=int, default=32,
                    help='ConvE number of output channels of the convolution layer (default: 32)')
parser.add_argument('--kernel_size', type=int, default=3, help='ConvE kernel size (default: 3)')
parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1, help='epsilon used for label smoothing')
parser.add_argument('--num_negative_samples', type=int, default=10,
                    help='number of negative samples to use for embedding-based methods')
parser.add_argument('--conve_state_dict_path', type=str, default='',
                    help='Path to the ConvE network state dict (default: '')')
parser.add_argument('--theta', type=float, default=0.2,
                    help='Threshold for sifting high-confidence facts (default: 0.2)')

# Reinforcement Learning
parser.add_argument('--model_name', type=str, default='rl.sac.conve')
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='number of rollouts (default: 20)')
parser.add_argument('--num_rollout_steps', type=int, default=3,
                    help='maximum path length (default: 3)')
parser.add_argument('--bandwidth', type=int, default=300,
                    help='maximum number of outgoing edges to explore at each step (default: 300)')
parser.add_argument('--relation_only', action='store_true',
                    help='search with relation information only, ignoring entity representation (default: False)')
parser.add_argument('--relation_only_in_path', action='store_true', help='')

# Policy Gradient
parser.add_argument('--policy_class', type=str, default='SACPolicy',
                    help='policy class for sac algorithm')
parser.add_argument('--baseline', type=str, default='na',
                    help='baseline used by the policy gradient algorithm (default: na)')
parser.add_argument('--ff_dropout_rate', type=float, default=0.1,
                    help='Feed-forward layer dropout rate (default: 0.1)')
parser.add_argument('--rnn_dropout_rate', type=float, default=0.0,
                    help='RNN Variational Dropout Rate (default: 0.0)')
parser.add_argument('--action_dropout_rate', type=float, default=0.5,
                    help='Dropout rate for randomly masking out knowledge graph edges (default: 0.1)')
parser.add_argument('--action_dropout_final_rate', type=float, default=0.5)
parser.add_argument('--action_dropout_fraction', type=float, default=0.1)
parser.add_argument('--activation_fn', type=str, default='relu')
parser.add_argument('--net_arch', type=str, default='64_64')
parser.add_argument('--verbose', type=int, default=0)


# DQN
parser.add_argument('--exploration_initial_eps', type=float, default=1.0)
parser.add_argument('--exploration_final_eps', type=float, default=0.05)
parser.add_argument('--exploration_fraction', type=float, default=0.1)

# Critic
parser.add_argument('--buffer_size', type=int, default=3000,
                    help='maximum number of outgoing edges to explore at each step (default: 3000)')
parser.add_argument('--magnification', type=int, default=1)
parser.add_argument('--replay_buffer_class', type=str, default='ReplayBuffer',
                    help='replay buffer class')
parser.add_argument('--beta', type=float, default=0.0,
                    help='entropy regularization weight (default: 0.0)')
parser.add_argument('--gamma', type=float, default=1,
                    help='reward decay parameter (default: 1.0)')
parser.add_argument('--tau', type=float, default=0.005,
                    help='parameter used in updating target q network (default: 0.005)')
parser.add_argument('--learning_starts', type=int, default=300)
parser.add_argument('--critic_optimize_epoch', type=int, default=1)
parser.add_argument('--target_update_interval', type=int, default=3)
parser.add_argument('--ent_coef', type=str, default='auto_0.01')
parser.add_argument('--share_features_extractor', action='store_true')
parser.add_argument('--target_entropy', type=str, default='auto')
parser.add_argument('--action_entropy_ratio', type=float, default=0.8)
parser.add_argument('--n_critics', type=int, default=1)


# Reward Shaping
parser.add_argument('--fn_state_dict_path', type=str, default='',
                    help='(Aborted) Path to the saved fact network model')
parser.add_argument('--fn_kg_state_dict_path', type=str, default='',
                    help='(Aborted) Path to the saved knowledge graph embeddings used by a fact network')
parser.add_argument('--reward_shaping_threshold', type=float, default=0,
		            help='Threshold cut off of reward shaping scores (default: 0)')
parser.add_argument('--mu', type=float, default=1.0,
                    help='Weight over the estimated reward (default: 1.0)')


# Search Decoding
parser.add_argument('--beam_size', type=int, default=100,
                    help='size of beam used in beam search inference (default: 100)')
parser.add_argument('--mask_test_false_negatives', type=bool, default=False,
                    help='mask false negative examples in the dev/test set during decoding (default: False. This flag '
                         'was implemented for sanity checking and was not used in any experiment.)')
parser.add_argument('--visualize_paths', action='store_true',
                    help='generate path visualizations during inference (default: False)')
parser.add_argument('--save_beam_search_paths', action='store_true',
                    help='save the decoded path into a CSV file (default: False)')

parser.add_argument('--remark', type=str, default='0')

parser.add_argument('--eval_with_train', action='store_true')
parser.add_argument('--boltzmann_exploration', action='store_true')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature coefficient for Boltzmann exploration')
parser.add_argument('--beam_search_with_q_value', action='store_true')
parser.add_argument('--target_net_dropout', action='store_true')

args = parser.parse_args()