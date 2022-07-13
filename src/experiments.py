import os, sys, copy
import torch as th

import random, logging
from collections import defaultdict as ddict

from src.common.knowledge_graph import KnowledgeGraph
import src.common.data_utils as data_utils
from src.emb.emb import EmbeddingBasedMethod
from src.emb.fact_network import ConvE
from src.sac.sac import SAC
from src.sac.sac_rs import SACrs
from src.pg.pg import PolicyGradient
from src.pg.rs_pg import RewardShapingPolicyGradient
from src.dqn.dqn import DQN
from src.dqn.rs_dqn import RewardShapingDQN
from src.reinforce.pg import REINFORCE
from src.reinforce.pn import GraphSearchPolicy
from src.reinforce.rs_pg import RewardShapingREINFORCE
from src.eval import hits_and_ranks
from src.parse_args import args


th.cuda.set_device(args.gpu)
args.device = th.device('cpu') if args.gpu == -1 else th.device('cuda:{}'.format(args.gpu))

th.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)

import wandb


def process_data(args):
    # data_dir = os.path.join('..', args.data_dir)
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test,
                                     args.add_reverse_relations)


def initialize_model_directory(args):
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''

    if args.model_name.startswith('embed'):
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_epochs,
            args.group_examples_by_query,
            args.num_wait_epochs,
            args.num_peek_epochs,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon,
            args.remark,
        )
    elif 'sac' in args.model_name:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.num_rollout_steps,
            args.num_epochs,
            args.group_examples_by_query,
            args.num_wait_epochs,
            args.num_peek_epochs,
            args.history_num_layers,
            args.buffer_size,
            args.train_freq_value,
            args.train_freq_unit,
            args.gradient_steps,
            args.n_critics,
            args.ent_coef,
            args.actor_learning_rate,
            args.critic_learning_rate,
            args.eof_learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.target_entropy,
            args.action_entropy_ratio,
            args.bandwidth,
            args.beta,
            args.run_analysis,
            args.add_reversed_training_edges,
            args.relation_only,
            args.remark
        )
    elif 'pg' in args.model_name:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.num_rollout_steps,
            args.num_epochs,
            args.num_wait_epochs,
            args.num_peek_epochs,
            args.history_num_layers,
            args.buffer_size,
            args.actor_learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta,
            args.run_analysis,
            args.group_examples_by_query,
            args.add_reversed_training_edges,
            args.relation_only,
            args.remark
        )
    elif 'reinforce' in args.model_name:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.num_rollout_steps,
            args.num_epochs,
            args.num_wait_epochs,
            args.num_peek_epochs,
            args.history_num_layers,
            args.buffer_size,
            args.actor_learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta,
            args.run_analysis,
            args.group_examples_by_query,
            args.add_reversed_training_edges,
            args.relation_only,
            args.remark
        )
    elif 'dqn' in args.model_name:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.num_rollout_steps,
            args.num_epochs,
            args.num_wait_epochs,
            args.num_peek_epochs,
            args.history_num_layers,
            args.buffer_size,
            args.critic_learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.exploration_initial_eps,
            args.exploration_final_eps,
            args.exploration_fraction,
            args.boltzmann_exploration,
            args.temperature,
            args.target_update_interval,
            args.bandwidth,
            args.beta,
            args.gamma,
            args.tau,
            args.run_analysis,
            args.group_examples_by_query,
            args.add_reversed_training_edges,
            args.relation_only,
            args.beam_search_with_q_value,
            args.remark
        )
    else:
        raise NotImplementedError

    if args.use_wandb:
        wandb.init(project="SAC_KGR", entity="cs_holder", name='{}-{}'.format(args.model_name, hyperparam_sig))

    model_sub_dir = '{}-{}{}-{}'.format(
        dataset,
        args.model_name,
        reverse_edge_tag,
        hyperparam_sig
    )

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

    log_dir = os.path.join(args.log_dir, dataset, args.model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(hyperparam_sig)
    logger.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(format)
    th = logging.FileHandler(filename=os.path.join(log_dir, hyperparam_sig + '.txt'), encoding='utf-8')
    th.setFormatter(format)
    logger.addHandler(sh)
    logger.addHandler(th)

    logger.info(vars(args))

    args.logger = logger


def construct_model(args):
    kg = KnowledgeGraph(args)
    if args.model_name == 'embed.conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn, logger=args.logger)
    elif args.model_name.startswith('rl.rs.sac'):
        fn_model = args.model_name.split('.')[3]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        else:
            raise NotImplementedError
        lf = SACrs(
            args,
            kg,
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            history_dim=args.history_dim,
            history_num_layers=args.history_num_layers,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            eof_learning_rate=args.eof_learning_rate,
            xavier_initialization=args.xavier_initialization,
            relation_only=args.relation_only,
            buffer_size=args.buffer_size,  # 1e6
            batch_size=args.buffer_batch_size,
            learning_starts=args.learning_starts,
            action_dropout_rate=args.action_dropout_rate,
            net_arch=args.net_arch,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq_value, args.train_freq_unit),
            gradient_steps=args.gradient_steps,
            n_critics=args.n_critics,
            ent_coef=args.ent_coef,
            target_update_interval=args.target_update_interval,
            target_entropy=args.target_entropy,
            action_entropy_ratio=args.action_entropy_ratio,
            max_grad_norm=args.max_grad_norm,
            replay_buffer_class=args.replay_buffer_class,
            policy_class=args.policy_class,
            mu=args.mu,
            fn_kg=fn_kg,
            fn=fn,
            verbose=args.verbose,
        )
    elif args.model_name.startswith('rl.sac'):
        lf = SAC(
            args,
            kg,
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            history_dim=args.history_dim,
            history_num_layers=args.history_num_layers,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            eof_learning_rate=args.eof_learning_rate,
            xavier_initialization=args.xavier_initialization,
            relation_only=args.relation_only,
            buffer_size=args.buffer_size,  # 1e6
            batch_size=args.buffer_batch_size,
            learning_starts=args.learning_starts,
            ff_dropout_rate=args.ff_dropout_rate,
            action_dropout_rate=args.action_dropout_rate,
            net_arch=args.net_arch,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq_value, args.train_freq_unit),
            gradient_steps=args.gradient_steps,
            n_critics=args.n_critics,
            ent_coef=args.ent_coef,
            target_update_interval=args.target_update_interval,
            target_entropy=args.target_entropy,
            action_entropy_ratio=args.action_entropy_ratio,
            max_grad_norm=args.grad_norm,
            replay_buffer_class=args.replay_buffer_class,
            policy_class=args.policy_class,
            verbose=args.verbose,
        )
        if args.use_wandb:
            wandb.config = {
            'entity_dim': args.entity_dim,
            'relation_dim': args.relation_dim,
            'history_dim': args.history_dim,
            'history_num_layers': args.history_num_layers,
            'learning_rate': args.critic_learning_rate,
            'actor_learning_rate': args.actor_learning_rate,
            'critic_learning_rate': args.critic_learning_rate,
            'eof_learning_rate': args.eof_learning_rate,
            'buffer_size': args.buffer_size,
            'buffer_batch_size': args.buffer_batch_size,
            'learning_starts': args.learning_starts,
            'ff_dropout_rate': args.ff_dropout_rate,
            'action_dropout_rate': args.action_dropout_rate,
            'tau': args.tau,
            'gamma': args.gamma,
            'train_freq_value': args.train_freq_value,
            'train_freq_unit': args.train_freq_unit,
            'policy_class': args.policy_class,
            'replay_buffer_class': args.replay_buffer_class,
            'gradient_steps': args.gradient_steps,
            'n_critics': args.n_critics,
            'ent_coef': args.ent_coef,
            'target_update_interval': args.target_update_interval,
            'target_entropy': args.target_entropy,
            'action_entropy_ratio': args.action_entropy_ratio,
            'exploration_fraction': args.exploration_fraction,
            'exploration_initial_eps': args.exploration_initial_eps,
            'exploration_final_eps': args.exploration_final_eps,
            'boltzmann_exploration': args.boltzmann_exploration,
            'temperature': args.temperature,
            'max_grad_norm': args.grad_norm,
            'xavier_initialization': args.xavier_initialization,
            'relation_only': args.relation_only,
        }
    elif args.model_name.startswith('rl.pg'):
        lf = PolicyGradient(
            args,
            kg,
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            history_dim=args.history_dim,
            history_num_layers=args.history_num_layers,
            actor_learning_rate=args.actor_learning_rate,
            ff_dropout_rate=args.ff_dropout_rate,
            action_dropout_rate=args.action_dropout_rate,
            net_arch=args.net_arch,
            policy_class=args.policy_class,
            xavier_initialization=args.xavier_initialization,
            relation_only=args.relation_only,
        )
    elif args.model_name.startswith('rl.rs.pg'):
        fn_model = args.model_name.split('.')[3]
        fn_args = copy.deepcopy(args)
        fn_args.model_name = fn_model
        fn = ConvE(fn_args, kg.num_entities)
        fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(
            args,
            kg,
            fn_kg,
            fn,
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            history_dim=args.history_dim,
            history_num_layers=args.history_num_layers,
            actor_learning_rate=args.actor_learning_rate,
            ff_dropout_rate=args.ff_dropout_rate,
            action_dropout_rate=args.action_dropout_rate,
            net_arch=args.net_arch,
            policy_class=args.policy_class,
            xavier_initialization=args.xavier_initialization,
            relation_only=args.relation_only,
        )
    elif args.model_name.startswith('rl.dqn'):
        lf = DQN(
            args,
            kg,
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            history_dim=args.history_dim,
            history_num_layers=args.history_num_layers,
            learning_rate=args.critic_learning_rate,
            buffer_size=args.buffer_size,
            buffer_batch_size=args.buffer_batch_size,
            magnification=args.magnification,
            learning_starts=args.learning_starts,
            ff_dropout_rate=args.ff_dropout_rate,
            action_dropout_rate=args.action_dropout_rate,
            net_arch=args.net_arch,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq_value, args.train_freq_unit),
            policy_class=args.policy_class,
            replay_buffer_class=args.replay_buffer_class,
            gradient_steps=args.gradient_steps,
            n_critics=args.n_critics,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            boltzmann_exploration=args.boltzmann_exploration,
            temperature=args.temperature,
            max_grad_norm=args.grad_norm,
            xavier_initialization=args.xavier_initialization,
            relation_only=args.relation_only,
            beam_search_with_q_value=args.beam_search_with_q_value,
            target_net_dropout=args.target_net_dropout,
        )
        if args.use_wandb:
            wandb.config = {
            'entity_dim': args.entity_dim,
            'relation_dim': args.relation_dim,
            'history_dim': args.history_dim,
            'history_num_layers': args.history_num_layers,
            'learning_rate': args.critic_learning_rate,
            'buffer_size': args.buffer_size,
            'buffer_batch_size': args.buffer_batch_size,
            'learning_starts': args.learning_starts,
            'ff_dropout_rate': args.ff_dropout_rate,
            'action_dropout_rate': args.action_dropout_rate,
            'tau': args.tau,
            'gamma': args.gamma,
            'train_freq_value': args.train_freq_value,
            'train_freq_unit': args.train_freq_unit,
            'policy_class': args.policy_class,
            'replay_buffer_class': args.replay_buffer_class,
            'gradient_steps': args.gradient_steps,
            'target_update_interval': args.target_update_interval,
            'exploration_fraction': args.exploration_fraction,
            'exploration_initial_eps': args.exploration_initial_eps,
            'exploration_final_eps': args.exploration_final_eps,
            'boltzmann_exploration': args.boltzmann_exploration,
            'temperature': args.temperature,
            'max_grad_norm': args.grad_norm,
            'xavier_initialization': args.xavier_initialization,
            'relation_only': args.relation_only,
            'beam_search_with_q_value': args.beam_search_with_q_value,
            'target_net_dropout': args.target_net_dropout,
        }
    elif args.model_name.startswith('rl.rs.dqn'):
        fn_model = args.model_name.split('.')[3]
        fn_args = copy.deepcopy(args)
        fn_args.model_name = fn_model
        fn = ConvE(fn_args, kg.num_entities)
        fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingDQN(
            args,
            kg,
            fn_kg,
            fn,
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            history_dim=args.history_dim,
            history_num_layers=args.history_num_layers,
            learning_rate=args.critic_learning_rate,
            buffer_size=args.buffer_size,
            buffer_batch_size=args.buffer_batch_size,
            magnification=args.magnification,
            learning_starts=args.learning_starts,
            ff_dropout_rate=args.ff_dropout_rate,
            action_dropout_rate=args.action_dropout_rate,
            net_arch=args.net_arch,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq_value, args.train_freq_unit),
            policy_class=args.policy_class,
            replay_buffer_class=args.replay_buffer_class,
            gradient_steps=args.gradient_steps,
            n_critics=args.n_critics,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            boltzmann_exploration=args.boltzmann_exploration,
            temperature=args.temperature,
            max_grad_norm=args.grad_norm,
            xavier_initialization=args.xavier_initialization,
            relation_only=args.relation_only,
            beam_search_with_q_value=args.beam_search_with_q_value,
            target_net_dropout=args.target_net_dropout,
        )
        if args.use_wandb:
            wandb.config = {
            'entity_dim': args.entity_dim,
            'relation_dim': args.relation_dim,
            'history_dim': args.history_dim,
            'history_num_layers': args.history_num_layers,
            'learning_rate': args.critic_learning_rate,
            'buffer_size': args.buffer_size,
            'buffer_batch_size': args.buffer_batch_size,
            'learning_starts': args.learning_starts,
            'ff_dropout_rate': args.ff_dropout_rate,
            'action_dropout_rate': args.action_dropout_rate,
            'tau': args.tau,
            'gamma': args.gamma,
            'train_freq_value': args.train_freq_value,
            'train_freq_unit': args.train_freq_unit,
            'policy_class': args.policy_class,
            'replay_buffer_class': args.replay_buffer_class,
            'gradient_steps': args.gradient_steps,
            'target_update_interval': args.target_update_interval,
            'exploration_fraction': args.exploration_fraction,
            'exploration_initial_eps': args.exploration_initial_eps,
            'exploration_final_eps': args.exploration_final_eps,
            'boltzmann_exploration': args.boltzmann_exploration,
            'temperature': args.temperature,
            'max_grad_norm': args.grad_norm,
            'xavier_initialization': args.xavier_initialization,
            'relation_only': args.relation_only,
            'beam_search_with_q_value': args.beam_search_with_q_value,
            'target_net_dropout': args.target_net_dropout,
        }
    elif args.model_name.startswith('rl.reinforce'):
        pn = GraphSearchPolicy(args)
        lf = REINFORCE(args, kg, pn)
    elif args.model_name.startswith('rl.rs.reinforce'):
        pn = GraphSearchPolicy(args)
        fn_model = args.model_name.split('.')[3]
        fn_args = copy.deepcopy(args)
        fn_args.model_name = 'embed.' + fn_model
        fn = ConvE(fn_args, kg.num_entities)
        fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingREINFORCE(args, kg, pn, fn_kg, fn)
    else:
        raise NotImplementedError
    return lf


def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path


def train(args, lf):
    train_path = data_utils.get_train_path(args)
    if args.eval_with_train:
        dev_path = os.path.join(args.data_dir, 'dev_from_train.txt')
    else:
        dev_path = os.path.join(args.data_dir, 'dev.txt')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities,
                                       add_reverse_relations=args.add_reversed_training_edges)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)


def inference(args, lf):
    lf.batch_size = args.eval_batch_size
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics = {
        'tail_dev': {},
        'head_dev': {},
        'dev': {},
        'tail_test': {},
        'head_test': {},
        'test': {}
    }

    test_path = os.path.join(args.data_dir, 'test.txt')
    # dev_data = data_utils.load_triples(
    #     dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    test_data = data_utils.load_triples(
        test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False,
        add_reverse_relations=args.add_reversed_training_edges)
    tail_test_data = [test_data[idx * 2] for idx in range(len(test_data) // 2)]
    head_test_data = [test_data[idx * 2 + 1] for idx in range(len(test_data) // 2)]
    print('Test set performance:')
    test_query_path_dict = ddict(list)
    if args.model_name.startswith('rl'):
        tail_pred_scores, test_query_path_dict = lf.forward(tail_test_data, query_path_dict=test_query_path_dict,
                                                            verbose=args.save_beam_search_paths)
        head_pred_scores, test_query_path_dict = lf.forward(head_test_data, query_path_dict=test_query_path_dict,
                                                            verbose=args.save_beam_search_paths)
    else:
        tail_pred_scores, _ = lf.forward(tail_test_data)
        head_pred_scores, _ = lf.forward(head_test_data)
    tail_test_metrics = hits_and_ranks(tail_test_data, tail_pred_scores, lf.kg.all_objects, lf.logger,
                                       verbose=True)
    head_test_metrics = hits_and_ranks(head_test_data, head_pred_scores, lf.kg.all_objects, lf.logger,
                                       verbose=True)

    eval_metrics['tail_test']['hits_at_1'] = tail_test_metrics[0]
    eval_metrics['tail_test']['hits_at_3'] = tail_test_metrics[1]
    eval_metrics['tail_test']['hits_at_5'] = tail_test_metrics[2]
    eval_metrics['tail_test']['hits_at_10'] = tail_test_metrics[3]
    eval_metrics['tail_test']['mrr'] = tail_test_metrics[4]
    eval_metrics['tail_test']['error_hits_0_example'] = tail_test_metrics[5]
    eval_metrics['tail_test']['error_hits_1_example'] = tail_test_metrics[6]
    eval_metrics['tail_test']['error_hits_3_example'] = tail_test_metrics[7]
    eval_metrics['tail_test']['error_hits_5_example'] = tail_test_metrics[8]
    eval_metrics['tail_test']['error_hits_10_example'] = tail_test_metrics[9]

    eval_metrics['head_test']['hits_at_1'] = head_test_metrics[0]
    eval_metrics['head_test']['hits_at_3'] = head_test_metrics[1]
    eval_metrics['head_test']['hits_at_5'] = head_test_metrics[2]
    eval_metrics['head_test']['hits_at_10'] = head_test_metrics[3]
    eval_metrics['head_test']['mrr'] = head_test_metrics[4]
    eval_metrics['head_test']['error_hits_0_example'] = head_test_metrics[5]
    eval_metrics['head_test']['error_hits_1_example'] = head_test_metrics[6]
    eval_metrics['head_test']['error_hits_3_example'] = head_test_metrics[7]
    eval_metrics['head_test']['error_hits_5_example'] = head_test_metrics[8]
    eval_metrics['head_test']['error_hits_10_example'] = head_test_metrics[9]

    eval_metrics['test']['hits_at_1'] = (tail_test_metrics[0] + head_test_metrics[0]) / 2
    eval_metrics['test']['hits_at_3'] = (tail_test_metrics[1] + head_test_metrics[1]) / 2
    eval_metrics['test']['hits_at_5'] = (tail_test_metrics[2] + head_test_metrics[2]) / 2
    eval_metrics['test']['hits_at_10'] = (tail_test_metrics[3] + head_test_metrics[3]) / 2
    eval_metrics['test']['mrr'] = (tail_test_metrics[4] + head_test_metrics[4]) / 2

    lf.logger.info(
        'Tail Hits@1 = {}\tHead Hits@1 = {}\tHits@1 = {}'.format(tail_test_metrics[0], head_test_metrics[0],
                                                                 eval_metrics['test']['hits_at_1']))
    lf.logger.info(
        'Tail Hits@3 = {}\tHead Hits@3 = {}\tHits@3 = {}'.format(tail_test_metrics[1], head_test_metrics[1],
                                                                 eval_metrics['test']['hits_at_3']))
    lf.logger.info(
        'Tail Hits@5 = {}\tHead Hits@5 = {}\tHits@5 = {}'.format(tail_test_metrics[2], head_test_metrics[2],
                                                                 eval_metrics['test']['hits_at_5']))
    lf.logger.info(
        'Tail Hits@10 = {}\tHead Hits@10 = {}\tHits@10 = {}'.format(tail_test_metrics[3], head_test_metrics[3],
                                                                    eval_metrics['test']['hits_at_10']))
    lf.logger.info('Tail MRR = {}\tHead MRR = {}\tMRR = {}'.format(tail_test_metrics[4], head_test_metrics[4],
                                                                   eval_metrics['test']['mrr']))

    if args.model_name.startswith('rl'):
        with open(os.path.join(args.model_dir, 'test_beam_search_paths.txt'), 'w', encoding='utf-8') as fout:
            for query, paths in test_query_path_dict.items():
                for path in paths:
                    path_prob, path_str = path
                    fout.write('Query: <{},{},{}>   Path: <{}>  Prob: <{}>\n'.format(lf.kg.id2entity[query[0]],
                                                                                     lf.kg.id2relation[query[2]],
                                                                                     lf.kg.id2entity[query[1]],
                                                                                     path_str, path_prob))
        # test_query_rel_path_dict = ddict(list)
        for hit in [1, 3, 5, 10]:
            with open(os.path.join(args.model_dir, 'test_beam_search_paths_{}_tail.txt'.format(hit)), 'w',
                      encoding='utf-8') as fout:
                for idx in eval_metrics['tail_test']['error_hits_{}_example'.format(hit)]:
                    e1, e2, r = test_data[idx]
                    paths = test_query_path_dict[(e1, e2, r)]
                    for path in paths:
                        path_prob, path_str = path
                        fout.write('{}\t{}\t{} <= {}\n'.format(lf.kg.id2entity[e1], lf.kg.id2relation[r],
                                                               lf.kg.id2entity[e2], path_str))
                        # elems = path_str.strip().split('\t')
                        # if elems[-1] == lf.kg.id2entity[e2] and hit == 0:
                        #     test_query_rel_path_dict[(e1, r, e2)].append(path_str)

            with open(os.path.join(args.model_dir, 'test_beam_search_paths_{}_head.txt'.format(hit)), 'w',
                      encoding='utf-8') as fout:
                for idx in eval_metrics['head_test']['error_hits_{}_example'.format(hit)]:
                    e1, e2, r = test_data[idx]
                    paths = test_query_path_dict[(e1, e2, r)]
                    for path in paths:
                        path_prob, path_str = path
                        fout.write('{}\t{}\t{} <= {}\n'.format(lf.kg.id2entity[e1], lf.kg.id2relation[r],
                                                               lf.kg.id2entity[e2], path_str))
                        # elems = path_str.strip().split('\t')
                        # if elems[-1] == lf.kg.id2entity[e2] and hit == 0:
                        #     test_query_rel_path_dict[(e1, r, e2)].append(path_str)

        with open(os.path.join(args.model_dir, 'test_beam_search_valid_paths_tail.txt'), 'w',
                  encoding='utf-8') as fout:
            for idx in eval_metrics['tail_test']['error_hits_0_example']:
                e1, e2, r = test_data[idx]
                paths = test_query_path_dict[(e1, e2, r)]
                for path in paths:
                    # print('path: ', path)
                    path_prob, path_str = path
                    fout.write(
                        '{}\t{}\t{} <= {}\n'.format(lf.kg.id2entity[e1], lf.kg.id2relation[r], lf.kg.id2entity[e2],
                                                    path_str))

        with open(os.path.join(args.model_dir, 'test_beam_search_valid_paths_head.txt'), 'w',
                  encoding='utf-8') as fout:
            for idx in eval_metrics['head_test']['error_hits_0_example']:
                e1, e2, r = test_data[idx]
                paths = test_query_path_dict[(e1, e2, r)]
                for path in paths:
                    # print('path: ', path)
                    path_prob, path_str = path
                    fout.write(
                        '{}\t{}\t{} <= {}\n'.format(lf.kg.id2entity[e1], lf.kg.id2relation[r], lf.kg.id2entity[e2],
                                                    path_str))

    for idx, data in enumerate(tail_test_data):
        if idx in eval_metrics['tail_test']['error_hits_10_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_10_tail.txt')
        elif idx in eval_metrics['tail_test']['error_hits_5_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_5_tail.txt')
        elif idx in eval_metrics['tail_test']['error_hits_3_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_3_tail.txt')
        elif idx in eval_metrics['tail_test']['error_hits_1_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_1_tail.txt')
        elif idx in eval_metrics['tail_test']['error_hits_0_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_0_tail.txt')
        else:
            raise ValueError

        with open(out_file, 'a+', encoding='utf-8') as fout:
            fout.write('{}\t{}\t{}\n'.format(lf.kg.id2entity[data[0]], lf.kg.id2relation[data[1]],
                                             lf.kg.id2entity[data[2]]))
            fout.close()

    for idx, data in enumerate(head_test_data):
        if idx in eval_metrics['head_test']['error_hits_10_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_10_head.txt')
        elif idx in eval_metrics['head_test']['error_hits_5_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_5_head.txt')
        elif idx in eval_metrics['head_test']['error_hits_3_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_3_head.txt')
        elif idx in eval_metrics['head_test']['error_hits_1_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_1_head.txt')
        elif idx in eval_metrics['head_test']['error_hits_0_example']:
            out_file = os.path.join(args.model_dir, 'test_error_data_0_head.txt')
        else:
            raise ValueError

        with open(out_file, 'a+', encoding='utf-8') as fout:
            fout.write('{}\t{}\t{}\n'.format(lf.kg.id2entity[data[0]], lf.kg.id2relation[data[1]],
                                             lf.kg.id2entity[data[2]]))
            fout.close()

    return eval_metrics


def run_experiment(args):
    if args.process_data:
        # Process knowledge graph data
        process_data(args)
    else:
        with th.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            initialize_model_directory(args)
            lf = construct_model(args)
            lf.to(args.device)

            if args.train:
                train(args, lf)
                inference(args, lf)
            elif args.inference:
                inference(args, lf)


if __name__ == "__main__":
    run_experiment(args)