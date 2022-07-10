from src.common.common_class import TrainFreq, TrainFrequencyUnit

import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable
import numpy as np
import random
from itertools import zip_longest


EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31


def get_action_embedding(action, kg):
    r, e = action
    relation_embedding = kg.get_relation_embeddings(r)
    entity_embedding = kg.get_entity_embeddings(e)
    action_embedding = th.cat([relation_embedding, entity_embedding], dim=-1)
    return action_embedding


def should_collect_more_steps(
    train_freq: TrainFreq,
    num_collected_steps: int,
    num_collected_episodes: int,
) -> bool:
    """
    Helper used in ``collect_rollouts()`` of off-policy algorithms
    to determine the termination condition.

    :param train_freq: How much experience should be collected before updating the policy.
    :param num_collected_steps: The number of already collected steps.
    :param num_collected_episodes: The number of already collected episodes.
    :return: Whether to continue or not collecting experience
        by doing rollouts of the current policy.
    """
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return num_collected_steps < train_freq.frequency

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return num_collected_episodes < train_freq.frequency

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer:
    :param learning_rate:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[th.nn.Parameter],
    target_params: Iterable[th.nn.Parameter],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def batch_lookup(M, idx, vector_output=True):
    """
    Perform batch lookup on matrix M using indices idx.
    :param M: (Variable) [batch_size, seq_len] Each row of M is an independent population.
    :param idx: (Variable) [batch_size, sample_size] Each row of idx is a list of sample indices.
    :param vector_output: If set, return a 1-D vector when sample size is 1.
    :return samples: [batch_size, sample_size] samples[i, j] = M[idx[i, j]]
    """
    batch_size, w = M.size()
    batch_size2, sample_size = idx.size()
    assert (batch_size == batch_size2)

    if sample_size == 1 and vector_output:
        # M_np = M.cpu().detach().numpy()
        # idx_np = idx.cpu().detach().numpy()
        # np.save('M.npy', M_np)
        # np.save('idx.npy', idx_np)
        samples = th.gather(M, 1, idx).view(-1)
    else:
        samples = th.gather(M, 1, idx)
    return samples


def format_batch(batch_data, num_labels=-1, num_tiles=1, device=th.device('cpu')):
    """
    Convert batched tuples to the tensors accepted by the NN.
    """
    def convert_to_binary_multi_subject(e1):
        e1_label = th.zeros([len(e1), num_labels], device=device)
        for i in range(len(e1)):
            e1_label[i][e1[i]] = 1
        return e1_label

    def convert_to_binary_multi_object(e2):
        e2_label = th.zeros([len(e2), num_labels], device=device)
        for i in range(len(e2)):
            e2_label[i][e2[i]] = 1
        return e2_label

    batch_e1, batch_r, batch_e2 = [], [], []
    for i in range(len(batch_data)):
        e1, r, e2 = batch_data[i]
        batch_e1.append(e1)
        batch_r.append(r)
        batch_e2.append(e2)
    batch_e1 = th.LongTensor(batch_e1).to(device)
    batch_r = th.LongTensor(batch_r).to(device)
    if type(batch_e2[0]) is list:
        batch_e2 = convert_to_binary_multi_object(batch_e2)
    elif type(batch_e1[0]) is list:
        batch_e1 = convert_to_binary_multi_subject(batch_e1)
    else:
        batch_e2 = th.LongTensor(batch_e2).to(device)
    # Rollout multiple times for each example
    if num_tiles > 1:
        batch_e1 = tile_along_beam(batch_e1, num_tiles)
        batch_r = tile_along_beam(batch_r, num_tiles)
        batch_e2 = tile_along_beam(batch_e2, num_tiles)
    return batch_e1, batch_r, batch_e2


def pad_and_cat_action_space(kg, action_spaces, inv_offset):
    db_r_space, db_e_space, db_action_mask = [], [], []
    for (r_space, e_space), action_mask in action_spaces:
        db_r_space.append(r_space)
        db_e_space.append(e_space)
        db_action_mask.append(action_mask)
    r_space = pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
    e_space = pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
    action_mask = pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
    action_space = ((r_space, e_space), action_mask)
    return action_space


def pad_and_cat(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return th.cat(padded_a, dim=0)


def unique_max(unique_x, x, values, marker_2D=None):
    unique_interval = 100
    unique_values, unique_indices = [], []
    # prevent memory explotion during decoding
    for i in range(0, len(unique_x), unique_interval):
        unique_x_b = unique_x[i:i+unique_interval]
        marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float()
        values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT
        unique_values_b, unique_idx_b = values_2D.max(dim=1)
        unique_values.append(unique_values_b)
        unique_indices.append(unique_idx_b)
    unique_values = th.cat(unique_values)
    unique_idx = th.cat(unique_indices)
    return unique_values, unique_idx


def format_path(path_trace, kg):
    def get_most_recent_relation(j):
        relation_id = int(path_trace[j][0])
        if relation_id == kg.self_edge:
            return '<null>'
        else:
            return kg.id2relation[relation_id]

    def get_most_recent_entity(j):
        return kg.id2entity[int(path_trace[j][1])]

    path_str = get_most_recent_entity(0)
    for j in range(1, len(path_trace)):
        rel = get_most_recent_relation(j)
        # if not rel.endswith('_inv'):
        #     path_str += ' -{}-> '.format(rel)
        # else:
        #     path_str += ' <-{}- '.format(rel[:-4])
        path_str += '\t{}\t'.format(rel)
        path_str += get_most_recent_entity(j)
    return path_str


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def tile_along_beam(v, beam_size, dim=0):
    """
    Tile a tensor along a specified dimension for the specified beam size.
    :param v: Input tensor.
    :param beam_size: Beam size.
    """
    if dim == -1:
        dim = len(v.size()) - 1
    v = v.unsqueeze(dim + 1)
    v = th.cat([v] * beam_size, dim=dim+1)
    new_size = []
    for i, d in enumerate(v.size()):
        if i == dim + 1:
            new_size[-1] *= d
        else:
            new_size.append(d)
    return v.view(new_size)


def rearrange_vector_list(l, offset):
    for i, v in enumerate(l):
        l[i] = v[offset]


def ones_var_cuda(s, requires_grad=False, device=th.device('cpu')):
    return nn.Parameter(th.ones(s), requires_grad=requires_grad).to(device)


def zeros_var_cuda(s, requires_grad=False, device=th.device('cpu')):
    return nn.Parameter(th.zeros(s), requires_grad=requires_grad).to(device)


def int_fill_var_cuda(s, value, requires_grad=False, device=th.device('cpu')):
    return int_var_cuda((th.zeros(s) + value), requires_grad=requires_grad, device=device)


def int_var_cuda(x, requires_grad=False, device=th.device('cpu')):
    return nn.Parameter(x, requires_grad=requires_grad).long().to(device)


def var_cuda(x, requires_grad=False, device=th.device('cpu')):
    return nn.Parameter(x, requires_grad=requires_grad).to(device)


def safe_log(x):
    return th.log(x + EPSILON)


def entropy(p):
    return th.sum(-p * safe_log(p), 1)


def detach_module(mdl):
    for param in mdl.parameters():
        param.requires_grad = False