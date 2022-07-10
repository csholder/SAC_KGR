"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Compute Evaluation Metrics.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/evaluation.py
"""

import numpy as np
import pickle

import torch

from src.parse_args import args
from src.common.data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID


def hits_and_ranks(examples, scores, all_answers, logger, verbose=False):
    """
    Compute ranking based metrics.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, r, e2 = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    error_hits_0_examples, error_hits_1_examples, error_hits_10_examples, error_hits_3_examples, error_hits_5_examples = [], [], [], [], []
    logger.info('examples size: {}'.format(len(examples)))
    for i, example in enumerate(examples):
        e1, r, e2 = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
                            error_hits_0_examples.append(i)
                        else:
                            error_hits_1_examples.append(i)  # 1< hit <=3
                    else:
                        error_hits_3_examples.append(i)  # 3< hit <=5
                else:
                    error_hits_5_examples.append(i)  # 5< hit <=10
            else:
                error_hits_10_examples.append(i)  # 10< hit
            mrr += 1.0 / (pos + 1)
        else:
            error_hits_10_examples.append(i)

    # logger.info('number of == 0: {}\n'.format(len(error_hits_0_examples)))
    # logger.info('number of 3 > >= 1: {}\n'.format(len(error_hits_1_examples)))
    # logger.info('number of 5 > >= 3: {}\n'.format(len(error_hits_3_examples)))
    # logger.info('number of 10 > >= 5: {}\n'.format(len(error_hits_5_examples)))
    # logger.info('number of >= 10: {}\n'.format(len(error_hits_10_examples)))

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        logger.info('Hits@1 = {}'.format(hits_at_1))
        logger.info('Hits@3 = {}'.format(hits_at_3))
        logger.info('Hits@5 = {}'.format(hits_at_5))
        logger.info('Hits@10 = {}'.format(hits_at_10))
        logger.info('MRR = {}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr, error_hits_0_examples, error_hits_1_examples, \
           error_hits_3_examples, error_hits_5_examples, error_hits_10_examples


def hits_at_k(examples, scores, all_answers, logger, verbose=False):
    """
    Hits at k metrics.
    :param examples: List of triples and labels (+/-).
    :param pred_targets:
    :param scores:
    :param all_answers:
    :param verbose:
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, r, e2 = example
        e2_multi = list(all_answers[e1][r]) + dummy_mask
        # save the relevant prediction
        target_score = scores[i, e2]
        # mask all false negatives
        scores[i][e2_multi] = 0
        scores[i][dummy_mask] = 0
        # write back the save prediction
        scores[i][e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    for i, example in enumerate(examples):
        e1, r, e2 = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if pos:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)

    if verbose:
        logger.info('Hits@1 = {}'.format(hits_at_1))
        logger.info('Hits@3 = {}'.format(hits_at_3))
        logger.info('Hits@5 = {}'.format(hits_at_5))
        logger.info('Hits@10 = {}'.format(hits_at_10))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10


def hits_and_ranks_by_seen_queries(examples, scores, all_answers, seen_queries, logger, verbose=False):
    seen_exps, unseen_exps = [], []
    seen_ids, unseen_ids = [], []
    for i, example in enumerate(examples):
        e1, r, e2 = example
        if (e1, r) in seen_queries:
            seen_exps.append(example)
            seen_ids.append(i)
        else:
            unseen_exps.append(example)
            unseen_ids.append(i)

    _, _, _, _, seen_mrr, _, _, _, _, _ = hits_and_ranks(seen_exps, scores[seen_ids], all_answers, logger,
                                                         verbose=False)
    _, _, _, _, unseen_mrr, _, _, _, _, _ = hits_and_ranks(unseen_exps, scores[unseen_ids], all_answers, logger,
                                                           verbose=False)
    if verbose:
        logger.info('MRR on seen queries: {}'.format(seen_mrr))
        logger.info('MRR on unseen queries: {}'.format(unseen_mrr))
    return seen_mrr, unseen_mrr


def hits_and_ranks_by_relation_type(examples, scores, all_answers, relation_by_types, logger, verbose=False):
    to_M_rels, to_1_rels = relation_by_types
    to_M_exps, to_1_exps = [], []
    to_M_ids, to_1_ids = [], []
    for i, example in enumerate(examples):
        e1, r, e2 = example
        if r in to_M_rels:
            to_M_exps.append(example)
            to_M_ids.append(i)
        else:
            to_1_exps.append(example)
            to_1_ids.append(i)

    _, _, _, _, to_m_mrr, _, _, _, _, _ = hits_and_ranks(to_M_exps, scores[to_M_ids], all_answers, logger,
                                                         verbose=False)
    _, _, _, _, to_1_mrr, _, _, _, _, _ = hits_and_ranks(to_1_exps, scores[to_1_ids], all_answers, logger,
                                                         verbose=False)
    if verbose:
        logger.info('MRR on to-M relations: {}'.format(to_m_mrr))
        logger.info('MRR on to-1 relations: {}'.format(to_1_mrr))
    return to_m_mrr, to_1_mrr


def link_MAP(examples, scores, labels, all_answers, logger, verbose=False):
    """
    Per-query mean average precision.
    """
    assert (len(examples) == len(scores))
    queries = {}
    for i, example in enumerate(examples):
        e1, r, e2 = example
        if not e1 in queries:
            queries[e1] = []
        queries[e1].append((examples[i], labels[i], scores[i][e2]))

    aps = []
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]

    for e1 in queries:
        ranked_examples = sorted(queries[e1], key=lambda x: x[2], reverse=True)
        acc_precision, offset, num_pos = 0, 0, 0
        for i in range(len(ranked_examples)):
            triple, label, score = ranked_examples[i]
            _, r, e2 = triple
            if label == '+':
                num_pos += 1
                acc_precision += float(num_pos) / (i + 1 - offset)
            else:
                answer_set = {}
                if e1 in all_answers and r in all_answers[e1]:
                    answer_set = all_answers[e1][r]
                if e2 in answer_set or e2 in dummy_mask:
                    logger.info('False negative found: {}'.format(triple))
                    offset += 1
        if num_pos > 0:
            ap = acc_precision / num_pos
            aps.append(ap)
    map = np.mean(aps)
    if verbose:
        logger.info('MAP = {:.3f}'.format(map))
    return map


def export_error_cases(examples, scores, all_answers, output_path, logger):
    """
    Export indices of examples to which the top-1 prediction is incorrect.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, r, e2 = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    top_1_errors, top_10_errors = [], []
    for i, example in enumerate(examples):
        e1, r, e2 = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) <= 0 or pos[0] > 0:
            top_1_errors.append(i)
        if len(pos) <= 0 or pos[0] > 9:
            top_10_errors.append(i)
    with open(output_path, 'wb') as o_f:
        pickle.dump([top_1_errors, top_10_errors], o_f)

    logger.info('{}/{} top-1 error cases written to {}'.format(len(top_1_errors), len(examples), output_path))
    logger.info('{}/{} top-10 error cases written to {}'.format(len(top_10_errors), len(examples), output_path))

