import torch as th

from src.common.common_class import Observation



def get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg):
    ground_truth_edge_mask = \
        ((e == e_s).unsqueeze(1) * (r_space == q.unsqueeze(1)) * (e_space == e_t.unsqueeze(1)))
    inv_q = kg.get_inv_relation_id_batch(q)
    inv_ground_truth_edge_mask = \
        ((e == e_t).unsqueeze(1) * (r_space == inv_q.unsqueeze(1)) * (e_space == e_s.unsqueeze(1)))
    return ((ground_truth_edge_mask + inv_ground_truth_edge_mask) * (e_s.unsqueeze(1) != kg.dummy_e)).float()


def get_answer_mask(e_space, e_s, q, kg):
    if kg.args.mask_test_false_negatives:
        answer_vectors = kg.all_object_vectors
    else:
        answer_vectors = kg.train_object_vectors
    answer_masks = []
    for i in range(len(e_space)):
        _e_s, _q = int(e_s[i]), int(q[i])
        if not _e_s in answer_vectors or not _q in answer_vectors[_e_s]:
            answer_vector = th.LongTensor([[kg.num_entities]]).to(e_s.device)
        else:
            answer_vector = answer_vectors[_e_s][_q]
        answer_mask = th.sum(e_space[i].unsqueeze(0) == answer_vector, dim=0).long()
        answer_masks.append(answer_mask)
    answer_mask = th.cat(answer_masks).view(len(e_space), -1)
    return answer_mask


def get_false_negative_mask(e_space, e_s, q, e_t, kg):        # 应该是为了考虑除了训练样例 (e_s,q,e_t) 中的 e_t，其它正确的尾实体都 mask 住
    answer_mask = get_answer_mask(e_space, e_s, q, kg)
    false_negative_mask = (answer_mask * (e_space != e_t.unsqueeze(1)).long()).float()
    return false_negative_mask


def validate_action_mask(action_mask):
    action_mask_min = action_mask.min()
    action_mask_max = action_mask.max()
    assert (action_mask_min == 0 or action_mask_min == 1)
    assert (action_mask_max == 0 or action_mask_max == 1)


def apply_action_masks(action_space, obs: Observation, kg):
    (r_space, e_space), action_mask = action_space
    e_s, q, e_t, path_length = obs.start_entity, obs.query_relation, obs.target_entity, obs.path_length
    current_e = obs.current_entity
    # Prevent the agent from selecting the ground truth edge
    ground_truth_edge_mask = get_ground_truth_edge_mask(current_e, r_space, e_space, e_s, q, e_t, kg)
    action_mask -= ground_truth_edge_mask
    validate_action_mask(action_mask)

    # Mask out false negatives in the final step
    if th.any(obs.last_step):
        false_negative_mask = get_false_negative_mask(e_space, e_s, q, e_t, kg)
        false_negative_mask *= obs.last_step.unsqueeze(dim=-1)
        action_mask *= (1 - false_negative_mask)
        validate_action_mask(action_mask)

    return (r_space, e_space), action_mask


def get_action_space_in_buckets(obs: Observation, kg):
    q, e_t, path_r, path_e, path_length = obs.query_relation, obs.target_entity, \
                                          obs.path_r, obs.path_e, obs.path_length
    current_e = obs.current_entity
    assert(len(current_e) == len(q))
    assert(len(current_e) == len(e_t))
    assert(len(current_e) == len(path_r))
    assert(len(current_e) == len(path_e))
    assert(len(current_e) == len(path_length))
    db_action_spaces, db_references, db_obs = [], [], []

    entity2bucketid = kg.entity2bucketid[current_e.tolist()]
    key1 = entity2bucketid[:, 0]
    key2 = entity2bucketid[:, 1]
    batch_ref = {}
    for i in range(len(current_e)):
        key = int(key1[i])
        if not key in batch_ref:
            batch_ref[key] = []
        batch_ref[key].append(i)
    for key in batch_ref:
        action_space = kg.action_space_buckets[key]
        l_batch_refs = batch_ref[key]
        g_bucket_ids = key2[l_batch_refs].tolist()
        r_space_b = action_space[0][0][g_bucket_ids]
        e_space_b = action_space[0][1][g_bucket_ids]
        action_mask_b = action_space[1][g_bucket_ids]
        path_r_b = path_r[l_batch_refs]
        path_e_b = path_e[l_batch_refs]
        path_length_b = path_length[l_batch_refs]
        q_b = q[l_batch_refs]
        e_t_b = e_t[l_batch_refs]
        obs_b = Observation(obs.num_rollout_steps, q_b, e_t_b, (path_r_b, path_e_b), path_length_b)
        action_space_b = ((r_space_b, e_space_b), action_mask_b)
        action_space_b = apply_action_masks(action_space_b, obs_b, kg)
        db_action_spaces.append(action_space_b)
        db_references.append(l_batch_refs)
        db_obs.append(obs_b)
    return db_action_spaces, db_references, db_obs


def get_action_space(obs: Observation, kg):
    current_e = obs.current_entity
    r_space, e_space = kg.action_space[0][0][current_e], kg.action_space[0][1][current_e]
    action_mask = kg.action_space[1][current_e]
    action_space = ((r_space, e_space), action_mask)
    return apply_action_masks(action_space, obs, kg)