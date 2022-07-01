import os
import numpy as np


def transfer_hrt_to_htr(data_dir):
    train_triples = []
    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            train_triples.append('\t'.join([elems[0], elems[2], elems[1]]))

    with open(os.path.join(data_dir, 'train.triples'), 'w', encoding='utf-8') as fout:
        for triple in train_triples:
            fout.write(triple + '\n')

    with open(os.path.join(data_dir, 'raw.kb'), 'w', encoding='utf-8') as fout:
        for triple in train_triples:
            fout.write(triple + '\n')

    dev_triples = []
    with open(os.path.join(data_dir, 'dev.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            dev_triples.append('\t'.join([elems[0], elems[2], elems[1]]))

    with open(os.path.join(data_dir, 'dev.triples'), 'w', encoding='utf-8') as fout:
        for triple in dev_triples:
            fout.write(triple + '\n')

    test_triples = []
    with open(os.path.join(data_dir, 'test.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            test_triples.append('\t'.join([elems[0], elems[2], elems[1]]))

    with open(os.path.join(data_dir, 'test.triples'), 'w', encoding='utf-8') as fout:
        for triple in test_triples:
            fout.write(triple + '\n')


def prepross(data_dir):
    train_triples = []
    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            train_triples.append([elems[0], elems[2], elems[1]])

    with open(os.path.join(data_dir, 'train.triples'), 'w', encoding='utf-8') as fout:
        for triple in train_triples:
            fout.write('\t'.join(triple) + '\n')


def statistic_degree(data_dir):
    entity2degree = {}

    entities = []
    with open(os.path.join(data_dir, 'entities.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split()
            entities.append('_'.join(elems))
    print('entity number: ', len(entities))
    for entity in entities:
        entity2degree[entity] = 0.

    with open(os.path.join(data_dir, 'train.triples'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            entity2degree[elems[0]] += 1
            entity2degree[elems[1]] += 1

    sorted_entities = sorted(entity2degree, key=lambda x: -entity2degree[x])
    print('entity number: ', len(sorted_entities))
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w', encoding='utf-8') as fout:
        fout.write('DUMMY_ENTITY\t0\n')
        fout.write('NO_OP_ENTITY\t1\n')
        for ent in sorted_entities:
            fout.write('{}\t{}\n'.format(ent, int(entity2degree[ent])))


def build_raw_pgrk(data_dir):
    entities, degrees = [], []
    with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin.readlines()):
            if idx < 2: continue
            entity, degree = line.strip().split('\t')
            entities.append(entity), degrees.append(int(degree))

    degrees = np.array(degrees)
    degrees = degrees / np.sum(degrees)
    with open(os.path.join(data_dir, 'raw.pgrk'), 'w', encoding='utf-8') as fout:
        for entity, degree in zip(entities, degrees):
            fout.write('{}\t:{}\n'.format(entity, degree))


if __name__ == '__main__':
    data_dir = os.path.join('.', 'FB15k-237_20')
    # data_dir = os.path.join('.', 'FB15k-237_40')
    # data_dir = os.path.join('.', 'FB15K-237_30')
    # data_dir = os.path.join('.', 'FB15K-237_15-v2')
    # data_dir = os.path.join('.', 'FB15K-237_20-v2')
    # data_dir = os.path.join('.', 'inferwiki-16k')
    # data_dir = os.path.join('.', 'inferwiki-16k-gen-6-0-10000-0.001-5-0.6-40-50-500-0.05-1-sparse-long-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.6-0.5-30-50-800-0.05-1-sparse-long-0.4-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.6-0.3-30-50-500-0.05-1-sparse-long-0.5-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.6-0.5-30-50-800-0.05-1-sparse-long-0.2-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.6-0.5-30-50-800-0.05-1-sparse-long-0.3-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.6-0.5-30-50-800-0.05-1-sparse-long-0.5-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.3-30-50-500-0.2-0.1-0.5-10-20000-2-0.2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-24')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-26')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-86')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-20')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-0')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-68')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-for-specific_task-80')
    # data_dir = os.path.join('.', 'NELL-995-gen-6-0-20000-0.01-10-0.01-30-50-500-0.1-1-sparse-long-1')
    # data_dir = os.path.join('.', 'NELL-995-gen-6-0-20000-0.01-10-0.01-30-50-500-0.1-1-sparse-long-2')
    # data_dir = os.path.join('.', 'FB15K-237-6-0-10000-0.1-5-0.8-0.1-0.5-sparse-long_for-specific-task-all')
    # data_dir = os.path.join('.', 'NELL-995-gen-6-0-20000-0.01-10-0.8-0.01-0.05-sparse-long_for-specific-task-all')
    # transfer_hrt_to_htr(data_dir)
    build_raw_pgrk(data_dir)

    # data_dir = os.path.join('.', 'inferwiki-16k')
    # data_dir = os.path.join('.', 'inferwiki-16k-long')
    # build_raw_pgrk(data_dir)
