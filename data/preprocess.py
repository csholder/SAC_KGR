import os
import random


def transfer_sor2sro(data_dir):
    triples = []
    with open(os.path.join(data_dir, 'raw.kb'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            triples.append([elems[0], elems[2], elems[1]])

    with open(os.path.join(data_dir, 'raw.kb'), 'w', encoding='utf-8') as fout:
        for triple in triples:
            fout.write('\t'.join(triple) + '\n')


def sparser(data_dir, output_dir):
    train_triples, dev_triples, test_triples = [], [], []
    entities, relations = set(), set()
    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as fin:
        train_triples = fin.readlines()

    train_triples = random.sample(train_triples, k=int(len(train_triples) * 0.1))
    for line in train_triples:
        elems = line.strip().split('\t')
        entities.add(elems[0])
        entities.add(elems[2])
        relations.add(elems[1])

    with open(os.path.join(data_dir, 'dev.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            if elems[0] in entities and elems[2] in entities and elems[1] in relations:
                dev_triples.append(line)
                entities.add(elems[0])
                entities.add(elems[2])
                relations.add(elems[1])

    with open(os.path.join(data_dir, 'test.txt'), 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            if elems[0] in entities and elems[2] in entities and elems[1] in relations:
                test_triples.append(line)
                entities.add(elems[0])
                entities.add(elems[2])
                relations.add(elems[1])

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as fout:
        for line in train_triples:
            fout.write(line)

    with open(os.path.join(output_dir, 'dev.txt'), 'w', encoding='utf-8') as fout:
        for line in dev_triples:
            fout.write(line)

    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as fout:
        for line in test_triples:
            fout.write(line)


if __name__ == '__main__':
    data_dir = 'FB15K-237_20'
    # transfer_sor2sro(data_dir)

    output_dir = 'FB15K-237_20_0.1'
    sparser(data_dir, output_dir)
