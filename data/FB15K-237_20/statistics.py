import os
import numpy as np
from collections import defaultdict as ddict


def statistics_connected_relation():
    # ent2rel = ddict(set)
    ent2rel = ddict(list)
    with open('train.triples', 'r', encoding='utf-8') as fin:
        for line in fin:
            elems = line.strip().split('\t')
            ent2rel[elems[0]].append(elems[2])

    ent2rel_n = [len(ent2rel[ent]) for ent in ent2rel]
    print('max relation number: ', np.max(ent2rel_n))
    print('min relation number: ', np.min(ent2rel_n))


if __name__ == '__main__':
    statistics_connected_relation()