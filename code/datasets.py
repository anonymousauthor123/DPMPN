import os
from collections import defaultdict

import numpy as np


class Dataset(object):
    def __init__(self, train_path, valid_path, test_path, graph_path=None, test_candidates_path=None,
                 do_reverse=False, do_reverse_on_graph=False, get_reverse=None, has_reverse=False,
                 test_paths=None, test_candidates_paths=None):
        train = self._load_triple_file(train_path)
        valid = self._load_triple_file(valid_path)
        test = self._load_triple_file(test_path)
        graph = self._load_triple_file(graph_path) if graph_path is not None else train

        if test_candidates_path is not None:
            test_candidates = self._load_test_candidates_file(test_candidates_path)
        elif test_candidates_paths is not None:
            test_candidates = {os.path.basename(path).split('_')[-1]: self._load_test_candidates_file(path)
                               for path in test_candidates_paths}
        else:
            test_candidates = None

        if test_paths is not None:
            test_by_rel = {os.path.basename(path).split('_')[-1]: self._load_triple_file(path)
                           for path in test_paths}
        else:
            test_by_rel = None


        if do_reverse:
            train = self._add_reverse_triples(train)
            valid = self._add_reverse_triples(valid)
            test = self._add_reverse_triples(test)
            graph = self._add_reverse_triples(graph)
        elif do_reverse_on_graph:
            graph = self._add_reverse_triples(graph)

        self.entity2id, self.id2entity, self.relation2id, self.id2relation = \
            self._make_dict(graph + train + valid + test)
        self.n_entities = len(self.entity2id)
        self.n_relations = len(self.relation2id)

        self.reversed_rel_dct = None
        if do_reverse or do_reverse_on_graph or has_reverse:
            self.reversed_rel_dct = self._get_reversed_relation_dict(self.relation2id, get_reverse=get_reverse)

        self.train = self._convert_to_id(train)
        self.valid = self._convert_to_id(valid)
        self.test = self._convert_to_id(test)
        self.graph = self._convert_to_id(graph)

        if test_candidates_path is not None:
            self.test_candidates = self._convert_to_id_v2(test_candidates)
        elif test_candidates_paths is not None:
            self.test_candidates = {rel: self._convert_to_id_v2(dct) for rel, dct in test_candidates.items()}
        else:
            self.test_candidates = None

        if test_paths is not None:
            self.test_by_rel = {rel: self._convert_to_id(triples) for rel, triples in test_by_rel.items()}
        else:
            self.test_by_rel = None

    def _load_triple_file(self, filepath):
        triples = []
        with open(filepath) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((h, t, r))
        return triples

    def _load_test_candidates_file(self, filepath):
        test_candidates = defaultdict(dict)
        with open(filepath) as fin:
            for line in fin:
                pair, ans = line.strip().split(': ')
                h, t = pair.split(',')
                h = h.replace('thing$', '')
                t = t.replace('thing$', '')
                test_candidates[h][t] = ans
        return test_candidates

    def _make_dict(self, triples):
        ent2id, rel2id = {}, {}
        id2ent, id2rel = {}, {}
        for h, t, r in triples:
            ent2id.setdefault(h, len(ent2id))
            id2ent[ent2id[h]] = h
            ent2id.setdefault(t, len(ent2id))
            id2ent[ent2id[t]] = t
            rel2id.setdefault(r, len(rel2id))
            id2rel[rel2id[r]] = r
        return ent2id, id2ent, rel2id, id2rel

    def _add_reverse_triples(self, triples):
        return triples + [(t, h, '_' + r) for h, t, r in triples]

    def _get_reversed_relation_dict(self, relation2id, get_reverse=None):
        if get_reverse is None:
            get_reverse = lambda r: '_' + r if r[0] != '_' else r[1:]
        return {id: relation2id[get_reverse(rel)] for rel, id in relation2id.items()
                if get_reverse(rel) in relation2id}

    def _convert_to_id(self, triples):
        return np.array([(self.entity2id[h], self.entity2id[t], self.relation2id[r])
                         for h, t, r in triples], dtype='int32')

    def _convert_to_id_v2(self, answers):
        return {self.entity2id[h]: {self.entity2id[t]: ans for t, ans in t_dct.items()}
                for h, t_dct in answers.items()}


class FB237(Dataset):
    path = '../data/KBC/FB237'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'FB237'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(FB237, self).__init__(train_path, valid_path, test_path,
                                    do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class FB237_v2(Dataset):
    path = '../data/MWalk/FB15K-237'

    def __init__(self):
        self.name = 'FB237_v2'
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        graph_path = os.path.join(self.path, 'graph.txt')
        super(FB237_v2, self).__init__(train_path, valid_path, test_path,
                                       graph_path=graph_path,
                                       has_reverse=True)


class FB15K(Dataset):
    path = '../data/KBC/FB15K'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'FB15K'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(FB15K, self).__init__(train_path, valid_path, test_path,
                                    do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class WN(Dataset):
    path = '../data/KBC/WN'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'WN'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(WN, self).__init__(train_path, valid_path, test_path,
                                 do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class WN18RR(Dataset):
    path = '../data/KBC/WN18RR'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'WN18RR'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(WN18RR, self).__init__(train_path, valid_path, test_path,
                                     do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class WN18RR_v2(Dataset):
    path = '../data/MWalk/WN18RR'

    def __init__(self):
        self.name = 'WN18RR_v2'
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        graph_path = os.path.join(self.path, 'graph.txt')
        super(WN18RR_v2, self).__init__(train_path, valid_path, test_path,
                                        graph_path=graph_path,
                                        has_reverse=True)


class YAGO310(Dataset):
    path = '../data/KBC/YAGO3-10'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'YAGO310'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(YAGO310, self).__init__(train_path, valid_path, test_path,
                                      do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class NELL995(Dataset):
    path = '../data/MWalk'

    query_relations = ['athleteplaysforteam',
                       'athleteplaysinleague',
                       'athletehomestadium',
                       'athleteplayssport',
                       'teamplayssport',
                       'organizationheadquarteredincity',
                       'worksfor',
                       'personborninlocation',
                       'personleadsorganization',
                       'organizationhiredperson',
                       'agentbelongstoorganization',
                       'teamplaysinleague']

    def __init__(self, query_relation=None):
        self.name = 'NELL995' if query_relation is None else query_relation
        if query_relation is None:
            self.path = os.path.join(NELL995.path, 'nell')
            test_candidates_path = None
        else:
            assert query_relation in NELL995.query_relations
            self.path = os.path.join(NELL995.path, query_relation)
            test_candidates_path = os.path.join(self.path, 'sort_test.pairs')
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        graph_path = os.path.join(self.path, 'graph.txt')

        super(NELL995, self).__init__(train_path, valid_path, test_path,
                                      graph_path=graph_path,
                                      test_candidates_path=test_candidates_path,
                                      has_reverse=True,
                                      get_reverse=lambda r: r + '_inv' if r[-4:] != '_inv' else r[:-4])

    @classmethod
    def datasets(cls, include_whole=False):
        for rel in cls.query_relations:
            yield cls(query_relation=rel)
        if include_whole:
            yield cls()
