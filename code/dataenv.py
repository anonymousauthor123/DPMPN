import time
from collections import defaultdict
from functools import partial

import numpy as np
import tensorflow as tf

from utils import get_segment_ids, get_unique, groupby_2cols_nlargest, groupby_1cols_nlargest, groupby_1cols_merge


class Graph(object):
    def __init__(self, graph_triples, n_ents, n_rels, reversed_rel_dct):
        self.reversed_rel_dct = reversed_rel_dct

        full_edges = np.array(graph_triples.tolist(), dtype='int32').view('<i4,<i4,<i4')
        full_edges = np.sort(full_edges, axis=0, order=['f0', 'f1', 'f2']).view('<i4')
        # `full_edges`: use all train triples
        # full_edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending and consecutive `id`s
        self.full_edges = np.concatenate([np.expand_dims(np.arange(len(full_edges), dtype='int32'), 1),
                                          full_edges], axis=1)
        self.n_full_edges = len(self.full_edges)

        self.n_entities = n_ents
        self.selfloop = n_rels
        self.n_relations = n_rels + 1

        # `edges`: for current train batch
        # edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending but not consecutive `id`s
        self.edges = None
        self.n_edges = 0

        # `memorized_nodes`: for current train batch
        self.memorized_nodes = None  # (np.array) (eg_idx, v) sorted by ed_idx, v

    def make_temp_edges(self, batch, remove_all_head_tail_edges=True):
        """ batch: (np.array) (head, tail, rel)
        """
        if remove_all_head_tail_edges:
            batch_set = set([(h, t) for h, t, r in batch])
            edges_idx = [i for i, (eid, h, t, r) in enumerate(self.full_edges)
                         if (h, t) not in batch_set and (t, h) not in batch_set]

        else:
            batch_set = set([(h, t, r) for h, t, r in batch])
            if self.reversed_rel_dct is None:
                edges_idx = [i for i, (eid, h, t, r) in enumerate(self.full_edges)
                             if (h, t, r) not in batch_set]
            else:
                edges_idx = [i for i, (eid, h, t, r) in enumerate(self.full_edges)
                             if (h, t, r) not in batch_set and (t, h, self.reversed_rel_dct.get(r, -1)) not in batch_set]
        self.edges = self.full_edges[edges_idx]
        self.n_edges = len(self.edges)

    def use_full_edges(self):
        self.edges = self.full_edges
        self.n_edges = len(self.edges)

    def get_candidate_edges(self, attended_nodes=None, tc=None):
        """ attended_nodes:
            (1) None: use all graph edges with batch_size=1
            (2) (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        """
        if tc is not None:
            t0 = time.time()

        if attended_nodes is None:
            candidate_edges = np.concatenate([np.zeros((self.n_edges, 1), dtype='int32'),
                                              self.edges], axis=1)  # (0, edge_id, vi, vj, rel) sorted by (0, edge_id)
        else:
            candidate_idx, new_eg_idx = groupby_1cols_merge(attended_nodes[:, 0], attended_nodes[:, 1],
                                                            self.edges[:, 1], self.edges[:, 0])
            if len(candidate_idx) == 0:
                return np.zeros((0, 5), dtype='int32')

            candidate_edges = np.concatenate([np.expand_dims(new_eg_idx, 1),
                                              self.full_edges[candidate_idx]], axis=1)  # (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)

        if tc is not None:
            tc['candi_e'] += time.time() - t0
        # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel)
        #   sorted by (eg_idx, edge_id) or (eg_idx, vi, vj, rel)
        return candidate_edges

    def get_sampled_edges(self, candidate_edges, mode=None, max_edges_per_eg=None, max_edges_per_vi=None, tc=None):
        """ candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
        """
        assert mode is not None
        if tc is not None:
            t0 = time.time()

        if len(candidate_edges) == 0:
            return np.zeros((0, 6), dtype='int32')

        logits = tf.random.uniform((len(candidate_edges),))
        if mode == 'by_eg':
            assert max_edges_per_eg is not None
            sampled_edges = candidate_edges[:, 0]  # n_candidate_edges
            sampled_idx = groupby_1cols_nlargest(sampled_edges, logits, max_edges_per_eg)  # n_sampled_edges
            sampled_edges = np.concatenate([candidate_edges[sampled_idx],
                                            np.expand_dims(sampled_idx, 1)], axis=1)  # n_sampled_edges x 6
        elif mode == 'by_vi':
            assert max_edges_per_vi is not None
            sampled_edges = candidate_edges[:, [0, 2]]  # n_candidate_edges x 2
            sampled_idx = groupby_2cols_nlargest(sampled_edges, logits, max_edges_per_vi)  # n_sampled_edges
            sampled_edges = np.concatenate([candidate_edges[sampled_idx],
                                            np.expand_dims(sampled_idx, 1)], axis=1)  # n_sampled_edges x 6
        else:
            raise ValueError('Invalid `mode`')

        if tc is not None:
            tc['sampl_e'] += time.time() - t0
        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx)
        #   sorted by (eg_idx, edge_id)
        return sampled_edges

    def get_selected_edges(self, sampled_edges, tc=None):
        """ sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        """
        if tc is not None:
            t0 = time.time()

        if len(sampled_edges) == 0:
           return np.zeros((0, 6), dtype='int32')

        idx_vi = get_segment_ids(sampled_edges[:, [0, 2]])
        _, idx_vj = np.unique(sampled_edges[:, [0, 3]], axis=0, return_inverse=True)

        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)

        selected_edges = np.concatenate([sampled_edges[:, [0, 2, 3, 4]], idx_vi, idx_vj], axis=1)

        if tc is not None:
            tc['sele_e'] += time.time() - t0
        # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj]
        #   sorted by (eg_idx, vi, vj)
        return selected_edges

    def set_init_memorized_nodes(self, heads, tc=None):
        """ heads: batch_size
        """
        if tc is not None:
            t0 = time.time()

        batch_size = heads.shape[0]
        eg_idx = np.array(np.arange(batch_size), dtype='int32')
        self.memorized_nodes = np.stack([eg_idx, heads], axis=1)

        if tc is not None:
            tc['i_memo_v'] += time.time() - t0
        # memorized_nodes: n_memorized_nodes (=batch_size) x 2, (eg_idx, v) sorted by (ed_idx, v)
        return self.memorized_nodes

    def get_topk_nodes(self, node_attention, max_nodes, tc=None):
        """ node_attention: (tf.Tensor) batch_size x n_nodes
        """
        if tc is not None:
            t0 = time.time()

        eps = 1e-20
        node_attention = node_attention.numpy()
        n_nodes = node_attention.shape[1]
        max_nodes = min(n_nodes, max_nodes)
        sorted_idx = np.argsort(-node_attention, axis=1)[:, :max_nodes]
        sorted_idx = np.sort(sorted_idx, axis=1)
        node_attention = np.take_along_axis(node_attention, sorted_idx, axis=1)  # sorted node attention
        mask = node_attention > eps
        eg_idx = np.repeat(np.expand_dims(np.arange(mask.shape[0]), 1), mask.shape[1], axis=1)[mask].astype('int32')
        vi = sorted_idx[mask].astype('int32')
        topk_nodes = np.stack([eg_idx, vi], axis=1)

        if tc is not None:
            tc['topk_v'] += time.time() - t0
        # topk_nodes: (np.array) n_topk_nodes x 2, (eg_idx, vi) sorted
        return topk_nodes

    def get_selfloop_edges(self, attended_nodes, tc=None):
        """ attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        """
        if tc is not None:
            t0 = time.time()

        eg_idx, vi = attended_nodes[:, 0], attended_nodes[:, 1]
        selfloop_edges = np.stack([eg_idx, vi, vi, np.repeat(np.array(self.selfloop, dtype='int32'), eg_idx.shape[0])],
                                  axis=1)  # (eg_idx, vi, vi, selfloop)

        if tc is not None:
            tc['sl_bt'] += time.time() - t0
        return selfloop_edges  # (eg_idx, vi, vi, selfloop)

    def get_union_edges(self, scanned_edges, selfloop_edges, tc=None):
        """ scanned_edges: (np.array) n_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
            selfloop_edges: (np.array) n_selfloop_edges x 4 (eg_idx, vi, vi, selfloop)
        """
        if tc is not None:
            t0 = time.time()

        scanned_edges = np.zeros((0, 4), dtype='int32') if len(scanned_edges) == 0 else scanned_edges[:, :4]  # (eg_idx, vi, vj, rel)
        all_edges = np.concatenate([scanned_edges, selfloop_edges], axis=0).copy()
        sorted_idx = np.squeeze(np.argsort(all_edges.view('<i4,<i4,<i4,<i4'),
                                           order=['f0', 'f1', 'f2'], axis=0), 1).astype('int32')
        aug_scanned_edges = all_edges[sorted_idx]  # sorted by (eg_idx, vi, vj)
        idx_vi = get_segment_ids(aug_scanned_edges[:, [0, 1]])
        _, idx_vj = np.unique(aug_scanned_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        aug_scanned_edges = np.concatenate([aug_scanned_edges, idx_vi, idx_vj], axis=1)

        if tc is not None:
            tc['union_e'] += time.time() - t0
        # aug_scanned_edges: n_aug_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        return aug_scanned_edges

    def add_nodes_to_memorized(self, selected_edges, inplace=False, tc=None):
        """ selected_edges: (np.array) n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        if len(selected_edges) > 0:
            selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)
            mask = np.in1d(selected_vj.view('<i4,<i4'), self.memorized_nodes.view('<i4,<i4'), assume_unique=True)
            mask = np.logical_not(mask)
            new_nodes = selected_vj[mask]  # n_new_nodes x 2

        if len(selected_edges) > 0 and len(new_nodes) > 0:
            memorized_and_new = np.concatenate([self.memorized_nodes, new_nodes], axis=0)  # n_memorized_and_new_nodes x 2
            sorted_idx = np.squeeze(np.argsort(memorized_and_new.view('<i4,<i4'),
                                               order=['f0', 'f1'], axis=0), 1).astype('int32')

            memorized_and_new = memorized_and_new[sorted_idx]
            n_memorized_and_new_nodes = len(memorized_and_new)

            new_idx = np.argsort(sorted_idx).astype('int32')
            n_memorized_nodes = self.memorized_nodes.shape[0]
            new_idx_for_memorized = np.expand_dims(new_idx[:n_memorized_nodes], 1)

            if inplace:
                self.memorized_nodes = memorized_and_new
        else:
            new_idx_for_memorized = None
            memorized_and_new = self.memorized_nodes
            n_memorized_and_new_nodes = len(memorized_and_new)

        if tc is not None:
            tc['add_scan'] += time.time() - t0
        # new_idx_for_memorized: n_memorized_nodes x 1
        # memorized_and_new: n_memorized_and_new_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        return new_idx_for_memorized, n_memorized_and_new_nodes, memorized_and_new

    def set_index_over_nodes(self, selected_edges, nodes, tc=None):
        """ selected_edges (or aug_selected_edges): n_selected_edges (or n_aug_selected_edges) x 6, sorted
            nodes: (eg_idx, v) unique and sorted
        """
        if tc is not None:
            t0 = time.time()

        if len(selected_edges) == 0:
            return np.zeros((0, 8), dtype='int32')

        selected_vi = get_unique(selected_edges[:, [0, 1]])  # n_selected_edges x 2
        selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)  # n_selected_edges x 2
        mask_vi = np.in1d(nodes.view('<i4,<i4'), selected_vi.view('<i4,<i4'), assume_unique=True)
        mask_vj = np.in1d(nodes.view('<i4,<i4'), selected_vj.view('<i4,<i4'), assume_unique=True)
        new_idx_e2vi = np.expand_dims(np.arange(mask_vi.shape[0])[mask_vi], 1).astype('int32')  # n_matched_by_idx_and_vi x 1
        new_idx_e2vj = np.expand_dims(np.arange(mask_vj.shape[0])[mask_vj], 1).astype('int32')  # n_matched_by_idx_and_vj x 1

        idx_vi = selected_edges[:, 4]
        idx_vj = selected_edges[:, 5]
        new_idx_e2vi = new_idx_e2vi[idx_vi]  # n_selected_edges x 1
        new_idx_e2vj = new_idx_e2vj[idx_vj]  # n_selected_edges x 1

        # selected_edges: n_selected_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        selected_edges = np.concatenate([selected_edges, new_idx_e2vi, new_idx_e2vj], axis=1)

        if tc is not None:
            tc['idx_v'] += time.time() - t0
        return selected_edges

    def get_seen_edges(self, seen_nodes, aug_scanned_edges, tc=None):
        """ seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique but not sorted
            aug_scanned_edges: (np.array) n_aug_scanned_edges x 8,
                (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        aug_scanned_vj = aug_scanned_edges[:, [0, 2]].copy()  # n_aug_scanned_edges x 2, (eg_idx, vj) not unique and not sorted
        mask_vj = np.in1d(aug_scanned_vj.view('<i4,<i4'), seen_nodes.view('<i4,<i4'))
        seen_edges = aug_scanned_edges[mask_vj][:, :4]  # n_seen_edges x 4, (eg_idx, vi, vj, rel) sorted by (eg_idx, vi, vj)

        idx_vi = get_segment_ids(seen_edges[:, [0, 1]])
        _, idx_vj = np.unique(seen_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        seen_edges = np.concatenate((seen_edges, idx_vi, idx_vj), axis=1)

        if tc is not None:
            tc['seen_e'] += time.time() - t0
        # seen_edges: n_seen_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        return seen_edges

    def get_vivj_edges(self, vi_nodes, vj_nodes, with_eg_idx=True):
        """ vi_nodes: n_attended_vi_nodes x 2, (eg_idx, vi) or n_attended_vi_nodes, (vi)
            vj_nodes: n_attended_vj_nodes x 2, (eg_idx, vj) or n_attended_vj_nodes, (vj)
        """
        if with_eg_idx:
            candidate_idx, new_eg_idx = groupby_1cols_merge(vi_nodes[:, 0], vi_nodes[:, 1],
                                                            self.edges[:, 1], self.edges[:, 0])
            candidate_edges = np.concatenate([np.expand_dims(new_eg_idx, 1), self.full_edges[candidate_idx]], axis=1)  # (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            candidate_vj = candidate_edges[:, [0, 3]].copy()  # n_candidate_edges x 2, (eg_idx, vj) not unique and not sorted
            mask_vj = np.in1d(candidate_vj.view('<i4,<i4'), vj_nodes.view('<i4,<i4'))
            vivj_edges = candidate_edges[mask_vj]  # n_vivj_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, vi, vj)
        else:
            candidate_idx = self.edges[:, 0][np.in1d(self.edges[:, 1], vi_nodes)]
            candidate_edges = self.full_edges[candidate_idx]  # (edge_id, vi, vj, rel) sorted by edge_id
            candidate_vj = candidate_edges[:, 2].copy()  # n_candidate_edges
            mask_vj = np.in1d(candidate_vj, vj_nodes)
            vivj_edges = candidate_edges[mask_vj]  # n_vivj_edges x 4, (edge_id, vi, vj, rel) sorted by edge_id

        return vivj_edges


class DataFeeder(object):
    def get_train_batch(self, train_data, graph, batch_size, shuffle=True, remove_all_head_tail_edges=True):
        n_train = len(train_data)
        rand_idx = np.random.permutation(n_train) if shuffle else np.arange(n_train)
        start = 0
        while start < n_train:
            end = min(start + batch_size, n_train)
            pad = max(start + batch_size - n_train, 0)
            batch = np.array([train_data[i] for i in np.concatenate([rand_idx[start:end], rand_idx[:pad]])], dtype='int32')
            graph.make_temp_edges(batch, remove_all_head_tail_edges=remove_all_head_tail_edges)
            yield batch, end - start
            start = end

    def get_eval_batch(self, eval_data, graph, batch_size, shuffle=False):
        n_eval = len(eval_data)
        rand_idx = np.random.permutation(n_eval) if shuffle else np.arange(n_eval)
        start = 0
        while start < n_eval:
            end = min(start + batch_size, n_eval)
            batch = np.array([eval_data[i] for i in rand_idx[start:end]], dtype='int32')
            graph.use_full_edges()
            yield batch, end - start
            start = end


class DataEnv(object):
    def __init__(self, dataset):
        self.data_feeder = DataFeeder()

        self.ds = dataset

        self.valid = dataset.valid
        self.test = dataset.test
        self.train = dataset.train
        self.test_candidates = dataset.test_candidates
        self.test_by_rel = dataset.test_by_rel

        self.graph = Graph(dataset.graph, dataset.n_entities, dataset.n_relations, dataset.reversed_rel_dct)

        self.filter_pool = defaultdict(set)
        for head, tail, rel in np.concatenate([self.train, self.valid, self.test], axis=0):
            self.filter_pool[(head, rel)].add(tail)

    def get_train_batcher(self, remove_all_head_tail_edges=True):
        return partial(self.data_feeder.get_train_batch, self.train, self.graph,
                       remove_all_head_tail_edges=remove_all_head_tail_edges)

    def get_valid_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.valid, self.graph)

    def get_test_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.test, self.graph)

    def get_test_relations(self):
        return self.test_by_rel.keys() if self.test_by_rel is not None else None

    def get_test_batcher_by_rel(self, rel):
        return partial(self.data_feeder.get_eval_batch, self.test_by_rel[rel], self.graph) \
            if self.test_by_rel is not None else None

    @property
    def n_train(self):
        return len(self.train)

    @property
    def n_valid(self):
        return len(self.valid)

    @property
    def n_test(self):
        return len(self.test)
