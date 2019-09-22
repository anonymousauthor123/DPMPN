import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

from utils import get, entropy, topk_occupy


class F(keras.layers.Layer):
    def __init__(self, n_dims, n_layers, name=None):
        super(F, self).__init__(name=name)
        self.n_dims = n_dims
        self.n_layers = n_layers

    def build(self, input_shape):
        if self.n_layers == 1:
            self.dense_1 = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='dense_1')
        elif self.n_layers == 2:
            self.dense_1 = keras.layers.Dense(self.n_dims, activation=tf.nn.leaky_relu, name='dense_1')
            self.dense_2 = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='dense_2')
        else:
            raise ValueError('Invalid `n_layers`')

    def call(self, inputs, training=None):
        """ inputs[i]: bs x ... x n_dims
        """
        x = tf.concat(inputs, axis=-1)
        if self.n_layers == 1:
            return self.dense_1(x)
        elif self.n_layers == 2:
            return self.dense_2(self.dense_1(x))


class G(keras.layers.Layer):
    def __init__(self, n_dims, name=None):
        super(G, self).__init__(name=name)
        self.n_dims = n_dims

    def build(self, input_shape):
        self.left_dense = keras.layers.Dense(self.n_dims, activation=tf.nn.leaky_relu, name='left_dense')
        self.right_dense = keras.layers.Dense(self.n_dims, activation=tf.nn.leaky_relu, name='right_dense')
        self.center_dense = keras.layers.Dense(self.n_dims, activation=None, name='center_dense')

    def call(self, inputs, training=None):
        """ inputs: (left, right)
                left[i]: bs x ... x n_dims
                right[i]: bs x ... x n_dims
        """
        left, right = inputs
        left_x = tf.concat(left, axis=-1)
        right_x = tf.concat(right, axis=-1)
        return tf.reduce_sum(self.left_dense(left_x) * self.center_dense(self.right_dense(right_x)), axis=-1)


def update_op(inputs, update):
    out = inputs + update
    return out


def node2edge_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): batch_size x n_nodes x n_dims
        selected_edges: n_selected_edges x 6 (or 8) ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
    """
    hidden = inputs
    batch_size = tf.shape(inputs)[0]
    n_selected_edges = len(selected_edges)
    idx = tf.cond(tf.equal(batch_size, 1), lambda: tf.zeros((n_selected_edges,), dtype='int32'), lambda: selected_edges[:, 0])
    result = []
    if return_vi:
        idx_and_vi = tf.stack([idx, selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        hidden_vi = tf.gather_nd(hidden, idx_and_vi)  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        idx_and_vj = tf.stack([idx, selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        hidden_vj = tf.gather_nd(hidden, idx_and_vj)  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def node2edge_v2_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): n_selected_nodes x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    assert return_vi or return_vj
    hidden = inputs
    result = []
    if return_vi:
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        hidden_vi = tf.gather(hidden, new_idx_e2vi)  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        hidden_vj = tf.gather(hidden, new_idx_e2vj)  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def aggregate_op(inputs, selected_edges, output_shape, at='vj', aggr_op_name='mean_v3'):
    """ inputs (edge_vec): n_seleted_edges x n_dims
        selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
        output_shape: (batch_size=1, n_nodes, n_dims)
    """
    assert selected_edges is not None
    assert output_shape is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        aggr_op = tf.math.segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.segment_sum if aggr_op_name == 'sum' else \
            tf.math.segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vi+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vi+1) x n_dims
        idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        idx_and_vi = tf.cast(tf.math.segment_max(idx_and_vi, idx_vi), tf.int32)  # (max_id_vi+1) x 2
        edge_vec_aggr = tf.scatter_nd(idx_and_vi, edge_vec_aggr, output_shape)  # batch_size x n_nodes x n_dims
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        max_idx_vj = tf.reduce_max(idx_vj)
        aggr_op = tf.math.unsorted_segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.unsorted_segment_sum if aggr_op_name == 'sum' else \
            tf.math.unsorted_segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vj+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vj+1) x n_dims
        idx_and_vj = tf.stack([selected_edges[:, 0], selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        idx_and_vj = tf.cast(tf.math.unsorted_segment_max(idx_and_vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1) x 2
        edge_vec_aggr = tf.scatter_nd(idx_and_vj, edge_vec_aggr, output_shape)  # batch_size x n_nodes x n_dims
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_aggr


def aggregate_v2_op(inputs, selected_edges, output_shape, at='vj', aggr_op_name='mean_v3'):
    """ inputs (edge_vec): n_seleted_edges x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
        output_shape: (n_visited_nodes, n_dims)
    """
    assert selected_edges is not None
    assert output_shape is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        aggr_op = tf.math.segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.segment_sum if aggr_op_name == 'sum' else \
            tf.math.segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vi+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vi+1) x n_dims
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        reduced_idx_e2vi = tf.cast(tf.math.segment_max(new_idx_e2vi, idx_vi), tf.int32)  # (max_id_vi+1)
        reduced_idx_e2vi = tf.expand_dims(reduced_idx_e2vi, 1)  # (max_id_vi+1) x 1
        edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vi, edge_vec_aggr, output_shape)  # n_visited_nodes x n_dims
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        max_idx_vj = tf.reduce_max(idx_vj)
        aggr_op = tf.math.unsorted_segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.unsorted_segment_sum if aggr_op_name == 'sum' else \
            tf.math.unsorted_segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vj+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vj+1) x n_dims
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        reduced_idx_e2vj = tf.cast(tf.math.unsorted_segment_max(new_idx_e2vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1)
        reduced_idx_e2vj = tf.expand_dims(reduced_idx_e2vj, 1)  # (max_idx_vj+1) x 1
        edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vj, edge_vec_aggr, output_shape)  # n_visited_nodes x n_dims
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_aggr


def sparse_softmax_op(logits, segment_ids, sort=True):
    if sort:
        logits_max = tf.math.segment_max(logits, segment_ids)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_diff = logits - logits_max
        logits_exp = tf.math.exp(logits_diff)
        logits_expsum = tf.math.segment_sum(logits_exp, segment_ids)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    else:
        num_segments = tf.reduce_max(segment_ids) + 1
        logits_max = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_diff = logits - logits_max
        logits_exp = tf.math.exp(logits_diff)
        logits_expsum = tf.math.unsorted_segment_sum(logits_exp, segment_ids, num_segments)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    return logits_norm


def neighbor_softmax_op(inputs, selected_edges, at='vi'):
    """ inputs (edge_vec): n_seleted_edges x ...
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        edge_vec_norm = sparse_softmax_op(edge_vec, idx_vi)  # n_selected_edges x ...
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        edge_vec_norm = sparse_softmax_op(edge_vec, idx_vj, sort=False)  # n_selected_edges x ...
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_norm


class SharedEmbedding(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims):
        super(SharedEmbedding, self).__init__(name='shared_emb')
        self.n_dims = n_dims
        self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims, name='entities')  # n_nodes x n_dims
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims, name='relations')  # n_rels x n_dims

    def call(self, inputs, target=None, training=None):
        assert target is not None
        if target == 'entity':
            return self.entity_embedding(inputs)
        elif target == 'relation':
            return self.relation_embedding(inputs)
        else:
            raise ValueError('Invalid `target`')

    def get_query_context(self, heads, rels):
        with tf.name_scope(self.name):
            head_emb = self.entity_embedding(heads)  # batch_size x n_dims
            rel_emb = self.relation_embedding(rels)  # batch_size x n_dims
        return head_emb, rel_emb


class UnconsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_dims, n_layers, aggr_op_name):
        super(UnconsciousnessFlow, self).__init__(name='uncon_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.aggr_op_name = aggr_op_name

        # fn(hidden_vi, rel_emb, hidden_vj)
        self.message_fn = F(self.n_dims, self.n_layers, name='message_fn')

        # fn(message_aggr, hidden, ent_emb)
        self.hidden_fn = F(self.n_dims, self.n_layers, name='hidden_fn')

    def call(self, inputs, selected_edges=None, shared_embedding=None, training=None, tc=None):
        """ inputs (hidden): 1 x n_nodes x n_dims
            selected_edges: n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)

            Here: batch_size = 1
        """
        assert selected_edges is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        # compute unconscious messages
        hidden = inputs
        hidden_vi, hidden_vj = node2edge_op(hidden, selected_edges)  # n_selected_edges x n_dims
        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_selected_edges x n_dims
        message = self.message_fn((hidden_vi, rel_emb, hidden_vj))  # n_selected_edges x n_dims

        # aggregate unconscious messages
        message_aggr = aggregate_op(message, selected_edges, (1, self.n_nodes, self.n_dims),
                                    aggr_op_name=self.aggr_op_name)  # 1 x n_nodes

        # update unconscious states
        ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
        ent_emb = shared_embedding(ent_idx, target='entity')  # 1 x n_nodes x n_dims
        update = self.hidden_fn((message_aggr, hidden, ent_emb))  # 1 x n_nodes x n_dims
        hidden = update_op(hidden, update)  # 1 x n_nodes x n_dims

        if tc is not None:
            tc['u.call'] += time.time() - t0
        return hidden  # 1 x n_nodes x n_dims

    def get_init_hidden(self, shared_embedding):
        with tf.name_scope(self.name):
            ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
            ent_emb = shared_embedding(ent_idx, target='entity')  # 1 x n_nodes x n_dims
            hidden = ent_emb
        return hidden


class ConsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_dims, n_layers, aggr_op_name):
        super(ConsciousnessFlow, self).__init__(name='con_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.aggr_op_name = aggr_op_name

        # fn(hidden_vi, rel_emb, hidden_vj, query_head_emb, query_rel_emb)
        self.message_fn = F(self.n_dims, self.n_layers, name='message_fn')

        # fn(message_aggr, hidden, hidden_uncon, query_head_emb, query_rel_emb)
        self.hidden_fn = F(self.n_dims, self.n_layers, name='hidden_fn')

        self.intervention_fn = keras.layers.Dense(self.n_dims, activation=None, use_bias=False, name='intervention_fn')

    def call(self, inputs, seen_edges=None, memorized_nodes=None, node_attention=None, hidden_uncon=None,
             query_head_emb=None, query_rel_emb=None, shared_embedding=None, training=None, tc=None):
        """ inputs (hidden): n_memorized_nodes x n_dims
            seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by (idx, vi, vj)
                (1) including selfloop edges and backtrace edges
                (2) batch_size >= 1
            memorized_nodes: n_memorized_nodes x 2, (eg_idx, v)
            node_attention: batch_size x n_nodes
            hidden_uncon: 1 x n_nodes x n_dims
            query_head_emb: batch_size x n_dims
            query_rel_emb: batch_size x n_dims
        """
        assert seen_edges is not None
        assert node_attention is not None
        assert hidden_uncon is not None
        assert query_head_emb is not None
        assert query_rel_emb is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        hidden = inputs

        # compute conscious messages
        hidden_vi, hidden_vj = node2edge_v2_op(hidden, seen_edges)  # n_seen_edges x n_dims
        rel_idx = seen_edges[:, 3]  # n_seen_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_seen_edges x n_dims
        eg_idx = seen_edges[:, 0]  # n_seen_edges
        query_head_vec = tf.gather(query_head_emb, eg_idx)  # n_seen_edges x n_dims
        query_rel_vec = tf.gather(query_rel_emb, eg_idx)  # n_seen_edges x n_dims

        message = self.message_fn((hidden_vi, rel_emb, hidden_vj, query_head_vec, query_rel_vec))  # n_seen_edges x n_dims

        # aggregate conscious messages
        n_memorized_nodes = tf.shape(hidden)[0]
        message_aggr = aggregate_v2_op(message, seen_edges, (n_memorized_nodes, self.n_dims),
                                       aggr_op_name=self.aggr_op_name)  # n_memorized_nodes x n_dims

        # get unconscious states
        eg_idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
        hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims
        hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims
        query_head_vec = tf.gather(query_head_emb, eg_idx)  # n_memorized_nodes x n_dims
        query_rel_vec = tf.gather(query_rel_emb, eg_idx)  # n_memorized_nodes x n_dims

        # attend unconscious states
        idx_and_v = tf.stack([eg_idx, v], axis=1)  # n_memorized_nodes x 2
        node_attention = tf.gather_nd(node_attention, idx_and_v)  # n_memorized_nodes
        hidden_uncon = tf.expand_dims(node_attention, 1) * hidden_uncon  # n_memorized_nodes x n_dims
        hidden_uncon = self.intervention_fn(hidden_uncon)  # n_memorized_nodes x n_dims

        # update conscious state
        update = self.hidden_fn((message_aggr, hidden, hidden_uncon, query_head_vec, query_rel_vec))  # n_memorized_nodes x n_dims
        hidden = update_op(hidden, update)  # n_memorized_nodes x n_dims

        if tc is not None:
            tc['c.call'] += time.time() - t0
        return hidden  # n_memorized_nodes x n_dims

    def get_init_hidden(self, hidden_uncon, memorized_nodes):
        """ hidden_uncon: 1 x n_nodes x n_dims
            memorized_nodes: n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        """
        with tf.name_scope(self.name):
            idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
            hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims
            hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims
            hidden_init = hidden_uncon
        return hidden_init  # n_memorized_nodes x n_dims


class AttentionFlow(keras.Model):
    def __init__(self, n_entities, n_dims_sm):
        super(AttentionFlow, self).__init__(name='att_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims_sm

        # fn((hidden_con_vi, rel_emb, query_head_emb, query_rel_emb),
        #    (hidden_con_vj, rel_emb, query_head_emb, query_rel_emb))
        self.transition_fn_1 = G(self.n_dims, name='transition_fn')

        # fn((hidden_con_vi, rel_emb, query_head_emb, query_rel_emb),
        #    (hidden_uncon_vj, rel_emb, query_head_emb, query_rel_emb))
        self.transition_fn_2 = G(self.n_dims, name='transition_fn')

        self.proj = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj')

    def call(self, inputs, scanned_edges=None, hidden_uncon=None, hidden_con=None, shared_embedding=None,
             new_idx_for_memorized=None, n_memorized_and_scanned_nodes=None, query_head_emb=None, query_rel_emb=None,
             training=None, tc=None):
        """ inputs (node_attention): batch_size x n_nodes
            scanned_edges (aug_scanned_edges): n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
              (1) including selfloop edges
              (2) batch_size >= 1
            hidden_uncon: 1 x n_nodes x n_dims
            hidden_con: n_memorized_nodes x n_dims
            query_head_emb: batch_size x n_dims
            query_rel_emb: batch_size x n_dims
        """
        assert scanned_edges is not None
        assert hidden_con is not None
        assert hidden_uncon is not None
        assert query_head_emb is not None
        assert query_rel_emb is not None
        assert n_memorized_and_scanned_nodes is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        hidden_con = self.proj(hidden_con)  # n_memorized_nodes x n_dims_sm
        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_scanned_nodes, self.n_dims)))  # n_memorized_and_scanned_nodes x n_dims_sm
        hidden_uncon = self.proj(hidden_uncon)  # 1 x n_nodes x n_dims_sm
        query_head_vec = self.proj(query_head_emb)  # batch_size x n_dims_sm
        query_rel_vec = self.proj(query_rel_emb)  # batch_size x n_dims_sm

        # compute transition
        hidden_con_vi, hidden_con_vj = node2edge_v2_op(hidden_con, scanned_edges)  # n_aug_scanned_edges x n_dims_sm
        hidden_uncon_vj, = node2edge_op(hidden_uncon, scanned_edges, return_vi=False)  # n_aug_scanned_edges x n_dims_sm

        rel_idx = scanned_edges[:, 3]  # n_aug_scanned_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_aug_scanned_edges x n_dims
        rel_emb = self.proj(rel_emb)  # n_aug_scanned_edges x n_dims_sm

        eg_idx = scanned_edges[:, 0]  # n_aug_scanned_edges
        q_head_vec = tf.gather(query_head_vec, eg_idx)  # n_seen_edges x n_dims
        q_rel_vec = tf.gather(query_rel_vec, eg_idx)  # n_seen_edges x n_dims

        transition_logits = self.transition_fn_1(((hidden_con_vi, rel_emb, q_head_vec, q_rel_vec),
                                                  (hidden_con_vj, rel_emb, q_head_vec, q_rel_vec)))
        transition_logits += self.transition_fn_2(((hidden_con_vi, rel_emb, q_head_vec, q_rel_vec),
                                                   (hidden_uncon_vj, rel_emb, q_head_vec, q_rel_vec)))  # n_aug_scanned_edges
        transition = neighbor_softmax_op(transition_logits, scanned_edges)  # n_aug_scanned_edges

        # compute transition attention
        node_attention = inputs  # batch_size x n_nodes
        idx_and_vi = tf.stack([scanned_edges[:, 0], scanned_edges[:, 1]], axis=1)  # n_aug_scanned_edges x 2
        gathered_node_attention = tf.gather_nd(node_attention, idx_and_vi)  # n_aug_scanned_edges
        trans_attention = gathered_node_attention * transition  # n_aug_scanned_edges

        # compute new node attention
        batch_size = tf.shape(inputs)[0]
        new_node_attention = aggregate_op(trans_attention, scanned_edges, (batch_size, self.n_nodes),
                                          aggr_op_name='sum')  # batch_size x n_nodes

        new_node_attention_sum = tf.reduce_sum(new_node_attention, axis=1, keepdims=True)  # batch_size x 1
        new_node_attention = new_node_attention / new_node_attention_sum  # batch_size x n_nodes

        if tc is not None:
            tc['a.call'] += time.time() - t0
        # new_node_attention: batch_size x n_nodes
        return new_node_attention

    def get_init_node_attention(self, heads):
        with tf.name_scope(self.name):
            node_attention = tf.one_hot(heads, self.n_nodes)  # batch_size x n_nodes
        return node_attention


class Model(object):
    def __init__(self, graph, hparams):
        self.graph = graph
        self.hparams = hparams

        self.shared_embedding = SharedEmbedding(graph.n_entities, graph.n_relations, hparams.n_dims)

        self.uncon_flow = UnconsciousnessFlow(graph.n_entities, hparams.n_dims, hparams.n_layers, hparams.aggregate_op)

        self.con_flow = ConsciousnessFlow(graph.n_entities, hparams.n_dims, hparams.n_layers, hparams.aggregate_op)

        self.att_flow = AttentionFlow(graph.n_entities, hparams.n_dims_sm)

        self.heads, self.rels = None, None

        # for visualization
        self.attended_nodes = None

        # for analysis
        self.entropy_along_steps = None
        self.top1_occupy_along_steps = None
        self.top3_occupy_along_steps = None
        self.top5_occupy_along_steps = None

    def set_init(self, heads, rels, tails, batch_i, epoch):
        """ heads: batch_size
            rels: batch_size
        """
        self.heads = heads
        self.rels = rels
        self.tails = tails
        self.batch_i = batch_i
        self.epoch = epoch

    def initialize(self, training=True, output_attention=False, analyze_attention=False, tc=None):
        query_head_emb, query_rel_emb = self.shared_embedding.get_query_context(self.heads, self.rels)  # batch_size x n_dims

        ''' initialize unconsciousness flow'''
        hidden_uncon = self.uncon_flow.get_init_hidden(self.shared_embedding)  # 1 x n_nodes x n_dims

        ''' run unconsciousness flow before running consciousness flow '''
        if self.hparams.uncon_steps is not None:
            for _ in range(self.hparams.uncon_steps):
                # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
                candidate_edges = self.graph.get_candidate_edges(tc=get(tc, 'graph'))

                # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
                max_edges_per_eg = self.hparams.max_edges_per_example if training else self.hparams.test_max_edges_per_example
                sampled_edges = self.graph.get_sampled_edges(candidate_edges, mode='by_eg',
                                                             max_edges_per_eg=max_edges_per_eg,
                                                             tc=get(tc, 'graph'))

                # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
                selected_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

                hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges,
                                               shared_embedding=self.shared_embedding,
                                               tc=get(tc, 'model'))  # 1 x n_nodes x n_dims

        ''' initialize attention flow '''
        node_attention = self.att_flow.get_init_node_attention(self.heads)  # batch_size x n_nodes

        ''' initialize consciousness flow '''
        memorized_v = self.graph.set_init_memorized_nodes(self.heads)  # n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        hidden_con = self.con_flow.get_init_hidden(hidden_uncon, memorized_v)  # n_memorized_nodes x n_dims

        if output_attention and not training:
            self.attended_nodes = []

        if analyze_attention and not training:
            self.entropy_along_steps = [tf.reduce_mean(entropy(node_attention))]
            self.top1_occupy_along_steps = [tf.reduce_mean(topk_occupy(node_attention, 1))]
            self.top3_occupy_along_steps = [tf.reduce_mean(topk_occupy(node_attention, 3))]
            self.top5_occupy_along_steps = [tf.reduce_mean(topk_occupy(node_attention, 5))]

        # hidden_uncon: 1 x n_nodes x n_dims
        # hidden_con: n_memorized_nodes x n_dims
        # node_attention: batch_size x n_nodes
        # query_head_emb, query_rel_emb: batch_size x n_dims
        return hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb

    def flow(self, hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb,
             training=True, output_attention=False, analyze_attention=False, tc=None):
        """ hidden_uncon: 1 x n_nodes x n_dims
            hidden_con: n_memorized_nodes x n_dims
            node_attention: batch_size x n_nodes
        """
        ''' get scanned edges '''
        # attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        max_attended_nodes = self.hparams.max_attended_nodes if training else self.hparams.test_max_attended_nodes
        attended_nodes = self.graph.get_topk_nodes(node_attention, max_attended_nodes, tc=get(tc, 'graph'))  # n_attended_nodes x 2

        if output_attention and not training:
            self.attended_nodes.append((attended_nodes, tf.gather_nd(node_attention, attended_nodes)))

        # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
        candidate_edges = self.graph.get_candidate_edges(attended_nodes=attended_nodes,
                                                         tc=get(tc, 'graph'))  # n_candidate_edges x 2

        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        max_edges_per_vi = self.hparams.max_edges_per_node if training else self.hparams.test_max_edges_per_node
        sampled_edges = self.graph.get_sampled_edges(candidate_edges,
                                                     mode='by_vi',
                                                     max_edges_per_vi=max_edges_per_vi,
                                                     tc=get(tc, 'graph'))

        # scanned_edges: (np.array) n_scanned_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        scanned_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

        ''' add selfloop edges '''
        selfloop_edges = self.graph.get_selfloop_edges(attended_nodes, tc=get(tc, 'graph'))

        # aug_scanned_edges: n_aug_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        aug_scanned_edges = self.graph.get_union_edges(scanned_edges, selfloop_edges, tc=get(tc, 'graph'))

        ''' run attention flow (over memorized and scanned nodes) '''
        # new_idx_for_memorized: n_memorized_nodes x 1 or None
        # memorized_and_scanned: n_memorized_and_scanned_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        new_idx_for_memorized, n_memorized_and_scanned_nodes, memorized_and_scanned = \
            self.graph.add_nodes_to_memorized(scanned_edges, tc=get(tc, 'graph'))

        # aug_scanned_edges: n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        aug_scanned_edges = self.graph.set_index_over_nodes(aug_scanned_edges, memorized_and_scanned, tc=get(tc, 'graph'))

        new_node_attention = self.att_flow(node_attention,
                                           scanned_edges=aug_scanned_edges,
                                           hidden_uncon=hidden_uncon,
                                           hidden_con=hidden_con,
                                           shared_embedding=self.shared_embedding,
                                           new_idx_for_memorized=new_idx_for_memorized,
                                           n_memorized_and_scanned_nodes=n_memorized_and_scanned_nodes,
                                           query_head_emb=query_head_emb,
                                           query_rel_emb=query_rel_emb,
                                           tc=get(tc, 'model'))  # n_aug_scanned_edges, batch_size x n_nodes

        ''' get seen edges '''
        # seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique and sorted
        max_seen_nodes = self.hparams.max_seen_nodes if training else self.hparams.test_max_seen_nodes
        seen_nodes = self.graph.get_topk_nodes(new_node_attention, max_seen_nodes, tc=get(tc, 'graph'))  # n_seen_nodes x 2

        # seen_edges: (np.array) n_seen_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        seen_edges = self.graph.get_seen_edges(seen_nodes, aug_scanned_edges, tc=get(tc, 'graph'))

        ''' run consciousness flow (over memorized and seen nodes) '''
        # new_idx_for_memorized: n_memorized_nodes x 1 or None
        # memorized_and_seen: _memorized_and_seen_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        new_idx_for_memorized, n_memorized_and_seen_nodes, memorized_and_seen = \
            self.graph.add_nodes_to_memorized(seen_edges, inplace=True, tc=get(tc, 'graph'))

        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_seen_nodes, self.hparams.n_dims)))  # n_memorized_nodes (new) x n_dims

        # seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        seen_edges = self.graph.set_index_over_nodes(seen_edges, memorized_and_seen, tc=get(tc, 'graph'))

        new_hidden_con = self.con_flow(hidden_con,
                                       seen_edges=seen_edges,
                                       memorized_nodes=self.graph.memorized_nodes,
                                       node_attention=new_node_attention,
                                       hidden_uncon=hidden_uncon,
                                       query_head_emb=query_head_emb,
                                       query_rel_emb=query_rel_emb,
                                       shared_embedding=self.shared_embedding,
                                       tc=get(tc, 'model'))  # n_memorized_nodes (new) x n_dims

        if analyze_attention and not training:
            self.entropy_along_steps.append(tf.reduce_mean(entropy(new_node_attention)))
            self.top1_occupy_along_steps.append(tf.reduce_mean(topk_occupy(new_node_attention, 1)))
            self.top3_occupy_along_steps.append(tf.reduce_mean(topk_occupy(new_node_attention, 3)))
            self.top5_occupy_along_steps.append(tf.reduce_mean(topk_occupy(new_node_attention, 5)))

        # hidden_uncon: 1 x n_nodes x n_dims,
        # new_hidden_con: n_memorized_nodes x n_dims,
        # new_node_attention: batch_size x n_nodes
        return hidden_uncon, new_hidden_con, new_node_attention

    def save_attention_to_file(self, final_node_attention, id2entity, id2relation, epoch, dir_name, training=True):
        max_attended_nodes = self.hparams.max_attended_nodes if training else self.hparams.test_max_attended_nodes
        attended_nodes = self.graph.get_topk_nodes(final_node_attention, max_attended_nodes)  # n_attended_nodes x 2
        self.attended_nodes.append((attended_nodes, tf.gather_nd(final_node_attention, attended_nodes)))

        batch_size = len(self.heads)
        for batch_i in range(batch_size):
            head, rel, tail = self.heads[batch_i], self.rels[batch_i], self.tails[batch_i]
            filename = '{:d}->{:d}->{:d}.txt'.format(head, rel, tail)
            filename = 'train_epoch{:d}_'.format(epoch) + filename if training \
                else 'test_epoch{:d}_'.format(epoch) + filename

            with open(os.path.join(dir_name, filename), 'w') as fout:
                fout.write('nodes:\n')
                nodes = []
                for att_nodes, att_scores in self.attended_nodes:
                    mask = (att_nodes[:, 0] == batch_i)
                    att_nds = att_nodes[:, 1][mask]
                    att_scs = att_scores[mask]
                    nd_idx = np.argsort(-att_scs)
                    node_line = '\t'.join(['{:d}({}):{:f}'.format(att_nds[nd_i],
                                                                  id2entity[att_nds[nd_i]],
                                                                  att_scs[nd_i])
                                           for nd_i in nd_idx])
                    fout.write(node_line + '\n')
                    nodes.append(att_nds[nd_idx])

                fout.write('edges:\n')
                for i, nds_vi in enumerate(nodes[:-1]):
                    nds_vj = nodes[i+1]
                    edges = self.graph.get_vivj_edges(nds_vi, nds_vj, with_eg_idx=False)  # n_vivj_edges x 4, (edge_id, vi, vj, rel) sorted by (vi, vj)
                    edge_line = '\t'.join(['{:d}({})->{:d}({})->{:d}({})'.format(vi, id2entity[vi],
                                                                                 rel, id2relation[rel],
                                                                                 vj, id2entity[vj])
                                           for _, vi, vj, rel in edges])
                    fout.write(edge_line + '\n')

    @property
    def trainable_variables(self):
        return self.shared_embedding.trainable_variables + \
               self.uncon_flow.trainable_variables + \
               self.con_flow.trainable_variables + \
               self.att_flow.trainable_variables
