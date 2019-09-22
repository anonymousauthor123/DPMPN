import itertools
import numpy as np
import tensorflow as tf


def get(dct, k):
    return dct.get(k, None) if isinstance(dct, dict) else None


def get_segment_ids(x):
    """ x: (np.array) d0 x 2, sorted
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    y = (x[1:] == x[:-1]).astype('uint8')
    return np.concatenate([np.array([0], dtype='int32'),
                           np.cumsum(1 - y[:, 0] * y[:, 1], dtype='int32')])


def get_unique(x):
    """ x: (np.array) d0 x 2, sorted
    """
    if len(x) == 0:
        return x

    y = (x[1:] == x[:-1]).astype('uint8')
    return x[np.concatenate([np.array([1], dtype='bool'),
                             (1 - y[:, 0] * y[:, 1]).astype('bool')])]


def groupby_2cols_nlargest(x, y, k):
    """ x: (np.array) d0 x 2, sorted
        y: (np.array) d1
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    mask = (x[1:] == x[:-1]).astype('uint8')
    mask = (1 - mask[:, 0] * mask[:, 1]).astype('bool')
    n = len(x)
    key_idx = np.concatenate([np.array([0], dtype='int32'),
                              np.arange(1, n).astype('int32')[mask],
                              np.array([n], dtype='int32')])
    res_idx = np.concatenate([np.sort(s + np.argpartition(-y[s:e], min(k - 1, e - s - 1))[:min(k, e - s)])
                              for s, e in zip(key_idx[:-1], key_idx[1:])])
    return res_idx.astype('int32')


def groupby_1cols_nlargest(x, y, k):
    """ x: (np.array) d0, sorted
        y: (np.array) d1
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    mask = (x[1:] != x[:-1])
    n = len(x)
    key_idx = np.concatenate([np.array([0], dtype='int32'),
                              np.arange(1, n).astype('int32')[mask],
                              np.array([n], dtype='int32')])
    res_idx = np.concatenate([np.sort(s + np.argpartition(-y[s:e], min(k - 1, e - s - 1))[:min(k, e - s)])
                              for s, e in zip(key_idx[:-1], key_idx[1:])])
    return res_idx.astype('int32')


def groupby_1cols_merge(x, x_key, y_key, y_id):
    """ x (group by): (np.array) d0, sorted
        x_key (merge left key): (np.array): d0, unique in group
        y_key (merge right key): (np.array): d1
        y_id: (np.array): d1
    """
    mask = (x[1:] != x[:-1])
    n = len(x)
    key_idx = np.concatenate([np.array([0], dtype='int32'),
                              np.arange(1, n).astype('int32')[mask],
                              np.array([n], dtype='int32')])
    yid_li = [y_id[np.in1d(y_key, x_key[s:e])]
              for s, e in zip(key_idx[:-1], key_idx[1:])]
    res_idx = np.concatenate(yid_li)
    grp_idx = np.concatenate([np.repeat(np.array([i], dtype='int32'), len(yid)) for i, yid in enumerate(yid_li)])
    return res_idx, grp_idx


def groupby_1cols_cartesian(x, v1, y, v2):
    """ x: d0, sorted
        v1: d0
        y: d1, sorted
        v2: d1
    """
    mask = (x[1:] != x[:-1])
    n = len(x)
    x_key_idx = np.concatenate([np.array([0], dtype='int32'),
                                np.arange(1, n).astype('int32')[mask],
                                np.array([n], dtype='int32')])
    mask = (y[1:] != y[:-1])
    n = len(y)
    y_key_idx = np.concatenate([np.array([0], dtype='int32'),
                                np.arange(1, n).astype('int32')[mask],
                                np.array([n], dtype='int32')])
    batch_size = len(x_key_idx) - 1
    return np.array([(eg_idx, vi, vj)
                     for eg_idx, s1, e1, s2, e2 in zip(np.arange(batch_size),
                                                       x_key_idx[:-1],
                                                       x_key_idx[1:],
                                                       y_key_idx[:-1],
                                                       y_key_idx[1:])
                     for vi, vj in itertools.product(v1[s1:e1], v2[s2:e2])], dtype='int32')


def entropy(x):
    return tf.reduce_sum(- tf.math.log(tf.math.maximum(x, 1e-20)) * x, axis=-1)


def topk_occupy(x, k):
    values, _ = tf.math.top_k(x, k=k)
    return tf.reduce_sum(values, axis=-1) / tf.reduce_sum(x, axis=-1)
