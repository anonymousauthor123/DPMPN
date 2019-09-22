import os
import glob

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import argparse

import datasets

def draw_a_graph(filename, dataset, topk_all=None, topk_per_step=None, font_size=4, node_size=100, edge_width=0.5, disable_draw=False):
    nodes_per_step = []
    rels_dct = {}
    edges_set = {}
    head, relation, tail = os.path.basename(filename)[len('test_epoch1_'):].split('.')[0].split('->')
    head = head.split('(')[0]
    relation = relation.split('(')[0]
    tail = tail.split('(')[0]
    with open(filename) as fin:
        mode = None
        for line in fin.readlines():
            line = line.strip()
            if line == 'nodes:':
                mode = 'nodes'
            elif line == 'edges:':
                mode = 'edges'
            else:
                if mode == 'nodes':
                    nodes = []
                    max_att = 0.
                    for sp in line.split('\t'):
                        sp2 = sp.split(':')
                        node_att = float(sp2[1])
                        max_att = max(max_att, node_att)
                        sp2 = sp2[0].split('(')
                        node_id = sp2[0]
                        node_name = sp2[1][:-1]
                        nodes.append((node_id, node_name, node_att))

                    if topk_per_step is not None:
                        sorted(nodes, key=lambda node: - node[2])
                        nodes = nodes[:topk_per_step]

                    nodes = {node_id: (node_name, node_att / max_att) for node_id, node_name, node_att in nodes}
                    nodes_per_step.append(nodes)

                elif mode == 'edges':
                    edges = []
                    for sp in line.split('\t'):
                        sp2 = sp.split('->')
                        node_id1 = sp2[0].split('(')[0]
                        node_id2 = sp2[2].split('(')[0]
                        rel_id = sp2[1].split('(')[0]
                        rel_name = sp2[1].split('(')[1][:-1]
                        rels_dct[rel_id] = rel_name
                        edges.append((node_id1, rel_id, node_id2))
                        if (node_id1, rel_id, node_id2) not in edges_set:
                            edges_set[(node_id1, rel_id, node_id2)] = len(edges_set)

    nodes = {}
    n_steps = len(nodes_per_step)
    i = 0
    for t in range(n_steps):
        for k, v in nodes_per_step[t].items():
            node_id = k
            node_name = v[0]
            node_att = v[1]
            att_h = node_att * np.power(0.9, t)
            att_t = node_att * np.power(0.9, n_steps-1-t)

            att = 0.5 - att_h / 2 if att_h > att_t else 0.5 + att_t / 2
            node_att = nodes[node_id][1] if node_id in nodes else 0.5
            nodes[node_id] = (node_name, att) if abs(att - 0.5) > abs(node_att - 0.5) else (node_name, node_att)

    nodes_all = [(i, {'id': k, 'name': v[0], 'att': v[1]}) for i, (k, v) in enumerate(nodes.items())]

    if topk_all is not None:
        sorted(nodes_all, key=lambda node: - node[1]['att'])
        nodes_all = [(i, e[1]) for i, e in enumerate(nodes_all[:topk_all])]

    id2i = {node[1]['id']: node[0] for node in nodes_all}
    i2id = {node[0]: node[1]['id'] for node in nodes_all}

    edges = list(edges_set.keys())
    sorted(edges, key=lambda e: edges_set[e])
    edges_all = [(id2i[n1], id2i[n2], {'rel_id': r, 'rel_name': rels_dct[r]})
                 for n1, r, n2 in edges if n1 in id2i and n2 in id2i]

    if not disable_draw:
        graph = nx.MultiGraph()
        graph.add_nodes_from(nodes_all)
        graph.add_edges_from(edges_all)

        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')

        cdict = {'red': [(0.0, 1.0, 1.0), (0.25, 1.0, 1.0), (0.5, 0.7, 0.7), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)],
                 'green': [(0.0, 0.9, 0.9), (0.25, 0.7, 0.7), (0.5, 0.7, 0.7), (0.75, 0.7, 0.7), (1.0, 0.2, 0.2)],
                 'blue': [(0.0, 0.0, 0.0), (0.25, 0.2, 0.2), (0.5, 0.7, 0.7), (0.75, 0.2, 0.2), (1.0, 0.0, 0.0)]}
        attcmp = colors.LinearSegmentedColormap('attCmap', segmentdata=cdict, N=256)

        def get_node_params(nodes, size=100, inflate=0, min_max=0.1):
            node_atts = [n[1]['att'] for n in nodes]
            cols = node_atts
            c = np.array(cols)
            vmin = c.min()
            vmax = max(c.max(), min_max)
            sizes = [size * 2.5 if nodes[i][1]['id'] == head or nodes[i][1]['id'] == tail else size * (1+a*inflate)
                     for i, a in enumerate(node_atts)]
            return cols, vmin, vmax, sizes

        cols, vmin, vmax, sizes = get_node_params(nodes_all, size=node_size)
        nx.draw_networkx_nodes(graph, pos, node_color=cols, vmin=vmin, vmax=vmax, cmap=attcmp, node_size=sizes,
                               edgecolors='y', alpha=0.95, linewidths=0.5)

        def get_edge_params(edges, nodes, width=1., inflate=0):
            node_atts = [n[1]['att'] for n in nodes]
            edge_atts = [ 0.5 - (abs(node_atts[e[0]] - 0.5) + abs(node_atts[e[1]] - 0.5)) / 2.
                         if (node_atts[e[0]] + node_atts[e[1]]) / 2. < 0.5  else
                          0.5 + (abs(node_atts[e[0]] - 0.5) + abs(node_atts[e[1]] - 0.5)) / 2.
                         for e in edges]
            cols = edge_atts
            c = np.array(cols)
            vmin = c.min()
            vmax = c.max()
            widths = [width * (1+a*inflate) for a in edge_atts]
            return cols, vmin, vmax, widths

        e_cols, e_vmin, e_vmax, widths = get_edge_params(edges_all, nodes_all, width=edge_width)
        nx.draw_networkx_edges(graph, pos, width=widths, edge_color=e_cols, edge_vmin=e_vmin, edge_vmax=e_vmax,
                               edge_cmap=attcmp, arrowstyle='->', alpha=0.7)

        nx.draw_networkx_labels(graph, pos, labels={nd[0]:nd[1]['name'] for nd in nodes_all},
                                font_size=font_size, font_color='k')

        assert nodes_all[id2i[head]][1]['name'] == dataset.id2entity[int(head)]
        if tail not in id2i:
            plt.title('{} - {} (missed)'.format(head, tail))
        # else:
        #     assert nodes_all[id2i[tail]][1]['name'] == dataset.id2entity[int(tail)]
        #     plt.title('{} - {}'.format(head, tail))

    edges = [(i2id[e[0]], e[2]['rel_id'], i2id[e[1]]) for e in edges_all]
    return head, relation, tail, edges


def draw(dataset, dirpath, new_dirpath):
    if not os.path.exists(new_dirpath):
        os.mkdir(new_dirpath)

    for filename in glob.glob(os.path.join(dirpath, '*.txt')):
        try:
            print(filename)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            draw_a_graph(filename, dataset, font_size=3)
            plt.subplot(1, 2, 2)
            draw_a_graph(filename, dataset, topk_per_step=5, font_size=5, node_size=180, edge_width=1)
            plt.tight_layout()
            plt.savefig(os.path.join(new_dirpath, os.path.basename(filename)[:-4] + '.pdf'), format='pdf')
            plt.close()

            head, rel, tail, edges = draw_a_graph(filename, dataset, topk_per_step=3, font_size=5, node_size=180, edge_width=1)
            with open(os.path.join(new_dirpath, os.path.basename(filename)), 'w') as fout:
                fout.write('{}\t{}\t{}\n\n'.format(dataset.id2entity[int(head)],
                                                     dataset.id2relation[int(rel)],
                                                     dataset.id2entity[int(tail)]))
                for h, r, t in edges:
                    fout.write('{}\t{}\t{}\n'.format(dataset.id2entity[int(h)],
                                                       dataset.id2relation[int(r)],
                                                       dataset.id2entity[int(t)]))
        except IndexError:
            print('Cause `IndexError` for file `{}`'.format(filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None,
                        choices=['FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995'])
    args = parser.parse_args()

    ds = getattr(datasets, args.dataset)()
    if args.dataset == 'NELL995':
        nell995_cls = getattr(datasets, args.dataset)
        for ds in nell995_cls.datasets():
            print('nell > ' + ds.name)
            dir_name = '../output/NELL995_subgraph/' + ds.name
            if not os.path.exists(dir_name):
                continue
            dir_name_2 = '../visual/NELL995_subgraph/' + ds.name
            os.makedirs(dir_name_2, exist_ok=True)
            draw(ds, dir_name, dir_name_2)
    else:
        ds = getattr(datasets, args.dataset)()
        print(ds.name)
        dir_name = '../output/' + ds.name + '_subgraph'
        dir_name_2 = '../visual/' + ds.name + '_subgraph'
        os.makedirs(dir_name_2, exist_ok=True)
        draw(ds, dir_name, dir_name_2)
