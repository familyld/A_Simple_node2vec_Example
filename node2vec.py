# -*- coding: utf-8 -*-
"""
Created on Sat Jun 8th 2019

@author: Zhi-Hong Deng
"""

import numpy as np
import networkx as nx
import os
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

sns.set()

num_nodes = 30      # cross: 11, flow: 26, crab: 10, kite: 16, bridge: 15, flower: 30
num_walks = 100
walk_length = 10
emb_size = 2
iteration = 200
path = './flower.txt' # cross.txt, flow.txt, crab.txt, kite.txt, bridge.txt, flower.txt

def load_graph(filename):
    g = nx.Graph()
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line_split = line.split()
            src, dst = line_split[0], line_split[1]
            g.add_edge(src, dst)
            if g[src][dst].get('weight') == None:
                g[src][dst]['weight'] = 1
            else:
                g[src][dst]['weight'] += 1
            line = f.readline()
    return g

def preprocess_transition_probs(g, p=1, q=1):
    alias_nodes, alias_edges = {}, {};
    for node in g.nodes():
        probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
        norm_const = sum(probs)
        norm_probs = [float(prob) / norm_const for prob in probs]
        alias_nodes[node] = get_alias_nodes(norm_probs)
    
    for edge in g.edges():
        alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
        alias_edges[(edge[1], edge[0])] = get_alias_edges(g, edge[1], edge[0], p, q)

    return alias_nodes, alias_edges


def get_alias_edges(g, src, dest, p=1, q=1):
    probs = [];
    for nei in sorted(g.neighbors(dest)):
        if nei==src:
            probs.append(g[dest][nei]['weight'] / p)
        elif g.has_edge(nei, src):
            probs.append(g[dest][nei]['weight'])
        else:
            probs.append(g[dest][nei]['weight'] / q)
    norm_probs = [float(prob) / sum(probs) for prob in probs]
    return get_alias_nodes(norm_probs)

def get_alias_nodes(probs):
    l = len(probs)
    a, b = np.zeros(l), np.zeros(l, dtype=np.int)
    small, large = [], []

    for i, prob in enumerate(probs):
        a[i] = l*prob
        if a[i] < 1.0:
            small.append(i)
        else:
            large.append(i)
            
    while small and large:
        sma, lar = small.pop(), large.pop()
        b[sma] = lar
        a[lar] += a[sma]-1.0
        if a[lar] < 1.0:
            small.append(lar)
        else:
            large.append(lar)
    return b, a


def node2vec_walk(g, start, alias_nodes, alias_edges, walk_length=30):
    path = [start]
    while len(path) < walk_length:
        node = path[-1]
        neis = sorted(g.neighbors(node))
        if len(neis) > 0:
            if len(path) == 1:
                l = len(alias_nodes[node][0])
                idx = int(np.floor(np.random.rand()*l))
                if np.random.rand() < alias_nodes[node][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_nodes[node][0][idx]])
            else:
                prev = path[-2]
                l = len(alias_edges[(prev, node)][0])
                idx = int(np.floor(np.random.rand()*l))
                if np.random.rand() < alias_edges[(prev, node)][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_edges[(prev, node)][0][idx]])
        else:
            break
    return path 

def get_wv(g, id_list, p=1, q=1):
    alias_nodes, alias_edges = preprocess_transition_probs(g, p, q)

    walks = []
    idx_total = []
    for i in range(num_walks):
        r = np.array(range(len(id_list)))
        np.random.shuffle(r)

        for node in [id_list[j] for j in r]:
            walks.append(node2vec_walk(g, node, alias_nodes, alias_edges, walk_length))
    model = Word2Vec(walks, size=emb_size, min_count=0, sg=1, window=5, iter=iteration)

    wv = []
    for i in range(num_nodes):
        wv.append(model.wv.get_vector(str(i)))
    wv = np.array(wv)
    # model = TSNE(perplexity=5)
    # wv = model.fit_transform(wv)
    return wv, walks

def normalization(a, b):
    a = (a - a.min())/(a.max() - a.min())
    b = (b - b.min())/(b.max() - b.min())
    return a, b

def main():
    id_list = list(range(num_nodes))
    id_list = [str(id) for id in id_list]
    g = load_graph(path)
    plt.figure()
    nx.draw(g, with_labels=True)
    
    plt.figure(figsize=(18, 9))
    # DFS
    p, q = 1.0, 0.1
    wv, dfs_walks = get_wv(g, id_list, p=p, q=q)
    plt.subplot(121)
    x, y = normalization(wv[:,0], wv[:,1])
    plt.plot(x, y, 'o')
    for n in range(num_nodes):
        plt.text(x[n], y[n], str(n), ha='right', va='bottom', fontsize=12)
    plt.title('DFS: p=%.1f, q=%.1f'%(p, q), fontsize='large', fontweight='bold')

    # BFS
    p, q = 1.0, 10.0
    wv, bfs_walks = get_wv(g, id_list, p=p, q=q)
    plt.subplot(122)
    x, y = normalization(wv[:,0], wv[:,1])
    plt.plot(x, y, 'o')
    for n in range(num_nodes):
        plt.text(x[n], y[n], str(n), ha='right', va='bottom', fontsize=12)
    plt.title('BFS: p=%.1f, q=%.1f'%(p, q), fontsize='large', fontweight='bold')

    for i in dfs_walks:
        print("dfs: ", i)
    for i in bfs_walks:
        print("bfs: ", i)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
