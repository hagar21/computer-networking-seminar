import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter


def combined_graphs_edges(bigGraph, smallGraph):
    for u, v, data in smallGraph.edges(data=True):
        w = data['weight']
        if bigGraph.has_edge(u, v):
            bigGraph[u][v][0]['weight'] += w
        else:
            bigGraph.add_edge(u, v, weight=w)

    return bigGraph


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq


def plot_degree_dist(G):
    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)
    degree_freq = degree_histogram_directed(G)

    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree')
    plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
    plt.loglog(range(len(degree_freq)), degree_freq, 'ro-', label='degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('MFG Degree distribution')
    plt.legend()
    plt.show()


def page_rank(G):
    pk = nx.pagerank_numpy(G, weight=None)
    return sorted(pk.items(), key=lambda x:x[1], reverse=True)[:10]


def degree_centrality(G):
    return sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[0:10]


def draw_graph(G):
    # nx.draw_spring(G, with_labels=True)
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1, \
            node_size=500, node_color='pink', alpha=0.9, \
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    usecols = ['from_address', 'to_address', 'value']
    dirname = 'files'

    g = nx.MultiDiGraph()

    for filename in os.listdir(dirname):
        print(filename)
        p = pd.read_csv(dirname+'/'+filename, usecols=usecols)
        p['weight'] = p.groupby(['from_address', 'to_address'])['value'].transform('sum')
        p.drop_duplicates(subset=['from_address', 'to_address'], inplace=True)
        gc = nx.convert_matrix.from_pandas_edgelist(df=p, source='from_address', target='to_address', edge_attr='weight', create_using=nx.MultiDiGraph())
        g = combined_graphs_edges(g, gc)

    print('Number of accounts:', g.number_of_nodes())
    plot_degree_dist(g)

    diG = nx.Graph(g)
    print('Clustering coefficient:', nx.average_clustering(diG))

    print('Assortativity coefficient:', nx.degree_assortativity_coefficient(g))
    print('Pearson coefficient:', nx.degree_pearson_correlation_coefficient(g))
    print('SCC/WCC:')
    print('     Largest SCC size:', len(max(nx.strongly_connected_components(g), key=len)))
    print('     Number of SCCs:', nx.number_strongly_connected_components(g))
    print('     Number of WCCs:', nx.number_weakly_connected_components(g))

    print('Top 10 most important nodes in MFG, ranked by PageRank')
    print(page_rank(g), '\n')
    print('Top 10 most important nodes in MFG, ranked by degree centrality')
    print(degree_centrality(g), '\n')

    draw_graph(g)
