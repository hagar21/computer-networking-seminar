import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random


def combined_graphs_edges(bigGraph, smallGraph):
    for u, v, data in smallGraph.edges(data=True):
        w = data['weight']
        if bigGraph.has_edge(u, v):
            bigGraph[u][v][0]['weight'] += w
        else:
            bigGraph.add_edge(u, v, weight=w)

    return bigGraph


def network_plot_3D(G, angle, save, nodeDict):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = 0
    for node in G.nodes:
        if edge_max < G.degree[node]:
            edge_max = G.degree[node]
    
    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree[i]/edge_max) for i in G.nodes] 

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20+20*G.degree(key), edgecolors='k', alpha=0.7)
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            j = j[1]
            j = nodeDict[j] 
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
    
    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
         plt.savefig(os.getcwd()+str(angle).zfill(3)+".png")
         plt.close('all')
    else:
          plt.show()
    
    return


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


def plot_degree_graph(freq, title):
    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(freq)), freq, 'bo-')
    plt.xlabel(title)
    plt.ylabel('Frequency')

    num = '(a) '
    if title == 'Indegree':
        num = '(b) '
    else:
        if title == 'Outdegree':
            num = '(c) '

    plt.title(num + title)
    plt.savefig(graphs_root_folder + '/' + title + '.png')


def plot_degree_dist(G):
    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)
    degree_freq = degree_histogram_directed(G)

    plot_degree_graph(in_degree_freq, 'Indegree')
    plot_degree_graph(out_degree_freq, 'Outdegree')
    plot_degree_graph(degree_freq, 'Degree')


def page_rank(G, number_of_elements):
    pk = nx.pagerank_numpy(G, weight=None)
    # return sorted(pk.items(), key=lambda x:x[1], reverse=True)[:10]
    # return heapq.nlargest(number_of_elements, pk.items(), key=lambda x: x[1])
    return max(pk.items(), key=lambda x: x[1])


def degree_centrality(G, number_of_elements):
    # return sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[0:10]
    dc = nx.degree_centrality(G)
    # return heapq.nlargest(number_of_elements, dc.items(), key=lambda x: x[1])
    return max(dc.items(), key=lambda x: x[1])


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


def draw_3d(H):
    # reorder nodes from 0,len(G)-1
    G = nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars = np.array(list(G.nodes())) + 5

    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.1,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    print("saving...")
    plt.savefig(graphs_root_folder + '/' + 'graph.png')


def network_plot_3d(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()
    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree[i] for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    if save is not False:
        plt.savefig(graphs_root_folder + '/' + '.png')
        plt.close('all')
    else:
        plt.show()

    return


def plot_3d_network(graph, angle):
    pos = nx.get_node_attributes(graph, 'weight')

    with plt.style.context("bmh"):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            ax.scatter(xi, yi, zi, edgecolor='b', alpha=0.9)
            for i, j in enumerate(graph.edges()):
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))
                ax.plot(x, y, z, c='black', alpha=0.9)
    ax.view_init(30, angle)
    pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
    plt.show()


def gen_random_3d_graph(n_nodes, radius):
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    graph = nx.random_geometric_graph(n_nodes, radius, pos=pos)
    return graph


if __name__ == '__main__':
    graphs_root_folder = 'C:/Users/hagarsheffer/PycharmProjects/dsProj/graphs'
    usecols = ['from_address', 'to_address', 'value']
    # dirname = '/mnt/s/Geth/ethereum-etl/output/transactions/'
    dirname = 'C:/Users/hagarsheffer/PycharmProjects/dsProj/trans'

    g = nx.MultiDiGraph()
    # os.chdir('/mnt/s/Geth/ethereum-etl/output/transactions/')
    os.chdir('C:/Users/hagarsheffer/PycharmProjects/dsProj/trans')
    for start_block in os.listdir(dirname):
        temp_dir = start_block.split('=')[1]
        if int(temp_dir) >= 2:
            end_block = os.listdir(start_block)[0]
            filename = os.listdir(start_block + '/' + end_block)[0]
            pip = start_block + '/' + '/' + end_block + '/' + filename
            print(pip)
            p = pd.read_csv(
                pip,
                usecols=usecols,
                error_bad_lines=False,
                index_col=False,
                dtype='unicode',
                low_memory=False,
            )
            p = p[p.value != 0]
            p = p.dropna()
            p['weight'] = p.groupby(['from_address', 'to_address'])['value'].transform('sum')
            p.drop_duplicates(subset=['from_address', 'to_address'], inplace=True)
            gc = nx.convert_matrix.from_pandas_edgelist(
                df=p,
                source='from_address',
                target='to_address',
                edge_attr='weight',
                create_using=nx.MultiDiGraph(),
            )
            g = combined_graphs_edges(g, gc)

    print(g.edges)
    # print('Number of accounts:', g.number_of_nodes())
    # plot_degree_dist(g)
    #
    # diG = nx.Graph(g)
    # print('Clustering coefficient:', nx.average_clustering(diG))
    #
    # print('Assortativity coefficient:', nx.degree_assortativity_coefficient(g))
    # print('Pearson coefficient:', nx.degree_pearson_correlation_coefficient(g))
    # print('SCC/WCC:')
    # print('     Largest SCC size:', len(max(nx.strongly_connected_components(g), key=len)))
    # print('     Number of SCCs:', nx.number_strongly_connected_components(g))
    # print('     Number of WCCs:', nx.number_weakly_connected_components(g))
    #
    # print('Top 10 most important nodes in MFG, ranked by PageRank')
    # print(page_rank(g, 10), '\n')
    # print('Top 10 most important nodes in MFG, ranked by degree centrality')
    # print(degree_centrality(g, 10), '\n')
    #
    # draw_3d(g)

    # draw_graph(g)
    # i = 0
    # nodeDict = {}
    # for node in g.nodes:
    #     nodeDict[node] = i
    #     i += 1
    #
    # diG = nx.Graph(g)
    # graph01 = gen_random_3d_graph(15, 0.6)
    # plot_3d_network(graph01, 0)
    # network_plot_3D(diG, 0, True, nodeDict)
    # plot_3d_network(diG, 0)