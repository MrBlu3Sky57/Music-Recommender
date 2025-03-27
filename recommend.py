from __future__ import annotations
from typing import Any
from numpy import dtype
import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from Graph import SongGraph
from Graph import cosine_sim
from matplotlib import pyplot as plt
import plotly.express as px
import community

import parse

def build_nx(ids: list[Any], vector_table: dict[Any, np.ndarray[Any, dtype]], threshold: float = 0.33, k: int = 15) -> nx.graph:
    vectors = np.array([vector_table[i] for i in ids])
    tree = KDTree(vectors, metric='euclidean')
    distances, indices = tree.query(vectors, k=k+1)

    g = nx.Graph()
    for i, song in enumerate(ids):
        g.add_node(song)  
        for j in range(1, k + 1):
            neighbour = indices[i][j]
            neighbour_id = ids[neighbour]
            weight = 1 / (1 + distances[i][j])
            if weight >= threshold:
                g.add_edge(song, neighbour_id, weight=weight)
    return g

def partition_graph(graph: nx.Graph) -> tuple[dict[Any, int], dict[int, Any]]:
    components = community.best_partition(graph)
    cluster_dict = {}
    for i, cluster_val in components.items():
        if cluster_val in cluster_dict:
            cluster_dict[cluster_val].append(i)
        else:
            cluster_dict[cluster_val] = [i]
    return components, cluster_dict

def visualize(graph: nx.Graph, vector_table: np.ndarray[Any, dtype], name_table: dict[Any, Any]) -> bool:
    """
    Visualize Graph.
    """
    feature_matrix = np.array([vector_table[i] for i in graph.nodes()])
    if feature_matrix.shape[0] == 1:
        return False

    perplexity = min(30.0, feature_matrix.shape[0])
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vis_features = tsne.fit_transform(feature_matrix)

    partition, cluster_dict = partition_graph(graph)
    print(len(cluster_dict))
    communities = [partition[node] for node in graph.nodes()]

    # Visual
    x_coords = vis_features[:, 0]
    y_coords = vis_features[:, 1]
    hover_data = [name_table[i] for i in graph.nodes()]
    colors = communities

    fig = px.scatter(
        x=x_coords, 
        y=y_coords, 
        color=colors,
        hover_name=hover_data,
        title="Song Communities Visualization",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()
    return True

def name_to_id(name_table: dict[Any, Any]) -> dict[Any, Any]:
    return {name: i for i, name in name_table.items()}

if __name__ == '__main__':
    labels, vectors, identifiers = parse.parse_file('high_popularity_spotify_data.csv', vector_idx=parse.VECTOR_IDX_HIGH, id_idx=parse.ID_IDX_HIGH)
    normed_vectors, mins, maxes = parse.normalize(vectors)
    ids, vector_table, name_table = parse.build_tables(normed_vectors, identifiers)
    id_table = name_to_id(name_table)
    ids = list(set(ids))

    g = build_nx(ids, vector_table)
    visualize(g, vector_table, name_table)
    partition, cluster_dict = partition_graph(g)
    song = id_table['Die With A Smile']
    com = partition[song]
    graph = SongGraph(np.array(list(cluster_dict[com])), vector_table)
    graph.visualize_heat_map(song, vector_table, name_table)
    print(graph.recommendations(song, vector_table, name_table))
    
    # res = np.linspace(1, 2, 10)
    # thresh = np.linspace(0.5, 1.0, 10)
    # thresh_max = 0.95
    # mods = np.zeros((10))
    # g = build_nx(ids, vector_table, thresh_max)
    # for i in range(len(res)):
    #     partition, cluster_dict = partition_graph(g, 1.0)
    #     mods[i] = community.modularity(partition, g, weight='weight')
    
    # plt.plot(res, mods)
    # plt.show()