from __future__ import annotations
from typing import Any
from numpy import dtype
import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
import plotly.express as px
import community

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
    components = community.best_partition(graph, resolution=1.2, random_state=42)
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
