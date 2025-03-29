from __future__ import annotations
from typing import Any
import numpy as np
from numpy import dtype
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
import plotly.graph_objects as go


def cosine_sim(A, B) -> Any:
    return 1 - cosine(A, B)

class _Vertex:
    """A vertex in a similarity graph, representing a song.

    Instance Attributes:
        - song_id: The song's id
        - neighbours: The vertices that are adjacent to this vertex.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
    """
    song_id: str
    neighbours: set[_Vertex]

    def __init__(self, song_id: Any, neighbours: set[_Vertex]) -> None:
        """Initialize a new vertex with the given item and neighbours."""
        self.song_id = song_id
        self.neighbours = neighbours


class SongGraph:
    """A similarity graph.

    Representation Invariants:
        - all(item == self._vertices[item].item for item in self._vertices)
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps id to _Vertex object.
    #     - _threshold:
    #        - A numeric threshold which determines whether two vertices are connected by an edge
    _vertices: dict[Any, _Vertex]

    def __init__(self, ids: np.ndarray[Any, dtype], vector_table: dict[Any, np.ndarray[Any, dtype]], threshold: int = 0.95) -> None:
        """Generate a song graph with the given song ids, vector table and name table"""
        self._vertices = {}
        self._threshold = threshold
        self._add_vertices(ids, vector_table)

    def _add_vertices(self, ids: np.ndarray[Any, dtype], vector_table: dict[Any, np.ndarray[Any, dtype]]) -> None:
        """ Insert a list of vertices and fill in neighbours based on
        similarity scores
        """
        for i in range(ids.shape[0]):
            self._vertices[ids[i]] = _Vertex(ids[i], set())
            for vertex_id in self._vertices:
                self._add_edge(ids[i], vertex_id, vector_table)

    def _add_edge(self, id1: Any, id2: Any, vector_table: dict[Any, np.ndarray[Any, dtype]]) -> bool:
        """ Add an edge between two vertices if they are similar enough"""

        if cosine_sim(vector_table[id1], vector_table[id2]) >= self._threshold:
            self._vertices[id1].neighbours.add(self._vertices[id2])
            self._vertices[id2].neighbours.add(self._vertices[id1])
            return True
        return False

    def similarity_matrix(self, vector_table: dict[Any, np.ndarray[Any, dtype]]) -> np.ndarray[Any, dtype]:
        """ Return similarity matrix of graph"""
        feature_matrix = np.array([vector_table[i] for i in self._vertices])

        return cosine_similarity(feature_matrix)
    
    def adjacency_matrix(self, similarity_matrix: np.ndarray[Any, dtype]) -> np.ndarray[Any, dtype]:
        """ Return adjacency matrix of the graph"""
        return (similarity_matrix >= self._threshold).astype(float)
    
    def diffusion(self, seed_id: Any, vector_table: dict[Any, np.ndarray[Any, dtype]],  soft_table: dict[Any, np.ndarray[Any, dtype]], alpha: float=0.85, iterations: int=8) -> list:
        """ Return a diffused state for each song in the graph"""
        adj_matrix = self.adjacency_matrix(self.similarity_matrix(vector_table))
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj_matrix = adj_matrix / row_sums
        ids = list(self._vertices.keys())

        initial_state = np.zeros(adj_matrix.shape[0])
        initial_state[ids.index(seed_id)] = 1.0
        state = initial_state
        for _ in range(iterations):
            new_state = np.dot(adj_matrix.T, state)
            state = (1 - alpha) * state + alpha * new_state
        self.apply_naive_scores(ids, seed_id, state, soft_table)
        return state / np.max(state)
    
    def apply_naive_scores(self, ids: list[Any], song_id: Any, state: np.ndarray[Any, dtype], soft_table: dict[Any, np.ndarray[Any, dtype]], bias: int =0.02) -> None:
        for i in range(state.shape[0]):
            if soft_table[ids[i]][0] == soft_table[song_id][0]:
                state[i] += bias
            if soft_table[ids[i]][2] == soft_table[song_id][2]:
                state[i] += bias
            if soft_table[ids[i]][3] == soft_table[song_id][3]:
                state[i] += bias
            state[i] += (float(soft_table[ids[i]][1]) / 100) * bias / 2

    def recommendations(self, song_id: Any, vector_table: dict[Any, np.ndarray[Any, dtype]],  soft_table: dict[Any, np.ndarray[Any, dtype]], name_table: dict[Any, Any], limit: int=10) -> list[Any]:
        """Return recommendations based on graph diffusion"""
        state = self.diffusion(song_id, vector_table, soft_table)
        sorted_list = np.argsort(state)[::-1][:(limit + 1)]
        ids = list(self._vertices.keys())
        return [name_table[ids[i]] for i in sorted_list if ids[i] != song_id]
    
    def emb_features(self, vector_table:dict[Any, np.ndarray[Any, dtype]], song_id: Any) -> np.ndarray[Any, dtype]:
        """Prepare a networkx graph for community visualization"""
        feature_matrix = np.array([vector_table[i] for i in self._vertices if i != song_id])
        perplexity = min(30, feature_matrix.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        vis_features = tsne.fit_transform(feature_matrix)

        return vis_features
    
    def visualize_heat_map(self, song_id: Any, vector_table: dict[Any, np.ndarray[Any, dtype]], soft_table: dict[Any, np.ndarray[Any, dtype]], name_table: dict[Any, Any]) -> bool:
            state = list(self.diffusion(seed_id=song_id, vector_table=vector_table, soft_table=soft_table))
            id_list = list(self._vertices.keys())
            idx = id_list.index(song_id)
            state.pop(idx)
            if len(id_list) <= 1:
                return False
            vis_features = self.emb_features(vector_table=vector_table, song_id=song_id)

            x_coords = vis_features[:, 0]
            y_coords = vis_features[:, 1]
            hover_data = [name_table[i] for i in self._vertices if i != song_id]

            fig = go.Figure(data=go.Scatter(
                x=x_coords, 
                y=y_coords, 
                mode='markers',
                marker=dict(
                    size=5,
                    color=state, 
                    colorscale='Viridis',
                    colorbar=dict(title="Similarity")
                ),
                text=hover_data,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="Song Communities Visualization",
                xaxis_title="TSNE Component 1",
                yaxis_title="TSNE Component 2"
            )

            fig.show()
            return True
