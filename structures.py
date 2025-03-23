""" DOCSTRING... """

from __future__ import annotations
import numpy as np
from numpy import dtype, ndarray
from typing import Any, Optional


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


class SimGraph:
    """A similarity graph.

    Representation Invariants:
        - all(item == self._vertices[item].item for item in self._vertices)
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps id to _Vertex object.
    #     - _vector_table:
    #        A dictionary mapping a song id to its feature vector
    #     - _THRESHOLD:
    #        - A numeric threshold which determines whether two vertices are connected by an edge
    _vertices: dict[Any, _Vertex]
    _vector_table: dict[Any, np.ndarray[Any, dtype]]
    _THRESHOLD = 0.5

    def __init__(self, ids: np.ndarray[Any, dtype], vector_table: dict[Any, np.ndarray[Any, dtype]]) -> None:
        """Generate a similarity graph with the given song ids and vector table"""
        self._vector_table = vector_table
        self.add_vertices(ids)

    def add_vertices(self, ids: np.ndarray[Any, dtype]) -> None:
        """ Insert a list of vertices and fill in neighbours based on
        similarity scores
        """
        for i in range(ids.shape[0]):
            self._vertices[ids[i]] = _Vertex(ids[i], set())
            for vertex_id in self._vertices:
                self.add_edge(ids[i], vertex_id)

    def add_edge(self, id1: Any, id2: Any) -> None:
        """ Add an edge between two vertices if they are similar enough"""

        if self.similarity(self._vector_table[id1], self._vector_table[id2]) <= self._THRESHOLD:
            self._vertices[id1].neighbours.add(self._vertices[id2])
            self._vertices[id2].neighbours.add(self._vertices[id1])

    def similarity(self, id1: Any, id2: Any) -> float:
        """ Calculate the similarity between two feature vectors using euclidean distance
        """
        vector1 = self._vector_table[id1]
        vector2 = self._vector_table[id2]

        return np.sum((vector1 - vector2)**2)
