from typing import Any
import numpy as np
import networkx as nx
from queue import Queue
import random

class Leiden:
    graph: nx.Graph
    degrees: dict[Any, int]
    partition: list[set[Any]]
    communities: dict[Any, int]
    M: int
    gamma: float

    def __init__(self, graph: nx.Graph, gamma: float=2.0) -> None:
        self.graph = graph
        self.degrees = dict(graph.degree(weight='weight'))
        self.partition = [{node} for node in graph.nodes()]
        self.communities = {node: i for i, node in enumerate(graph.nodes())}
        self.M = graph.size(weight='weight')
        self.gamma = gamma
    
    def leiden(self, iterations: int = 5) -> tuple[list[set[Any]], dict[Any, int]]:
        for _ in range(iterations):
            self.single_leiden()
        return self.partition, self.communities

    def single_leiden(self) -> None:
        prev_partition_size = len(self.partition) + 1
        while len(self.partition) < prev_partition_size:
            self.move_nodes_fast()
            prev_partition_size = len(self.partition)
        refined_partition = self.refine_partition()
        self.graph = self.aggregate_graph(refined_partition)
        self.partition = [set.union(*[self.partition[i] for i in self.graph.nodes() if self.partition[i].issubset(community)]) for community in self.partition]
        self.communities = {node: i for i, com in self.partition for node in com}

    def modularity_change(self, new_community: set[Any], node: Any) -> float:
        prev_community = self.partition[self.communities[node]]
        pre_node_adj = sum(self.graph[node][neighbour]['weight'] for neighbour in self.graph.neighbors(node) if neighbour in prev_community)
        new_node_adj = sum(self.graph[node][neighbour]['weight'] for neighbour in self.graph.neighbors(node) if neighbour in new_community)
        k_node = self.degrees[node]
        k_new= sum(self.degrees[n] for n in new_community)
        k_prev= sum(self.degrees[n] for n in prev_community)

        return (pre_node_adj -new_node_adj) - self.gamma * (k_node * (k_new - k_prev)) / (2 * self.M)
        
    def move_nodes_fast(self) -> None:
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)
        moved = True
        iteration = 0
        while moved and iteration < len(nodes):
            iteration += 1
            moved = False
            for node in nodes:
                cur_com = self.communities[node]
                best_com = cur_com
                best_gain = 0.0
                neighbour_coms = {self.communities[n] for n in self.graph.neighbors(node)}
                for comm_idx in neighbour_coms:
                    if comm_idx == cur_com:
                        continue
                    gain = self.modularity_change(self.partition[comm_idx], node)
                    print(f"Node {node}, Comm {comm_idx}, Gain {gain}")
                    if gain > best_gain:
                        best_gain = gain
                        best_com = comm_idx
                if best_gain > 1e-10 and best_com != cur_com:
                    print(f"Moving {node} from {cur_com} to {best_com} with gain {best_gain}")
                    self.partition[cur_com].remove(node)
                    if not self.partition[cur_com]:
                        del self.partition[cur_com]
                    self.partition[best_com].add(node)
                    self.communities[node] = best_com
                    moved = True
            if moved:
                self.partition = [com for com in self.partition if com]
                self.communities = {node: i for i, comm in enumerate(self.partition) for node in comm}
        self.partition = [com for com in self.partition if com]
        self.communities = {node: i for i, com in enumerate(self.partition) for node in com}
    
    def refine_partition(self) -> list[set[Any]]:
        refined = []
        for community in self.partition:
            subgraph = self.graph.subgraph(community)
            subpartition = [{n} for n in subgraph.nodes()]
            for node in subgraph.nodes():
                best_com = subpartition[node]
                best_gain = 0.0
                for i, candidate in enumerate(subpartition):
                    gain = self.modularity_change(candidate, node)
                    if gain > best_gain:
                        best_gain = gain
                        best_com = candidate
                        self.communities[node] = i
                if best_gain > 0.0:
                    best_com.add(node)
            refined.extend([c for c in subpartition if c])
        return refined
    
    def aggregate_graph(self, partition: list[set[Any]]) -> nx.Graph:
        graph = nx.Graph()
        for i, community in enumerate(self.partition):
            total_degree = sum(self.degrees[n] for n in community)
            graph.add_node(i, weight=total_degree)
        for i in range(len(partition)):
            for j in range(i+1, len(self.partition)):
                weight = sum(self.graph[u][v].get('weight', 1) for u in partition[i] for v in partition[j] if graph.has_edge(u, v))
                if weight > 0:
                    graph.add_edge(i, j)
        return graph
                    
    def get_modularity(self) -> float:
        mod = 0.0
        for u, v, attr in self.graph.edges(data=True):
            weight_uv = attr.get("weight", 1)
            if self.communities[u] == self.communities[v]:
                mod += (weight_uv - self.gamma * (self.degrees[u] * self.degrees[v]) / (2 * self.M))
        return mod / (2 * self.M)
        
G = nx.karate_club_graph()
leiden = Leiden(G, gamma=1.0)
partition, communities = leiden.leiden(iterations=5)
print(f"Final partition size: {len(partition)}")
print(f"Modularity: {leiden.get_modularity()}")