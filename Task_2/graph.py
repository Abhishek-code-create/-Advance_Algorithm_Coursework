"""
graph.py
--------
Task 2 (a): Graph representation for a city transportation network.

We model the network as a WEIGHTED DIRECTED GRAPH using an ADJACENCY LIST.

Why adjacency list over adjacency matrix?
    - Real transportation networks are SPARSE (a city connects to only a
      handful of neighbouring cities, not all of them).
    - Adjacency list uses O(V + E) space vs O(V^2) for a matrix.
    - Edge iteration (used by Dijkstra/Prim/Bellman-Ford) is O(degree(v))
      instead of O(V) per vertex.
    - Adjacency matrix would only be preferable if the graph were dense
      (E close to V^2) or if O(1) edge-existence lookups were the priority.

Complexity summary (V = vertices, E = edges):
    Adjacency List : space O(V+E), add edge O(1), check edge O(degree(v))
    Adjacency Matrix: space O(V^2), add edge O(1), check edge O(1)
"""

import random
from collections import defaultdict


class Graph:
    """Weighted directed graph stored as an adjacency list."""

    def __init__(self, directed=True):
        self.directed = directed
        self.adj = defaultdict(list)   # vertex -> list of (neighbour, weight)
        self.vertices = set()

    def add_vertex(self, v):
        self.vertices.add(v)
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v, weight):
        """Add edge u -> v with given weight. O(1)."""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def neighbours(self, u):
        return self.adj[u]

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return sum(len(edges) for edges in self.adj.values())

    def edges(self):
        """Yield (u, v, weight) for every edge."""
        seen = set()
        for u in self.adj:
            for v, w in self.adj[u]:
                if self.directed:
                    yield (u, v, w)
                else:
                    key = tuple(sorted((u, v)))
                    if key not in seen:
                        seen.add(key)
                        yield (u, v, w)

    def to_undirected(self):
        """Return an undirected copy (needed for Prim's MST)."""
        g = Graph(directed=False)
        for u, v, w in self.edges():
            g.add_edge(u, v, w)
        return g

    def __repr__(self):
        return f"Graph(V={self.num_vertices()}, E={self.num_edges()}, directed={self.directed})"


def generate_random_graph(n_vertices, edge_density=0.15, min_w=1, max_w=20,
                           allow_negative=False, seed=None):
    """
    Generate a random weighted directed graph for empirical testing.

    n_vertices    : number of city nodes
    edge_density  : fraction of possible directed edges to create (sparse ~0.05-0.15)
    allow_negative: if True, ~10% of edges get a negative weight (for Bellman-Ford tests)
    """
    rng = random.Random(seed)
    g = Graph(directed=True)
    nodes = [f"C{i}" for i in range(n_vertices)]
    for node in nodes:
        g.add_vertex(node)

    max_possible_edges = n_vertices * (n_vertices - 1)
    target_edges = int(max_possible_edges * edge_density)

    created = set()
    attempts = 0
    while len(created) < target_edges and attempts < target_edges * 20:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        if (u, v) in created:
            continue
        w = rng.randint(min_w, max_w)
        if allow_negative and rng.random() < 0.1:
            w = -rng.randint(1, max_w // 4 + 1)
        g.add_edge(u, v, w)
        created.add((u, v))

    # Ensure graph is weakly connected by chaining nodes together
    for i in range(len(nodes) - 1):
        if not any(v == nodes[i + 1] for v, _ in g.adj[nodes[i]]):
            g.add_edge(nodes[i], nodes[i + 1], rng.randint(min_w, max_w))

    return g


if __name__ == "__main__":
    g = generate_random_graph(8, edge_density=0.3, seed=42)
    print(g)
    for u, v, w in g.edges():
        print(f"  {u} -> {v} (w={w})")
