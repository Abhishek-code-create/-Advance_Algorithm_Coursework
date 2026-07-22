"""
prim.py
-------
Task 2 (b): Prim's algorithm for Minimum Spanning Tree (MST) construction.

Prim's is defined on an UNDIRECTED graph. If given a directed graph we
first convert it via graph.to_undirected() (each directed edge treated
as an undirected connection; if both (u,v) and (v,u) exist with
different weights, we effectively keep the first one encountered --
documented limitation for this coursework's simplified model).

Complexity (binary heap / lazy deletion implementation, as used here):
    Time  : O(E log V)   -- every edge may be pushed to the heap once,
                             each push/pop is O(log V)  (E >= V-1 for a
                             connected graph, so this dominates over
                             O(V log V))
    Space : O(V + E)
"""

import heapq
from graph import Graph


def prim_mst(graph: Graph, source=None, track_steps=False):
    """
    Returns (mst_edges, total_weight, steps)
        mst_edges   : list of (u, v, w) edges in the MST
        total_weight: sum of MST edge weights
        steps       : list of (u, v, w) in the order edges were added
    """
    if graph.directed:
        graph = graph.to_undirected()

    if not graph.vertices:
        return [], 0, []

    if source is None:
        source = next(iter(graph.vertices))

    visited = {source}
    mst_edges = []
    steps = []
    total_weight = 0

    # heap entries: (weight, u, v)
    edge_heap = [(w, source, v) for v, w in graph.neighbours(source)]
    heapq.heapify(edge_heap)

    while edge_heap and len(visited) < graph.num_vertices():
        w, u, v = heapq.heappop(edge_heap)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_weight += w
        if track_steps:
            steps.append((u, v, w))

        for nxt, w2 in graph.neighbours(v):
            if nxt not in visited:
                heapq.heappush(edge_heap, (w2, v, nxt))

    connected = len(visited) == graph.num_vertices()
    if not connected and track_steps:
        steps.append(("WARNING", "graph not fully connected - MST covers only reachable component", None))

    return mst_edges, total_weight, steps


if __name__ == "__main__":
    from graph import generate_random_graph

    g = generate_random_graph(8, edge_density=0.3, seed=42)
    mst_edges, total_weight, steps = prim_mst(g, track_steps=True)

    print("MST edges (step-by-step order added):")
    for u, v, w in steps:
        print(f"  add {u} -- {v} (w={w})")

    print(f"\nTotal MST weight: {total_weight}")
