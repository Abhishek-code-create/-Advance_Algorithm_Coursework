"""
dijkstra.py
-----------
Task 2 (b): Dijkstra's algorithm for single-source shortest paths.

Requires NON-NEGATIVE edge weights.

Complexity (binary heap implementation, as used here):
    Time  : O((V + E) log V)   -- each vertex is popped once O(log V),
                                   each edge may cause a decrease-key /
                                   push O(log V)
    Space : O(V + E)            -- adjacency list + distance/heap arrays

    (With a Fibonacci heap this improves to O(E + V log V), but Python's
    heapq only gives a binary heap, which is what we analyse empirically.)
"""

import heapq
from graph import Graph


def dijkstra(graph: Graph, source, track_steps=False):
    """
    Returns (dist, prev, steps)
        dist  : dict vertex -> shortest distance from source
        prev  : dict vertex -> predecessor on shortest path
        steps : list of (vertex_finalised, dist_at_that_point) if track_steps
    """
    dist = {v: float('inf') for v in graph.vertices}
    prev = {v: None for v in graph.vertices}
    dist[source] = 0

    pq = [(0, source)]
    visited = set()
    steps = []

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if track_steps:
            steps.append((u, d))

        for v, w in graph.neighbours(u):
            if w < 0:
                raise ValueError("Dijkstra's algorithm requires non-negative weights.")
            if v in visited:
                continue
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev, steps


def reconstruct_path(prev, source, target):
    """Rebuild the shortest path from source to target using prev pointers."""
    if prev.get(target) is None and target != source:
        return None  # unreachable
    path = []
    node = target
    while node is not None:
        path.append(node)
        if node == source:
            break
        node = prev[node]
    path.reverse()
    return path if path and path[0] == source else None


if __name__ == "__main__":
    from graph import generate_random_graph

    g = generate_random_graph(8, edge_density=0.3, seed=42)
    dist, prev, steps = dijkstra(g, "C0", track_steps=True)

    print("Shortest distances from C0:")
    for v in sorted(dist):
        print(f"  {v}: {dist[v]}")

    print("\nOrder in which vertices were finalised (step-by-step):")
    for v, d in steps:
        print(f"  finalise {v} at distance {d}")

    print("\nExample path C0 -> C6:", reconstruct_path(prev, "C0", "C6"))
