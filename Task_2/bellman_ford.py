"""
bellman_ford.py
----------------
Task 2 (b): Bellman-Ford algorithm.

Unlike Dijkstra's, Bellman-Ford correctly handles NEGATIVE edge weights,
and can DETECT negative-weight cycles (in which case "shortest path" is
undefined, since you could loop forever to reduce cost further).

Complexity:
    Time  : O(V * E)   -- we relax every edge, up to (V-1) times, then
                           do one more pass to detect negative cycles
    Space : O(V + E)
"""

from graph import Graph


def bellman_ford(graph: Graph, source, track_steps=False):
    """
    Returns (dist, prev, has_negative_cycle, steps)
        dist               : dict vertex -> shortest distance from source
        prev               : dict vertex -> predecessor
        has_negative_cycle : True if a negative-weight cycle reachable
                              from source was detected
        steps              : list of (iteration, num_updates) for analysis
    """
    dist = {v: float('inf') for v in graph.vertices}
    prev = {v: None for v in graph.vertices}
    dist[source] = 0

    edge_list = list(graph.edges())
    steps = []
    V = graph.num_vertices()

    # Relax all edges V-1 times
    for i in range(V - 1):
        updated = 0
        for u, v, w in edge_list:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated += 1
        if track_steps:
            steps.append((i + 1, updated))
        if updated == 0:
            # Early exit: no changes means we have converged
            break

    # One more pass: if anything still improves, there's a negative cycle
    has_negative_cycle = False
    for u, v, w in edge_list:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle, steps


if __name__ == "__main__":
    from graph import Graph

    # Example WITHOUT a negative cycle
    g = Graph(directed=True)
    g.add_edge("A", "B", 4)
    g.add_edge("A", "C", 5)
    g.add_edge("B", "C", -3)
    g.add_edge("C", "D", 4)
    g.add_edge("B", "D", 6)

    dist, prev, neg_cycle, steps = bellman_ford(g, "A", track_steps=True)
    print("Graph WITHOUT negative cycle:")
    print("  Distances:", dist)
    print("  Negative cycle detected:", neg_cycle)
    print("  Convergence steps:", steps)

    # Example WITH a negative cycle
    g2 = Graph(directed=True)
    g2.add_edge("A", "B", 1)
    g2.add_edge("B", "C", -1)
    g2.add_edge("C", "A", -1)  # A->B->C->A = 1-1-1 = -1 < 0 : negative cycle

    dist2, prev2, neg_cycle2, steps2 = bellman_ford(g2, "A", track_steps=True)
    print("\nGraph WITH negative cycle:")
    print("  Negative cycle detected:", neg_cycle2)
