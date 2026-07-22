"""
main.py
-------
Task 2: Graph Algorithms and Pathfinding - single entry point.

Run this to execute everything in sequence:
    1. Build a sample transportation network graph
    2. Run Dijkstra, Prim, and Bellman-Ford on it and print results
    3. Run the full empirical benchmark + constant-factor analysis
    4. Generate the visualisation images

    python3 main.py
"""

from graph import generate_random_graph, Graph
from dijkstra import dijkstra, reconstruct_path
from prim import prim_mst
from bellman_ford import bellman_ford


def demo_small_example():
    print("=" * 70)
    print("STEP 1: Small worked example (8-city network)")
    print("=" * 70)
    g = generate_random_graph(8, edge_density=0.3, seed=42)
    print(g)

    source = "C0"

    dist, prev, _ = dijkstra(g, source)
    print(f"\nDijkstra shortest distances from {source}:")
    for v in sorted(dist):
        path = reconstruct_path(prev, source, v)
        print(f"  {v}: dist={dist[v]}  path={path}")

    mst_edges, total_weight, _ = prim_mst(g)
    print(f"\nPrim's MST total weight: {total_weight}")
    print("MST edges:", mst_edges)

    dist_bf, prev_bf, neg_cycle, _ = bellman_ford(g, source)
    print(f"\nBellman-Ford from {source} (negative cycle: {neg_cycle}):")
    for v in sorted(dist_bf):
        print(f"  {v}: dist={dist_bf[v]}")

    # Negative-weight demo
    print("\n--- Negative weight handling demo ---")
    g_neg = generate_random_graph(8, edge_density=0.3, allow_negative=True, seed=99)
    dist_neg, _, neg_cycle2, _ = bellman_ford(g_neg, source)
    print(f"Bellman-Ford on graph with negative edges, negative cycle detected: {neg_cycle2}")
    print("Distances:", dist_neg)


if __name__ == "__main__":
    demo_small_example()

    print("\n" + "=" * 70)
    print("STEP 2: Full empirical benchmark (see compare_analysis.py for details)")
    print("=" * 70)
    import compare_analysis
    compare_analysis.run()

    print("\n" + "=" * 70)
    print("STEP 3: Generating visualisations (see visualize.py)")
    print("=" * 70)
    import visualize
    visualize.run()
