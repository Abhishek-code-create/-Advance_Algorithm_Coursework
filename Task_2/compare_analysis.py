"""
compare_analysis.py
--------------------
Task 2 (c): Empirical comparison of Dijkstra, Prim, and Bellman-Ford.

Produces:
    1. A printed complexity comparison table.
    2. Empirical wall-clock timing on sparse and dense random graphs of
       increasing size.
    3. A matplotlib chart: runtime vs number of vertices for each algorithm.
    4. Observed "constant factor" estimate: actual_time / theoretical_ops,
       so we don't claim Big-O directly predicts wall-clock time.

Run: python3 compare_analysis.py
Outputs: runtime_comparison.png, saved in the same directory.
"""

import time
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph import generate_random_graph
from dijkstra import dijkstra
from prim import prim_mst
from bellman_ford import bellman_ford


COMPLEXITY_TABLE = """
Algorithm      | Time Complexity   | Space  | Handles Neg. Weights | Best For
---------------|-------------------|--------|-----------------------|------------------
Dijkstra       | O((V+E) log V)    | O(V+E) | No                    | Sparse/dense, non-negative weights
Prim (MST)     | O(E log V)        | O(V+E) | N/A (undirected MST)  | Sparse/dense connected graphs
Bellman-Ford   | O(V * E)          | O(V+E) | Yes (+cycle detection)| Graphs with negative weights
"""


def time_algorithm(fn, *args, repeats=3, **kwargs):
    """Run fn `repeats` times and return the minimum wall-clock time (seconds)."""
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best


def run_benchmark(sizes, density, label):
    results = {"dijkstra": [], "prim": [], "bellman_ford": []}

    for n in sizes:
        g = generate_random_graph(n, edge_density=density, seed=1)
        source = next(iter(g.vertices))

        t_dij = time_algorithm(dijkstra, g, source)
        t_prim = time_algorithm(prim_mst, g, source)
        t_bf = time_algorithm(bellman_ford, g, source)

        results["dijkstra"].append(t_dij)
        results["prim"].append(t_prim)
        results["bellman_ford"].append(t_bf)

        e = g.num_edges()
        print(f"[{label}] n={n:>6} E={e:>7} | "
              f"Dijkstra={t_dij:.6f}s  Prim={t_prim:.6f}s  Bellman-Ford={t_bf:.6f}s")

    return results


def estimate_constant_factor(sizes, results, big_o_fn, label):
    """
    Estimate the hidden constant c in T(n) ~= c * f(n) using the LARGEST
    data point, then show what that constant predicts for smaller n --
    demonstrating why raw Big-O alone doesn't predict wall-clock time.
    """
    n_last = sizes[-1]
    t_last = results[label][-1]
    f_last = big_o_fn(n_last)
    c = t_last / f_last if f_last > 0 else 0
    print(f"\n[{label}] estimated constant factor c = {c:.3e} "
          f"(from n={n_last}, T={t_last:.6f}s, f(n)={f_last:.1f})")
    print(f"  Predicted vs actual using this constant:")
    for n, t_actual in zip(sizes, results[label]):
        predicted = c * big_o_fn(n)
        print(f"    n={n:>6}: predicted={predicted:.6f}s  actual={t_actual:.6f}s")


def plot_results(sizes, sparse_results, dense_results, filename="runtime_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, marker in [("dijkstra", "o"), ("prim", "s"), ("bellman_ford", "^")]:
        axes[0].plot(sizes, sparse_results[label], marker=marker, label=label)
        axes[1].plot(sizes, dense_results[label], marker=marker, label=label)

    axes[0].set_title("Sparse graphs (density=0.05)")
    axes[1].set_title("Dense graphs (density=0.4)")
    for ax in axes:
        ax.set_xlabel("Number of vertices (V)")
        ax.set_ylabel("Wall-clock time (seconds)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Empirical Runtime: Dijkstra vs Prim vs Bellman-Ford")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"\nSaved chart to {filename}")


def run(sizes=None):
    """Run the full comparison + benchmark + plotting pipeline."""
    if sizes is None:
        sizes = [50, 100, 250, 500, 1000]

    print("=" * 70)
    print("THEORETICAL COMPLEXITY COMPARISON")
    print("=" * 70)
    print(COMPLEXITY_TABLE)

    print("=" * 70)
    print("EMPIRICAL BENCHMARK: SPARSE GRAPHS")
    print("=" * 70)
    sparse_results = run_benchmark(sizes, density=0.05, label="sparse")

    print("\n" + "=" * 70)
    print("EMPIRICAL BENCHMARK: DENSE GRAPHS")
    print("=" * 70)
    dense_results = run_benchmark(sizes, density=0.4, label="dense")

    print("\n" + "=" * 70)
    print("HIDDEN CONSTANT FACTOR ANALYSIS (dense graphs)")
    print("=" * 70)
    estimate_constant_factor(
        sizes, dense_results,
        lambda n: (n + n * n * 0.4) * math.log2(max(n, 2)), "dijkstra")
    estimate_constant_factor(
        sizes, dense_results,
        lambda n: n * n, "bellman_ford")

    plot_results(sizes, sparse_results, dense_results)
    return sparse_results, dense_results


if __name__ == "__main__":
    run()
