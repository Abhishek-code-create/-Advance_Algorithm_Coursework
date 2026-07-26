"""
evaluation.py
--------------
Compares the Greedy, Local Search, and Simulated Annealing heuristics
for the Multi-dimensional Bin Packing Problem across several instance
sizes. Reports solution quality (bins used, average utilisation) and
runtime, then saves a comparison chart (bins used vs. n, runtime vs. n).
"""

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from items import generate_random_instance, average_utilization, validate_solution
from greedy_heuristic import greedy_first_fit_decreasing
from local_search_heuristic import local_search
from simulated_annealing_heuristic import simulated_annealing


def run_comparison(sizes=(50, 100, 200, 500)):
    results = []
    for n in sizes:
        items = generate_random_instance(n)

        t0 = time.perf_counter()
        greedy_bins = greedy_first_fit_decreasing(items)
        t_greedy = time.perf_counter() - t0

        t0 = time.perf_counter()
        ls_bins = local_search(greedy_bins)
        t_ls = time.perf_counter() - t0

        t0 = time.perf_counter()
        sa_bins = simulated_annealing(greedy_bins)
        t_sa = time.perf_counter() - t0

        # sanity checks -- every solution must remain feasible & complete
        assert validate_solution(greedy_bins, items), "Greedy solution invalid"
        assert validate_solution(ls_bins, items), "Local search solution invalid"
        assert validate_solution(sa_bins, items), "SA solution invalid"

        results.append({
            "n": n,
            "greedy_bins": len(greedy_bins), "greedy_time": t_greedy,
            "greedy_util": average_utilization(greedy_bins),
            "ls_bins": len(ls_bins), "ls_time": t_ls,
            "ls_util": average_utilization(ls_bins),
            "sa_bins": len(sa_bins), "sa_time": t_sa,
            "sa_util": average_utilization(sa_bins),
        })
    return results


def print_table(results):
    header = f"{'n':>5} | {'Greedy':>22} | {'LocalSearch':>22} | {'SimAnneal':>22}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['n']:>5} | "
              f"bins={r['greedy_bins']:>3} util={r['greedy_util']:.2f} t={r['greedy_time']:.3f}s | "
              f"bins={r['ls_bins']:>3} util={r['ls_util']:.2f} t={r['ls_time']:.3f}s | "
              f"bins={r['sa_bins']:>3} util={r['sa_util']:.2f} t={r['sa_time']:.3f}s")


def plot_results(results, out_path="comparison.png"):
    sizes = [r["n"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sizes, [r["greedy_bins"] for r in results], "o-", label="Greedy")
    axes[0].plot(sizes, [r["ls_bins"] for r in results], "s-", label="Local Search")
    axes[0].plot(sizes, [r["sa_bins"] for r in results], "^-", label="Simulated Annealing")
    axes[0].set_xlabel("Number of items")
    axes[0].set_ylabel("Bins used")
    axes[0].set_title("Solution quality (fewer bins is better)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(sizes, [r["greedy_time"] for r in results], "o-", label="Greedy")
    axes[1].plot(sizes, [r["ls_time"] for r in results], "s-", label="Local Search")
    axes[1].plot(sizes, [r["sa_time"] for r in results], "^-", label="Simulated Annealing")
    axes[1].set_xlabel("Number of items")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Runtime comparison")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    results = run_comparison()
    print_table(results)
    plot_results(results)
