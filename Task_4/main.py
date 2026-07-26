"""
main.py
-------
Entry point for Task 4: NP-Hard Problem and Heuristics.

Problem chosen: MULTI-DIMENSIONAL BIN PACKING.
Items each carry demand across several resource dimensions (CPU, RAM,
bandwidth); they must be packed into the minimum number of fixed-
capacity bins without exceeding any dimension's capacity in any bin.

Why is this problem NP-Hard?
-----------------------------
Even the classic 1-dimensional Bin Packing Problem is NP-Hard: it can
be shown by reduction from PARTITION (an NP-Complete problem that
asks whether a multiset of numbers can be split into two subsets with
equal sum). Given a PARTITION instance, set the bin capacity to half
the total sum of all numbers and ask "can these items be packed into
2 bins?" -- a "yes" answer to that Bin Packing question is exactly a
"yes" answer to PARTITION, and vice versa. Since PARTITION is
NP-Complete, 1D Bin Packing is NP-Hard.

Multi-dimensional Bin Packing is a generalisation of 1D Bin Packing
(recovered exactly by setting all but one resource dimension's
demand to zero for every item), so it is at least as hard as the 1D
case and remains NP-Hard. In practice it is harder still: a packing
must satisfy several simultaneous capacity constraints per bin, which
shrinks the feasible search space and makes simple constructive
heuristics (like First-Fit-Decreasing) less effective than in the
1D case.

Run this file to:
  1. Generate a sample problem instance.
  2. Solve it with the Greedy (First-Fit-Decreasing) heuristic.
  3. Improve the greedy solution with Local Search (hill climbing).
  4. Improve the greedy solution with Simulated Annealing.
  5. Print a full comparison table and save a runtime/quality chart.
"""

from items import generate_random_instance
from greedy_heuristic import greedy_first_fit_decreasing
from local_search_heuristic import local_search
from simulated_annealing_heuristic import simulated_annealing
from evaluation import run_comparison, print_table, plot_results


def demo_single_instance(n_items: int = 100):
    print(f"\n=== Multi-dimensional Bin Packing: instance with {n_items} items ===\n")
    items = generate_random_instance(n_items)

    greedy_bins = greedy_first_fit_decreasing(items)
    print(f"Greedy (FFD):        {len(greedy_bins)} bins")

    ls_bins = local_search(greedy_bins)
    print(f"Local Search:        {len(ls_bins)} bins")

    sa_bins = simulated_annealing(greedy_bins)
    print(f"Simulated Annealing: {len(sa_bins)} bins")


if __name__ == "__main__":
    demo_single_instance(100)

    print("\n=== Full comparison across instance sizes ===\n")
    results = run_comparison(sizes=(50, 100, 200, 500))
    print_table(results)
    plot_results(results, out_path="task4_comparison.png")
