"""
greedy_heuristic.py
--------------------
Greedy construction heuristic for Multi-dimensional Bin Packing.

Strategy: First-Fit-Decreasing (FFD) generalised to multiple
dimensions. Items are sorted in decreasing order of total resource
demand (sum across all dimensions), then each item is placed in the
first open bin it fits into. If it fits nowhere, a new bin is opened.

Complexity: sorting is O(n log n). Placement checks each item against
at most all currently-open bins, so placement is O(n * b) where b is
the number of bins opened so far (b <= n), giving an O(n^2) worst
case overall. In practice b << n once bins start filling up, so the
constant factor is much smaller than the worst case suggests.
"""

import time
from typing import List
from items import Item, DIMENSIONS, BIN_CAPACITY, fits, generate_random_instance


def greedy_first_fit_decreasing(items: List[Item], capacity=BIN_CAPACITY) -> List[List[Item]]:
    sorted_items = sorted(items, key=lambda it: sum(it.demand.values()), reverse=True)

    bins: List[List[Item]] = []
    bin_loads = []  # parallel list of dicts tracking load per bin

    for item in sorted_items:
        placed = False
        for i, load in enumerate(bin_loads):
            if fits(load, item, capacity):
                bins[i].append(item)
                for d in DIMENSIONS:
                    bin_loads[i][d] += item.demand[d]
                placed = True
                break
        if not placed:
            bins.append([item])
            bin_loads.append({d: item.demand[d] for d in DIMENSIONS})

    return bins


if __name__ == "__main__":
    print("Greedy First-Fit-Decreasing -- standalone test\n")
    for n in (50, 200, 1000):
        items = generate_random_instance(n)
        start = time.perf_counter()
        bins = greedy_first_fit_decreasing(items)
        elapsed = time.perf_counter() - start
        print(f"n={n:5d}  bins_used={len(bins):4d}  time={elapsed:.4f}s")
