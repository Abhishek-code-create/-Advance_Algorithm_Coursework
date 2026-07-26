"""
local_search_heuristic.py
--------------------------
Local search improvement heuristic for Multi-dimensional Bin Packing,
based on BIN CONSOLIDATION: the most effective and standard local
search move for bin-packing-style problems.

Starting from a greedy (FFD) solution, the algorithm repeatedly:
  1. Orders bins from least-utilised to most-utilised.
  2. Attempts to fully empty the least-utilised bin by relocating
     every one of its items into some other bin that has room.
  3. If ALL items of that bin can be relocated, the bin is
     eliminated (bin count decreases by 1) and the process repeats
     from a re-sorted bin list.
  4. If the least-utilised bin cannot be fully emptied, the next
     least-utilised bin is tried, and so on.
The search stops when a full pass eliminates no bin (a local
optimum under this move) or a pass limit is reached.

This directly targets the primary objective (minimise bin count)
rather than relying on random single-item swaps, which rarely help
once a greedy construction has already packed bins tightly.

Complexity per pass: for each of up to n_bins candidate "victim"
bins, relocating its items costs O(items_in_bin * n_bins) fits
checks, so a full pass is O(n_bins^2) in the worst case. The number
of passes is bounded by the number of bins that can be eliminated
(<= n_bins), giving an O(n_bins^3) worst case -- still far cheaper in
practice because most passes eliminate a bin quickly or stop early.
"""

import time
from typing import List, Dict
from items import (
    Item, DIMENSIONS, BIN_CAPACITY, fits,
    generate_random_instance, bin_utilization,
)
from greedy_heuristic import greedy_first_fit_decreasing


def _bin_load(items_in_bin: List[Item]) -> Dict[str, float]:
    load = {d: 0.0 for d in DIMENSIONS}
    for it in items_in_bin:
        for d in DIMENSIONS:
            load[d] += it.demand[d]
    return load


def _try_consolidate_bin(bins: List[List[Item]], idx: int, capacity) -> bool:
    """Try to empty bins[idx] by relocating each of its items into
    another bin. Returns True and commits the change if ALL items
    could be relocated; otherwise returns False and leaves bins
    unchanged."""
    target_items = list(bins[idx])
    other_indices = [i for i in range(len(bins)) if i != idx]

    # Trial copies so we only commit if the WHOLE bin can be emptied
    trial_bins = {i: list(bins[i]) for i in other_indices}
    trial_loads = {i: _bin_load(trial_bins[i]) for i in other_indices}

    for item in target_items:
        placed = False
        for i in other_indices:
            if fits(trial_loads[i], item, capacity):
                trial_bins[i].append(item)
                for d in DIMENSIONS:
                    trial_loads[i][d] += item.demand[d]
                placed = True
                break
        if not placed:
            return False  # cannot fully empty this bin -- abandon attempt

    # Success: commit the trial placement
    for i in other_indices:
        bins[i] = trial_bins[i]
    bins[idx] = []
    return True


def local_search(bins: List[List[Item]], capacity=BIN_CAPACITY,
                  max_passes: int = 100) -> List[List[Item]]:
    bins = [list(b) for b in bins]
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        order = sorted(range(len(bins)), key=lambda i: sum(_bin_load(bins[i]).values()))
        for idx in order:
            if not bins[idx]:
                continue
            if _try_consolidate_bin(bins, idx, capacity):
                improved = True
                break  # bin layout changed -- restart from a fresh ordering
        bins = [b for b in bins if b]

    return bins


if __name__ == "__main__":
    print("Local Search (bin consolidation) -- standalone test\n")
    for n in (50, 200, 1000):
        items = generate_random_instance(n)
        greedy_bins = greedy_first_fit_decreasing(items)

        start = time.perf_counter()
        improved_bins = local_search(greedy_bins)
        elapsed = time.perf_counter() - start

        print(f"n={n:5d}  greedy_bins={len(greedy_bins):4d}  "
              f"local_search_bins={len(improved_bins):4d}  time={elapsed:.4f}s")
