"""
simulated_annealing_heuristic.py
----------------------------------
Simulated Annealing (SA) heuristic for Multi-dimensional Bin Packing.

Starts from a greedy solution. At each step a random neighbouring
move (relocate or swap) is generated. Moves that improve the
objective are always accepted; moves that worsen it are accepted
with probability exp(-delta / temperature). Temperature is reduced
geometrically each iteration (cooling schedule), so the algorithm
behaves more like random search early on -- helping it escape local
optima that trap plain hill climbing -- and more like greedy descent
towards the end.

Complexity: each iteration does O(1) work to generate and evaluate a
neighbour (bounded by items per bin), and the algorithm runs for a
fixed max_iterations, so overall cost is O(max_iterations) --
independent of n in the neighbour-generation step, though larger n
means more items per bin to search through when relocating/swapping.
"""

import math
import random
import time
from typing import List
from items import (
    Item, DIMENSIONS, BIN_CAPACITY, fits,
    generate_random_instance, bin_utilization,
)
from greedy_heuristic import greedy_first_fit_decreasing


def _bin_load(b: List[Item]) -> dict:
    return {d: sum(it.demand[d] for it in b) for d in DIMENSIONS}


def _objective_scalar(bins: List[List[Item]], w_bins: float = 1000.0, w_var: float = 1.0) -> float:
    """Single scalar cost for SA: heavily penalise extra bins, lightly
    penalise utilisation imbalance."""
    non_empty = [b for b in bins if b]
    n_bins = len(non_empty)
    utils = bin_utilization(non_empty) if non_empty else [0.0]
    mean_u = sum(utils) / len(utils)
    variance = sum((u - mean_u) ** 2 for u in utils) / len(utils)
    return w_bins * n_bins + w_var * variance


def _try_empty_bin(bins: List[List[Item]], idx: int, capacity):
    """Attempt to fully empty bins[idx] by relocating every item it
    holds into some other bin. Returns a new bins list on success
    (this is the single move type most likely to reduce bin count),
    or None if the bin cannot be fully emptied."""
    other_indices = [i for i in range(len(bins)) if i != idx]
    trial = {i: list(bins[i]) for i in other_indices}
    loads = {i: _bin_load(trial[i]) for i in other_indices}

    for item in bins[idx]:
        placed = False
        for i in other_indices:
            if fits(loads[i], item, capacity):
                trial[i].append(item)
                for d in DIMENSIONS:
                    loads[i][d] += item.demand[d]
                placed = True
                break
        if not placed:
            return None

    new_bins = [trial[i] for i in other_indices]
    return [b for b in new_bins if b]


def _random_neighbour(bins: List[List[Item]], capacity, rng: random.Random):
    """Return a new bins configuration one random feasible move away,
    or None if no feasible move could be generated this attempt.

    Three move types are tried: relocate and swap explore fine-grained
    rebalancing, while consolidate specifically targets the primary
    objective (fewer bins) by trying to fully empty the least-loaded
    bin -- single relocate/swap moves almost never achieve this
    directly, since a bin usually holds more than one item."""
    bins = [list(b) for b in bins]
    non_empty_idx = [i for i, b in enumerate(bins) if b]
    if len(non_empty_idx) < 2:
        return None

    move_type = rng.choice(["relocate", "swap", "consolidate"])

    if move_type == "consolidate":
        target = min(non_empty_idx, key=lambda i: sum(_bin_load(bins[i]).values()))
        return _try_empty_bin(bins, target, capacity)

    i = rng.choice(non_empty_idx)

    if move_type == "relocate":
        item = rng.choice(bins[i])
        candidates = [j for j in range(len(bins)) if j != i]
        rng.shuffle(candidates)
        for j in candidates:
            if fits(_bin_load(bins[j]), item, capacity):
                bins[i].remove(item)
                bins[j].append(item)
                return [b for b in bins if b]
        return None

    # swap
    candidates = [j for j in non_empty_idx if j != i]
    if not candidates:
        return None
    j = rng.choice(candidates)
    item_i = rng.choice(bins[i])
    item_j = rng.choice(bins[j])
    load_i_after = {d: _bin_load(bins[i])[d] - item_i.demand[d] for d in DIMENSIONS}
    load_j_after = {d: _bin_load(bins[j])[d] - item_j.demand[d] for d in DIMENSIONS}
    if fits(load_i_after, item_j, capacity) and fits(load_j_after, item_i, capacity):
        bins[i].remove(item_i)
        bins[j].remove(item_j)
        bins[i].append(item_j)
        bins[j].append(item_i)
        return [b for b in bins if b]
    return None


def simulated_annealing(bins: List[List[Item]], capacity=BIN_CAPACITY,
                         initial_temp: float = 100.0, cooling_rate: float = 0.995,
                         max_iterations: int = 3000, seed: int = 1) -> List[List[Item]]:
    rng = random.Random(seed)
    current = [list(b) for b in bins]
    current_cost = _objective_scalar(current)
    best = current
    best_cost = current_cost
    temp = initial_temp

    for _ in range(max_iterations):
        neighbour = _random_neighbour(current, capacity, rng)
        if neighbour is None:
            temp *= cooling_rate
            continue

        neighbour_cost = _objective_scalar(neighbour)
        delta = neighbour_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
            current = neighbour
            current_cost = neighbour_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        temp *= cooling_rate
        if temp < 1e-3:
            break

    return best


if __name__ == "__main__":
    print("Simulated Annealing -- standalone test\n")
    for n in (50, 200, 500):
        items = generate_random_instance(n)
        greedy_bins = greedy_first_fit_decreasing(items)

        start = time.perf_counter()
        sa_bins = simulated_annealing(greedy_bins)
        elapsed = time.perf_counter() - start

        print(f"n={n:4d}  greedy_bins={len(greedy_bins):4d}  "
              f"sa_bins={len(sa_bins):4d}  time={elapsed:.4f}s")
