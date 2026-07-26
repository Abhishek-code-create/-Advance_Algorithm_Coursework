"""
items.py
--------
Data model for the Multi-dimensional Bin Packing Problem (MDBPP).

Each item requires a fixed amount of several resources (e.g. CPU, RAM,
bandwidth). Each bin has a fixed capacity for every resource dimension.
The goal is to pack all items into the minimum number of bins such
that, in every bin, the sum of each resource dimension across all
items placed in it does not exceed that bin's capacity.
"""

import random
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Item:
    item_id: int
    demand: Dict[str, float]  # e.g. {"cpu": 40, "ram": 30, "bw": 20}

    def __repr__(self):
        return f"Item({self.item_id}, {self.demand})"


DIMENSIONS = ["cpu", "ram", "bw"]
BIN_CAPACITY = {"cpu": 100.0, "ram": 100.0, "bw": 100.0}


def generate_random_instance(n_items: int, seed: int = 42) -> List[Item]:
    """Generate a random MDBPP instance with n_items items."""
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        demand = {dim: rng.uniform(10, 60) for dim in DIMENSIONS}
        items.append(Item(item_id=i, demand=demand))
    return items


def fits(bin_load: Dict[str, float], item: Item,
         capacity: Dict[str, float] = BIN_CAPACITY) -> bool:
    """Check whether item fits into a bin with current bin_load."""
    return all(bin_load[d] + item.demand[d] <= capacity[d] for d in DIMENSIONS)


def bin_utilization(bins: List[List[Item]],
                     capacity: Dict[str, float] = BIN_CAPACITY) -> List[float]:
    """Average utilisation (fraction of capacity used, averaged across
    dimensions) for each bin."""
    utils = []
    for b in bins:
        load = {d: sum(it.demand[d] for it in b) for d in DIMENSIONS}
        util = sum(load[d] / capacity[d] for d in DIMENSIONS) / len(DIMENSIONS)
        utils.append(util)
    return utils


def average_utilization(bins: List[List[Item]],
                         capacity: Dict[str, float] = BIN_CAPACITY) -> float:
    utils = bin_utilization(bins, capacity)
    return sum(utils) / len(utils) if utils else 0.0


def validate_solution(bins: List[List[Item]], all_items: List[Item],
                       capacity: Dict[str, float] = BIN_CAPACITY) -> bool:
    """Sanity check: every item packed exactly once, no bin over capacity."""
    packed_ids = [it.item_id for b in bins for it in b]
    if sorted(packed_ids) != sorted(it.item_id for it in all_items):
        return False
    for b in bins:
        load = {d: sum(it.demand[d] for it in b) for d in DIMENSIONS}
        if any(load[d] > capacity[d] + 1e-9 for d in DIMENSIONS):
            return False
    return True
