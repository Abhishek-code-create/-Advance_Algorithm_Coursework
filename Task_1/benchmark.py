"""
benchmark.py
------------
Empirical performance testing for Task 1 (Analysis - 10 marks).

Measures actual wall-clock time for insertion and search across:
    - BST            (random-order insertion AND sorted/worst-case insertion)
    - AVL Tree       (self-balancing)
    - Hash Table     (chaining)
    - Min-Heap       (push / pop, used for priority-queue style access)

at n = 100, 1,000, 10,000 nodes, then:
    1. Prints a results table (also saved as results_table.csv)
    2. Saves labelled comparison graphs as PNG files:
        - insertion_time.png
        - search_time.png
        - bst_vs_avl_worst_case.png

Run with:  python3 benchmark.py
Requires:  matplotlib  (pip install matplotlib --break-system-packages)
"""

import random
import sys
import time
import csv

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNG files
import matplotlib.pyplot as plt

from city import City
from bst import BST
from avl_tree import AVLTree
from hash_table import HashTableChaining
from min_heap import MinHeap


SIZES = [100, 1_000, 10_000]

# The plain BST is implemented recursively. When we deliberately feed it
# SORTED input (the worst case) to demonstrate O(n) degeneration, the tree
# becomes a linked list of depth n, so Python's default recursion limit
# (1000) is exceeded for n = 1,000 / 10,000. We raise the limit here purely
# so the worst-case demonstration can complete; this is itself evidence of
# a real practical cost of an unbalanced BST's degenerate O(n) depth.
sys.setrecursionlimit(30_000)


def make_cities(n, shuffled=True):
    ids = list(range(1, n + 1))
    if shuffled:
        random.shuffle(ids)
    return [City(i, f"City{i}", i * 0.1, i * 0.2, i * 10, distance=random.random() * 1000)
            for i in ids]


def time_it(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def benchmark_insertion(n, cities):
    results = {}

    bst = BST()
    results["BST (random order)"] = time_it(lambda: [bst.insert(c) for c in cities])

    avl = AVLTree()
    results["AVL Tree"] = time_it(lambda: [avl.insert(c) for c in cities])

    ht = HashTableChaining(capacity=max(16, n // 2))
    results["Hash Table (chaining)"] = time_it(lambda: [ht.insert(c) for c in cities])

    heap = MinHeap()
    results["Min-Heap (push)"] = time_it(lambda: [heap.push(c) for c in cities])

    return results, bst, avl, ht, heap


def benchmark_search(n, cities, bst, avl, ht):
    # search for every id once, in a fresh random order
    lookup_ids = [c.city_id for c in cities]
    random.shuffle(lookup_ids)

    results = {}
    results["BST"] = time_it(lambda: [bst.search(i) for i in lookup_ids])
    results["AVL Tree"] = time_it(lambda: [avl.search(i) for i in lookup_ids])
    results["Hash Table (chaining)"] = time_it(lambda: [ht.search(i) for i in lookup_ids])
    return results


def benchmark_worst_case_bst_vs_avl(n):
    """Insert in SORTED order to show BST degeneration vs AVL self-balancing."""
    sorted_cities = [City(i, f"City{i}", i, i, i) for i in range(1, n + 1)]

    bst = BST()
    bst_time = time_it(lambda: [bst.insert(c) for c in sorted_cities])

    avl = AVLTree()
    avl_time = time_it(lambda: [avl.insert(c) for c in sorted_cities])

    return bst_time, avl_time, bst.height(), avl.height()


def main():
    insertion_results = {size: {} for size in SIZES}
    search_results = {size: {} for size in SIZES}
    worst_case_results = {size: {} for size in SIZES}

    for n in SIZES:
        print(f"\nRunning benchmarks for n = {n} ...")
        cities = make_cities(n)

        ins_res, bst, avl, ht, heap = benchmark_insertion(n, cities)
        insertion_results[n] = ins_res

        search_res = benchmark_search(n, cities, bst, avl, ht)
        search_results[n] = search_res

        bst_t, avl_t, bst_h, avl_h = benchmark_worst_case_bst_vs_avl(n)
        worst_case_results[n] = {
            "BST insert time (sorted input)": bst_t,
            "AVL insert time (sorted input)": avl_t,
            "BST height (sorted input)": bst_h,
            "AVL height (sorted input)": avl_h,
        }

    # ---------------- Print + save results table ----------------
    print("\n\n================ INSERTION TIME (seconds) ================")
    print_table(insertion_results)

    print("\n================ SEARCH TIME (seconds) ================")
    print_table(search_results)

    print("\n================ BST vs AVL WORST CASE (sorted input) ================")
    print_table(worst_case_results)

    save_csv(insertion_results, "insertion_results.csv")
    save_csv(search_results, "search_results.csv")
    save_csv(worst_case_results, "worst_case_results.csv")

    # ---------------- Plots ----------------
    plot_results(insertion_results, "Insertion Time vs n", "insertion_time.png")
    plot_results(search_results, "Search Time vs n", "search_time.png")
    plot_worst_case(worst_case_results, "bst_vs_avl_worst_case.png")

    print("\nSaved: insertion_time.png, search_time.png, bst_vs_avl_worst_case.png")
    print("Saved: insertion_results.csv, search_results.csv, worst_case_results.csv")


def print_table(results_by_n):
    structures = list(next(iter(results_by_n.values())).keys())
    header = f"{'n':>8} | " + " | ".join(f"{s:>28}" for s in structures)
    print(header)
    print("-" * len(header))
    for n, res in results_by_n.items():
        row = f"{n:>8} | " + " | ".join(f"{res[s]:>28.6f}" for s in structures)
        print(row)


def save_csv(results_by_n, filename):
    structures = list(next(iter(results_by_n.values())).keys())
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n"] + structures)
        for n, res in results_by_n.items():
            writer.writerow([n] + [res[s] for s in structures])


def plot_results(results_by_n, title, filename):
    structures = list(next(iter(results_by_n.values())).keys())
    ns = list(results_by_n.keys())

    plt.figure(figsize=(8, 5))
    for s in structures:
        times = [results_by_n[n][s] for n in ns]
        plt.plot(ns, times, marker="o", label=s)

    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Wall-clock time (seconds)")
    plt.title(title)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_worst_case(results_by_n, filename):
    ns = list(results_by_n.keys())
    bst_times = [results_by_n[n]["BST insert time (sorted input)"] for n in ns]
    avl_times = [results_by_n[n]["AVL insert time (sorted input)"] for n in ns]
    bst_heights = [results_by_n[n]["BST height (sorted input)"] for n in ns]
    avl_heights = [results_by_n[n]["AVL height (sorted input)"] for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ns, bst_times, marker="o", label="BST (sorted input)")
    axes[0].plot(ns, avl_times, marker="o", label="AVL (sorted input)")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Wall-clock insertion time (s)")
    axes[0].set_title("Insertion Time: Sorted Input")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    axes[1].plot(ns, bst_heights, marker="o", label="BST height")
    axes[1].plot(ns, avl_heights, marker="o", label="AVL height")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Tree height")
    axes[1].set_title("Tree Height: Sorted Input (BST degenerates to O(n))")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
