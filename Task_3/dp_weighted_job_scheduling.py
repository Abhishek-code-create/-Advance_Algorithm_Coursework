"""
Task 3 - Dynamic Programming
============================
Problem: Weighted Job Scheduling with Time Windows

Given a set of n jobs, each with a start time, an end time, and a profit,
select a subset of NON-OVERLAPPING jobs so that the total profit is
maximised.

--------------------------------------------------------------------------
Subproblem definition
--------------------------------------------------------------------------
Sort jobs by finish time: job[0], job[1], ..., job[n-1].

Let dp[i] = the maximum profit obtainable using only jobs 0..i-1
            (the first i jobs in finish-time order).

Recurrence:
    dp[0] = 0
    dp[i] = max( dp[i-1],                      # skip job i-1
                 profit[i-1] + dp[p(i-1)] )     # take job i-1

where p(i-1) = index of the latest job that finishes at or before the
start time of job i-1 (found via binary search), or 0 if none exists.

The final answer is dp[n].

--------------------------------------------------------------------------
Bottom-up (tabulation) strategy
--------------------------------------------------------------------------
We build dp[] iteratively from dp[0] up to dp[n] instead of using
recursive memoisation, avoiding recursion-depth issues for large n and
keeping the constant factor low (a single pass + one binary search/job).

--------------------------------------------------------------------------
Complexity
--------------------------------------------------------------------------
Sorting jobs by finish time:            O(n log n)
For each job, binary search for p(i):   O(log n)   ->  O(n log n) total
Building the dp table:                  O(n)
--------------------------------------------------------------------------
Overall time complexity:  O(n log n)
Overall space complexity: O(n)   (dp table + sorted job list)

Hidden constant factor:
- The binary search (bisect) has a small but non-trivial constant due to
  repeated list indexing and comparisons; for small n (<< 1000) a simple
  O(n^2) linear scan for p(i) can actually be *faster* in wall-clock time
  because it avoids function-call/bisect overhead and benefits from
  cache-friendly sequential access. The O(n log n) advantage only
  dominates once n is large enough that the extra comparisons of the
  O(n^2) approach outweigh the constant overhead of bisect.
"""

from bisect import bisect_right
from dataclasses import dataclass
import random
import time


@dataclass
class Job:
    start: int
    end: int
    profit: int


def latest_non_conflicting(ends, jobs, i):
    """
    Binary search for the rightmost job index (0-based) whose end time
    is <= start time of jobs[i]. Returns -1 if no such job exists.

    `ends` is a precomputed list of end times for `jobs` (sorted by end
    time). Precomputing it once outside the main loop is what keeps this
    an O(log n) lookup rather than an accidental O(n) one -- building the
    ends list freshly on every call would silently degrade the whole
    algorithm from O(n log n) to O(n^2).
    """
    # bisect_right finds insertion point for jobs[i].start among ends;
    # everything before that index has end <= jobs[i].start.
    idx = bisect_right(ends, jobs[i].start, 0, i)
    return idx - 1  # -1 if none


def weighted_job_scheduling(jobs):
    """
    Solve Weighted Job Scheduling using bottom-up DP.

    Returns:
        (max_profit, chosen_jobs)
    """
    if not jobs:
        return 0, []

    jobs = sorted(jobs, key=lambda j: j.end)
    n = len(jobs)
    ends = [j.end for j in jobs]  # precomputed ONCE -> keeps lookups O(log n)

    dp = [0] * (n + 1)
    choice = [False] * (n + 1)  # did we take job i-1 in the optimal dp[i]?
    parent_take = [-1] * (n + 1)  # for reconstructing the solution

    for i in range(1, n + 1):
        job = jobs[i - 1]
        p = latest_non_conflicting(ends, jobs, i - 1)  # index into jobs (0-based), or -1
        take_profit = job.profit + (dp[p + 1] if p != -1 else 0)
        skip_profit = dp[i - 1]

        if take_profit > skip_profit:
            dp[i] = take_profit
            choice[i] = True
            parent_take[i] = p + 1  # dp index to jump back to
        else:
            dp[i] = skip_profit
            choice[i] = False

    # Reconstruct chosen jobs
    selected = []
    i = n
    while i > 0:
        if choice[i]:
            selected.append(jobs[i - 1])
            i = parent_take[i]
        else:
            i -= 1
    selected.reverse()

    return dp[n], selected


def brute_force_job_scheduling(jobs):
    """
    Exact exponential baseline (2^n subsets) used only for small n to
    validate correctness of the DP solution. NOT used for large n.
    """
    n = len(jobs)
    best = 0
    best_subset = []

    def overlaps(a, b):
        return a.start < b.end and b.start < a.end

    for mask in range(1 << n):
        subset = [jobs[i] for i in range(n) if mask & (1 << i)]
        valid = True
        for x in range(len(subset)):
            for y in range(x + 1, len(subset)):
                if overlaps(subset[x], subset[y]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            profit = sum(j.profit for j in subset)
            if profit > best:
                best = profit
                best_subset = subset

    return best, best_subset


def generate_random_jobs(n, seed=42, max_time=1000, max_profit=100):
    rng = random.Random(seed)
    jobs = []
    for _ in range(n):
        start = rng.randint(0, max_time - 1)
        end = start + rng.randint(1, 50)
        profit = rng.randint(1, max_profit)
        jobs.append(Job(start, end, profit))
    return jobs


def demo():
    print("=" * 70)
    print("Weighted Job Scheduling - Dynamic Programming demo")
    print("=" * 70)

    # Classic textbook example
    sample_jobs = [
        Job(1, 3, 5),
        Job(2, 5, 6),
        Job(4, 6, 5),
        Job(6, 7, 4),
        Job(5, 8, 11),
        Job(7, 9, 2),
    ]
    profit, chosen = weighted_job_scheduling(sample_jobs)
    print(f"\nSample instance: {len(sample_jobs)} jobs")
    print(f"Max profit (DP): {profit}")
    print("Selected jobs (start, end, profit):",
          [(j.start, j.end, j.profit) for j in chosen])

    # Correctness check against brute force on a small random instance
    small_jobs = generate_random_jobs(15, seed=1)
    dp_profit, _ = weighted_job_scheduling(small_jobs)
    bf_profit, _ = brute_force_job_scheduling(small_jobs)
    print(f"\nValidation on 15 random jobs -> DP: {dp_profit}, Brute force: {bf_profit}")
    assert dp_profit == bf_profit, "Mismatch between DP and brute force!"
    print("DP result matches brute force. Correctness check passed.")

    # Empirical timing (feeds into Task 1/3 complexity discussion)
    print("\nEmpirical timing for increasing n (DP only, O(n log n)):")
    for n in (100, 1000, 10000, 100000):
        jobs = generate_random_jobs(n, seed=n)
        start_t = time.perf_counter()
        weighted_job_scheduling(jobs)
        elapsed = time.perf_counter() - start_t
        print(f"  n = {n:>7}: {elapsed:.6f} s")


if __name__ == "__main__":
    demo()
