"""
Task 3 - Greedy Algorithm
=========================
Problem: Minimum Number of Platforms

Given the arrival and departure times of n trains at a station, find the
minimum number of platforms required so that no train has to wait.

--------------------------------------------------------------------------
Greedy choice
--------------------------------------------------------------------------
Sort all arrival times and all departure times independently.
Walk through both sorted lists with two pointers (a "sweep" over events):
  - If the next event in time is an ARRIVAL, a platform is needed
    -> increment the current platform count.
  - If the next event in time is a DEPARTURE, a platform frees up
    -> decrement the current platform count.
Track the maximum number of platforms in use at any point in time; that
maximum is the answer.

Greedy choice justification: at any instant, the number of platforms
needed equals the number of trains simultaneously "in the station"
(arrived but not yet departed). The true minimum number of platforms for
the whole day is therefore exactly the maximum, over all instants, of
that count. Sweeping the sorted arrival/departure events computes this
maximum directly and optimally -- there is no need to consider specific
train identities, only the counts of arrivals/departures up to each
point in time. This greedy sweep is *always* optimal for this exact
problem (unlike, e.g., weighted interval scheduling, where a greedy
choice is not always optimal).

--------------------------------------------------------------------------
Complexity
--------------------------------------------------------------------------
Sorting arrivals and departures: O(n log n)
Two-pointer sweep over 2n events: O(n)
--------------------------------------------------------------------------
Overall time complexity:  O(n log n)   (dominated by the sort)
Overall space complexity: O(n)         (copies of arrival/departure lists)

Hidden constant factor:
- Two separate O(n log n) sorts are performed (arrivals, departures).
  Python's Timsort has a real per-element constant cost from tuple/list
  comparisons; for small n (e.g. n < 50) a naive O(n^2) brute-force
  "count overlapping trains for each train" approach can be faster in
  wall-clock terms because it avoids the overhead of two sort calls and
  works directly on small arrays that fit comfortably in cache.
"""

import random
import time


def min_platforms(arrivals, departures):
    """
    Greedy two-pointer solution.

    Args:
        arrivals:   list of arrival times (int)
        departures: list of departure times (int), same length as arrivals

    Returns:
        Minimum number of platforms required (int)
    """
    n = len(arrivals)
    if n == 0:
        return 0

    arr = sorted(arrivals)
    dep = sorted(departures)

    platforms_needed = 0
    max_platforms = 0
    i, j = 0, 0

    while i < n and j < n:
        if arr[i] <= dep[j]:
            # A train arrives before (or exactly when) another departs
            # -> needs a new platform.
            platforms_needed += 1
            max_platforms = max(max_platforms, platforms_needed)
            i += 1
        else:
            # A train departs, freeing a platform.
            platforms_needed -= 1
            j += 1

    return max_platforms


def brute_force_min_platforms(arrivals, departures):
    """
    Exact O(n^2) baseline: for every point in time equal to some train's
    arrival, count how many trains are present at that moment. Used to
    validate the greedy result and as the "exact approach" comparison
    required by the brief.
    """
    n = len(arrivals)
    max_count = 0
    for t in arrivals:
        count = 0
        for i in range(n):
            if arrivals[i] <= t <= departures[i]:
                count += 1
        max_count = max(max_count, count)
    return max_count


def generate_random_schedule(n, seed=42, day_length=1440):
    """Generate n random (arrival, departure) pairs within a 'day'."""
    rng = random.Random(seed)
    arrivals, departures = [], []
    for _ in range(n):
        a = rng.randint(0, day_length - 1)
        d = a + rng.randint(1, 120)  # trains stay for up to 2 hours
        arrivals.append(a)
        departures.append(d)
    return arrivals, departures


def demo():
    print("=" * 70)
    print("Minimum Number of Platforms - Greedy demo")
    print("=" * 70)

    # Classic textbook example
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    result = min_platforms(arrivals, departures)
    print(f"\nSample instance: {len(arrivals)} trains")
    print(f"Minimum platforms required (greedy): {result}")

    # Validate against brute force on a small random instance
    a, d = generate_random_schedule(20, seed=7)
    greedy_result = min_platforms(a, d)
    exact_result = brute_force_min_platforms(a, d)
    print(f"\nValidation on 20 random trains -> Greedy: {greedy_result}, "
          f"Brute force: {exact_result}")
    assert greedy_result == exact_result, "Mismatch between greedy and brute force!"
    print("Greedy result matches brute force. Optimality confirmed on this instance.")

    # Empirical timing
    print("\nEmpirical timing for increasing n (greedy O(n log n)):")
    for n in (100, 1000, 10000, 100000):
        a, d = generate_random_schedule(n, seed=n)
        start_t = time.perf_counter()
        min_platforms(a, d)
        elapsed = time.perf_counter() - start_t
        print(f"  n = {n:>7}: {elapsed:.6f} s")


if __name__ == "__main__":
    demo()
