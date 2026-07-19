"""
min_heap.py
-----------
Array-based binary Min-Heap, used as a priority queue keyed on
City.distance (e.g. "next nearest city to visit" in route planning /
Dijkstra-style algorithms).

Time Complexity (n = number of elements currently in the heap):
    insert (push):        O(log n)   -- append then sift up
    extract_min (pop):    O(log n)   -- swap root with last, sift down
    peek_min:             O(1)
    decrease_key:         O(log n)   -- update priority then sift up
    build_heap (heapify):  O(n)      -- amortised, from an unsorted array

Space Complexity: O(n).

Practical note: because the heap is array-backed (a Python list), it has
excellent cache locality compared with pointer-based trees, so its
constant factor is small -- it is usually noticeably faster in wall-clock
terms than its O(log n) BST/AVL counterparts for the same n, despite
having the same asymptotic complexity for search-like operations.
"""

from city import City


class MinHeap:
    def __init__(self):
        self._heap = []  # list of City objects, ordered by .distance
        # index map for O(1) location lookup, needed for decrease_key
        self._pos = {}  # city_id -> index in self._heap

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return len(self._heap) == 0

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------
    def _swap(self, i, j):
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._pos[self._heap[i].city_id] = i
        self._pos[self._heap[j].city_id] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[i].distance < self._heap[parent].distance:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._heap)
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            smallest = i
            if left < n and self._heap[left].distance < self._heap[smallest].distance:
                smallest = left
            if right < n and self._heap[right].distance < self._heap[smallest].distance:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def push(self, city: City):
        """Insert a city - O(log n)."""
        self._heap.append(city)
        i = len(self._heap) - 1
        self._pos[city.city_id] = i
        self._sift_up(i)

    def pop(self):
        """Remove and return the city with smallest distance - O(log n)."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        min_city = self._heap[0]
        last = self._heap.pop()
        del self._pos[min_city.city_id]
        if self._heap:
            self._heap[0] = last
            self._pos[last.city_id] = 0
            self._sift_down(0)
        return min_city

    def peek(self):
        """Return (without removing) the city with smallest distance - O(1)."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def decrease_key(self, city_id: int, new_distance: float):
        """
        Lower the distance of an existing city and restore heap order.
        Used heavily by Dijkstra's algorithm. O(log n).
        """
        if city_id not in self._pos:
            raise KeyError(f"city_id {city_id} not in heap")
        i = self._pos[city_id]
        if new_distance > self._heap[i].distance:
            raise ValueError("new_distance is greater than current distance")
        self._heap[i].distance = new_distance
        self._sift_up(i)

    @classmethod
    def build_heap(cls, cities):
        """Build a heap from an existing list of cities in O(n)."""
        h = cls()
        h._heap = list(cities)
        h._pos = {c.city_id: idx for idx, c in enumerate(h._heap)}
        for i in range(len(h._heap) // 2 - 1, -1, -1):
            h._sift_down(i)
        return h


if __name__ == "__main__":
    import random

    heap = MinHeap()
    cities = [City(i, f"City{i}", i, i, i * 100, distance=random.randint(1, 100))
              for i in range(1, 11)]
    for c in cities:
        heap.push(c)

    print("Heap size:", len(heap))
    print("Peek (min):", heap.peek())

    heap.decrease_key(city_id=cities[-1].city_id, new_distance=0)
    print("After decrease_key on last city -> new min:", heap.peek())

    print("Popping in order:")
    while not heap.is_empty():
        print(" ", heap.pop())
