"""
hash_table.py
-------------
Hash Table keyed on City.city_id, providing O(1) average-case lookup.

Two collision-handling strategies are implemented so the report can
compare them:

1. HashTableChaining        -- each bucket is a Python list ("chain").
2. HashTableOpenAddressing   -- linear probing with a single flat array.

Time Complexity (n = number of entries, m = number of buckets/slots,
load factor alpha = n / m):
    Chaining:
        Average case: O(1 + alpha) for insert/search/delete
        Worst case:   O(n) if all keys collide into one bucket
    Open addressing (linear probing):
        Average case: O(1 / (1 - alpha)) for insert/search/delete
        Worst case:   O(n) (degrades badly as alpha -> 1, clustering)

Space Complexity: O(n) for chaining (plus O(m) empty buckets),
                   O(m) for open addressing (m must stay > n).

Practical note: chaining is simpler and degrades more gracefully at high
load factors, but has pointer/list overhead per bucket. Open addressing
has better cache locality (contiguous array) but requires resizing well
before the load factor gets close to 1, and suffers from primary
clustering under linear probing.
"""

from city import City


# =====================================================================
# 1. Hash Table with Separate Chaining
# =====================================================================
class HashTableChaining:
    def __init__(self, capacity: int = 16, max_load_factor: float = 0.75):
        self._capacity = capacity
        self._buckets = [[] for _ in range(capacity)]
        self._size = 0
        self._max_load_factor = max_load_factor

    def __len__(self):
        return self._size

    def _hash(self, key: int) -> int:
        return hash(key) % self._capacity

    def _resize(self):
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        for bucket in old_buckets:
            for city in bucket:
                self.insert(city)

    def insert(self, city: City):
        """O(1) average case; O(n) worst case (resize / bad hash)."""
        if (self._size + 1) / self._capacity > self._max_load_factor:
            self._resize()
        idx = self._hash(city.city_id)
        bucket = self._buckets[idx]
        for i, existing in enumerate(bucket):
            if existing.city_id == city.city_id:
                bucket[i] = city  # update
                return
        bucket.append(city)
        self._size += 1

    def search(self, city_id: int):
        """O(1) average case; O(n) worst case (all keys in one bucket)."""
        idx = self._hash(city_id)
        for city in self._buckets[idx]:
            if city.city_id == city_id:
                return city
        return None

    def delete(self, city_id: int) -> bool:
        """O(1) average case; O(n) worst case."""
        idx = self._hash(city_id)
        bucket = self._buckets[idx]
        for i, city in enumerate(bucket):
            if city.city_id == city_id:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def load_factor(self):
        return self._size / self._capacity


# =====================================================================
# 2. Hash Table with Open Addressing (Linear Probing)
# =====================================================================
_DELETED = object()  # tombstone marker


class HashTableOpenAddressing:
    def __init__(self, capacity: int = 16, max_load_factor: float = 0.5):
        self._capacity = capacity
        self._slots = [None] * capacity
        self._size = 0
        self._max_load_factor = max_load_factor

    def __len__(self):
        return self._size

    def _hash(self, key: int) -> int:
        return hash(key) % self._capacity

    def _resize(self):
        old_slots = self._slots
        self._capacity *= 2
        self._slots = [None] * self._capacity
        self._size = 0
        for slot in old_slots:
            if slot is not None and slot is not _DELETED:
                self.insert(slot)

    def insert(self, city: City):
        """O(1) average case (low load factor); degrades as table fills up."""
        if (self._size + 1) / self._capacity > self._max_load_factor:
            self._resize()

        idx = self._hash(city.city_id)
        first_deleted = None
        for _ in range(self._capacity):
            slot = self._slots[idx]
            if slot is None:
                target = first_deleted if first_deleted is not None else idx
                self._slots[target] = city
                self._size += 1
                return
            if slot is _DELETED:
                if first_deleted is None:
                    first_deleted = idx
            elif slot.city_id == city.city_id:
                self._slots[idx] = city  # update
                return
            idx = (idx + 1) % self._capacity
        raise RuntimeError("Hash table is full")

    def search(self, city_id: int):
        """O(1) average case; O(n) worst case under heavy clustering."""
        idx = self._hash(city_id)
        for _ in range(self._capacity):
            slot = self._slots[idx]
            if slot is None:
                return None
            if slot is not _DELETED and slot.city_id == city_id:
                return slot
            idx = (idx + 1) % self._capacity
        return None

    def delete(self, city_id: int) -> bool:
        """O(1) average case; uses tombstones to keep probe chains intact."""
        idx = self._hash(city_id)
        for _ in range(self._capacity):
            slot = self._slots[idx]
            if slot is None:
                return False
            if slot is not _DELETED and slot.city_id == city_id:
                self._slots[idx] = _DELETED
                self._size -= 1
                return True
            idx = (idx + 1) % self._capacity
        return False

    def load_factor(self):
        return self._size / self._capacity


if __name__ == "__main__":
    cities = [City(i, f"City{i}", i, i, i * 1000) for i in range(1, 51)]

    print("--- Chaining ---")
    ht_chain = HashTableChaining(capacity=8)
    for c in cities:
        ht_chain.insert(c)
    print("Size:", len(ht_chain), "Load factor:", round(ht_chain.load_factor(), 2))
    print("Search 25:", ht_chain.search(25))
    print("Delete 25:", ht_chain.delete(25))
    print("Search 25 after delete:", ht_chain.search(25))

    print("\n--- Open Addressing ---")
    ht_open = HashTableOpenAddressing(capacity=8)
    for c in cities:
        ht_open.insert(c)
    print("Size:", len(ht_open), "Load factor:", round(ht_open.load_factor(), 2))
    print("Search 25:", ht_open.search(25))
    print("Delete 25:", ht_open.delete(25))
    print("Search 25 after delete:", ht_open.search(25))
