"""
city.py
-------
Common data record used by every data structure in Task 1.

A City is keyed by a unique integer `city_id` (this is the key used for
BST / AVL / Hash Table ordering and lookup). It also stores coordinates,
population, and `distance` (e.g. distance from a fixed origin/source),
which is the field used as the priority key in the Min-Heap.
"""

from dataclasses import dataclass


@dataclass
class City:
    city_id: int          # unique key used by BST / AVL / Hash Table
    name: str
    x: float               # x-coordinate
    y: float               # y-coordinate
    population: int
    distance: float = float("inf")  # priority key used by the Min-Heap

    def __repr__(self):
        return (f"City(id={self.city_id}, name='{self.name}', "
                f"pos=({self.x:.1f},{self.y:.1f}), "
                f"pop={self.population}, dist={self.distance})")


if __name__ == "__main__":
    c = City(city_id=1, name="Kathmandu", x=85.3, y=27.7, population=1_500_000, distance=0)
    print(c)
