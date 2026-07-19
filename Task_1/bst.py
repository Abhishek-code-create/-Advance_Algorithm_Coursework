"""
bst.py
------
Unbalanced Binary Search Tree (BST) keyed on City.city_id.

Time Complexity (n = number of nodes):
    Average case (roughly balanced by chance): O(log n) for insert/search/delete
    Worst case (degenerate / sorted insertion order): O(n) for insert/search/delete
        -> the tree degrades into a linked list.

Space Complexity: O(n) for storage, O(h) recursion stack for operations
    (h = height of tree, which is O(log n) average / O(n) worst case).

The BST intentionally does NOT self-balance -- it is implemented here so
that Task 1's empirical tests can demonstrate how performance degrades on
adversarial (e.g. sorted) input, in contrast with the AVL tree.
"""

from city import City


class BSTNode:
    __slots__ = ("city", "left", "right")

    def __init__(self, city: City):
        self.city = city
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    # ---------------------------------------------------------------
    # Insert - O(h) where h is tree height
    # ---------------------------------------------------------------
    def insert(self, city: City):
        self.root = self._insert(self.root, city)

    def _insert(self, node, city):
        if node is None:
            self._size += 1
            return BSTNode(city)
        if city.city_id < node.city.city_id:
            node.left = self._insert(node.left, city)
        elif city.city_id > node.city.city_id:
            node.right = self._insert(node.right, city)
        else:
            node.city = city  # update existing key
        return node

    # ---------------------------------------------------------------
    # Search - O(h)
    # ---------------------------------------------------------------
    def search(self, city_id: int):
        node = self.root
        while node is not None:
            if city_id == node.city.city_id:
                return node.city
            node = node.left if city_id < node.city.city_id else node.right
        return None

    # ---------------------------------------------------------------
    # Delete - O(h)
    # ---------------------------------------------------------------
    def delete(self, city_id: int):
        self.root, deleted = self._delete(self.root, city_id)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, city_id):
        if node is None:
            return node, False

        if city_id < node.city.city_id:
            node.left, deleted = self._delete(node.left, city_id)
        elif city_id > node.city.city_id:
            node.right, deleted = self._delete(node.right, city_id)
        else:
            deleted = True
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            # two children: replace with in-order successor
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.city = successor.city
            node.right, _ = self._delete(node.right, successor.city.city_id)

        return node, deleted

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def height(self, node="root"):
        if node == "root":
            node = self.root
        if node is None:
            return 0
        return 1 + max(self.height(node.left), self.height(node.right))

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.city)
            self._inorder(node.right, result)


if __name__ == "__main__":
    import random

    tree = BST()
    ids = list(range(1, 21))
    random.shuffle(ids)
    for i in ids:
        tree.insert(City(i, f"City{i}", i * 1.1, i * 2.2, i * 1000))

    print("Size:", len(tree))
    print("Height (random order):", tree.height())
    print("Search 10:", tree.search(10))
    print("Delete 10:", tree.delete(10))
    print("Search 10 after delete:", tree.search(10))

    # Demonstrate worst-case degeneration with sorted insertion
    sorted_tree = BST()
    for i in range(1, 21):
        sorted_tree.insert(City(i, f"City{i}", i, i, i))
    print("Height (sorted order, worst case):", sorted_tree.height())
