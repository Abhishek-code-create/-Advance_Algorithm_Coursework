"""
avl_tree.py
-----------
Self-balancing AVL Tree keyed on City.city_id.

Invariant: for every node, |height(left) - height(right)| <= 1.
This is enforced after every insert/delete via rotations, which keeps
the tree height at O(log n) at all times -- including worst-case input
order (e.g. sorted insertion), unlike a plain BST.

Time Complexity (n = number of nodes):
    Insert:  O(log n) worst case  (search path + O(1) amortised rotations,
             but up to O(log n) rotations can occur along the path in theory;
             in practice AVL insert needs at most 2 rotations)
    Search:  O(log n) worst case
    Delete:  O(log n) worst case  (may require rotations up the whole path)

Space Complexity: O(n) storage, O(log n) recursion stack.

Practical note: AVL trees have a larger constant factor than a plain BST
for insertion/deletion because of the extra height bookkeeping and
rotation logic, but this is repaid by guaranteed O(log n) height, which
matters a lot for search-heavy / adversarial-input workloads.
"""

from city import City


class AVLNode:
    __slots__ = ("city", "left", "right", "height")

    def __init__(self, city: City):
        self.city = city
        self.left = None
        self.right = None
        self.height = 1  # height of a leaf is 1


class AVLTree:
    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    # ---------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------
    def _h(self, node):
        return node.height if node else 0

    def _update_height(self, node):
        node.height = 1 + max(self._h(node.left), self._h(node.right))

    def _balance_factor(self, node):
        return self._h(node.left) - self._h(node.right) if node else 0

    def _rotate_right(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node):
        self._update_height(node)
        balance = self._balance_factor(node)

        # Left heavy
        if balance > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)  # left-right case
            return self._rotate_right(node)  # left-left case

        # Right heavy
        if balance < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)  # right-left case
            return self._rotate_left(node)  # right-right case

        return node

    # ---------------------------------------------------------------
    # Insert - O(log n)
    # ---------------------------------------------------------------
    def insert(self, city: City):
        self.root = self._insert(self.root, city)

    def _insert(self, node, city):
        if node is None:
            self._size += 1
            return AVLNode(city)
        if city.city_id < node.city.city_id:
            node.left = self._insert(node.left, city)
        elif city.city_id > node.city.city_id:
            node.right = self._insert(node.right, city)
        else:
            node.city = city  # update
            return node
        return self._rebalance(node)

    # ---------------------------------------------------------------
    # Search - O(log n)
    # ---------------------------------------------------------------
    def search(self, city_id: int):
        node = self.root
        while node is not None:
            if city_id == node.city.city_id:
                return node.city
            node = node.left if city_id < node.city.city_id else node.right
        return None

    # ---------------------------------------------------------------
    # Delete - O(log n)
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
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.city = successor.city
            node.right, _ = self._delete(node.right, successor.city.city_id)

        if node is None:
            return node, deleted
        return self._rebalance(node), deleted

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def height(self):
        return self._h(self.root)

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
    tree = AVLTree()
    # Insert in SORTED order -- the adversarial case that breaks plain BST
    for i in range(1, 21):
        tree.insert(City(i, f"City{i}", i, i, i))

    print("Size:", len(tree))
    print("Height (sorted insertion, AVL stays balanced):", tree.height())
    print("Search 15:", tree.search(15))
    print("Delete 15:", tree.delete(15))
    print("Search 15 after delete:", tree.search(15))
    print("Height after delete:", tree.height())
