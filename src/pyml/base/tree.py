import numpy as np
from numpy.typing import ArrayLike, NDArray


class Node:
    def __init__(self,
            depth : int,
            index : NDArray, 
            ):
        self.depth = depth
        self.index = index
        self.t: float = None  # threshold
        self.j: int   = None  # feature
        self.target: float = None
        self.left  : Node = None
        self.right : Node = None

    @property
    def num_samples(self) -> int:
        return self.index.size

    def __str__(self):
        return f"N{self.depth}" if self.target is None else f"L{self.depth}"


class Tree:
    root: Node | None

    def fit(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x)
        y = np.asarray(y)
        N, F = x.shape

        root = Node(0, np.arange(N))
        queue = [root]

        while queue:
            node = queue.pop(0)
            j, t = self.search_split(x, y, node)
            
            if j is None or t is None:
                node.target = self.leaf_target(y[node.index])
            else:
                node.j = j
                node.t = t
                mask = x[node.index, j] < t
                node.left  = Node(node.depth+1, node.index[mask])
                node.right = Node(node.depth+1, node.index[~mask])
                queue.append(node.left)
                queue.append(node.right)

        self.root = root

    def search_split(self, x: NDArray, y: NDArray, node: Node):
        N, F = x.shape
        best = 0.0  # min_score later on
        t_star = None
        j_star = None

        if self.skip(node) == True:
            return j_star, t_star

        yi = y[node.index]
        xi = x[node.index]

        for j in range(F):
            xij = xi[:, j]
            t, score = self.threshold(xij, yi)
            if score > best:
                best = score
                j_star = j
                t_star = t

        return j_star, t_star

    def skip(self, node: Node) -> bool:
        raise NotImplementedError

    def threshold(self, xij: NDArray, yi: NDArray):
        raise NotImplementedError

    def leaf_target(self, yi: NDArray) -> float:
        raise NotImplementedError

    def predict(self, x: ArrayLike) -> NDArray:
        assert self.root is not None
        x = np.asarray(x)
        N, F = x.shape
        predictions = np.full((N,), fill_value=np.nan)
        queue = [(self.root, np.arange(N))]
        while queue:
            node, index = queue.pop(0)

            if node.target is not None:
                predictions[index] = node.target
                continue

            mask = x[index, node.j] < node.t
            queue.append((node.left, index[mask]))
            queue.append((node.right, index[~mask]))
            
        return predictions

    def prd(self, x: NDArray) -> NDArray:
        return self.predict(x)

    def ravel(self) -> list[Node]:
        assert self.root is not None
        stack = [self.root]
        nodes = []
        while stack:
            node = stack.pop(0)
            if node.right is not None:
                stack.insert(0, node.right)
            if node.left is not None:
                stack.insert(0, node.left)
            nodes.append(node)
        return nodes

    @property
    def depth(self) -> int:
        return max(node.depth for node in self.ravel())

    @property
    def num_nodes(self) -> int:
        return len(self.ravel())

    @property
    def num_leaves(self) -> int:
        return len([node for node in self.ravel() if node.target is not None])

    def __str__(self):
        string = ""
        for node in self.ravel():
            string += "|  "*(node.depth-1) + "|--" + node.__str__() + f"\n"
        return string




# def reduced_error_pruning(self, X_val:np.ndarray, y_val:np.ndarray) -> None:
#     raise NotImplementedError



# class DecisionTree:
#     def __init__(
#             self,
#             # criterion : Literal['mse', 'mae'] = 'mse',
#             # criterion : Literal['info', 'gini'] = 'info',
#             max_depth : int = None,
#         ):

# min_samples_leaf : int = 1,
# # min_samples_split : int  = 2,
# max_features : int = None,
# min_impurity_decrease : float = 0.0,
# random_state : int = None




# Example tree
# Here the first N2 could have its leaf children (L3, L3) being pruned!
# Though the second N2 could not have its children (L3, N3) pruned.
# N0: indicies [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16], feature 1, threshold 2 
#  |  N1: indicies [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13], feature 1, threshold 1 
#  |   |  N2: indicies [ 8 11 12 13], feature 0, threshold 6 
#  |   |   |  L3: indicies [11 12] 
#  |   |   |  L3: indicies [ 8 13] 
#  |   |  N2: indicies [ 0  1  2  3  4  5  6  7  9 10], feature 0, threshold 4 
#  |   |   |  L3: indicies [ 1  3 10] 
#  |   |   |  N3: indicies [0 2 4 5 6 7 9], feature 0, threshold 56 
#  |   |   |   |  L4: indicies [0 4 5 6 7] 
#  |   |   |   |  L4: indicies [2 9] 
#  |  L1: indicies [14 15 16]

