import numpy as np
from numpy.typing import NDArray


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

    def predict(self, x: NDArray) -> NDArray:
        raise NotImplementedError

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




# @property
# def nodes(self) -> list[Node]:
#     return [node for node in self.ravel_dfs() if not node.is_leaf]

# @property
# def leaves(self) -> list[Node]:
#     return [node for node in self.ravel_dfs() if node.is_leaf]

# def reduced_error_pruning(self, X_val:np.ndarray, y_val:np.ndarray) -> None:
#     raise NotImplementedError

# def _search_split(self, x:np.ndarray, y:np.ndarray) -> tuple[int, int|float]:
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

