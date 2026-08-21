import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable

from pyml.estimator.base import Estimator
from pyml.estimator.base import Tree, Node


def variance_reduction(y: NDArray, y1: NDArray, y2: NDArray):
    N  = y.size
    N1 = y1.size
    N2 = y2.size
    score = y.var() - N1 / N * y1.var() - N2 / N * y2.var()
    return score




class DecisionTree(Estimator, Tree):
    def __init__(
            self,
            criterion: Callable,
            min_samples: int = 2,
            max_depth: int = 10,
        ):
        self.criterion = criterion
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def fit(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x)
        y = np.asarray(y)
        N, F = x.shape

        index = np.arange(N)
        root = Node(0, index)
        queue = [root]

        while queue:
            node = queue.pop(0)
            j, t = self.search_split(x, y, node)
            
            if j is None or t is None:
                node.target = y[node.index].mean()
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

        for j in range(F):
            xij = x[node.index, j]
            yi  = y[node.index]
            num_dt = min(node.num_samples, 10)
            x05 = np.quantile(xij, q=0.05)
            x95 = np.quantile(xij, q=0.95)
            dt = (x95 - x05) / num_dt

            for k in range(1, num_dt):
                t = x05 + k*dt
                mask = xij < t

                y1, y2 = yi[mask], yi[~mask]
                if y1.size < self.min_samples or y2.size < self.min_samples:
                    continue
                
                score = self.criterion(yi, y1, y2)

                if score > best:
                    best = score
                    t_star = t
                    j_star = j

        return j_star, t_star

    def skip(self, node: Node) -> bool:
        if node.num_samples < 2 * self.min_samples:
            return True
        if node.depth >= self.max_depth:
            return True
        return False
            
    def predict(self, x: ArrayLike) -> NDArray:
        assert self.root is not None
        x = np.asarray(x)
        N, F = x.shape
        predictions = np.full((N,), fill_value=np.nan)
        for i in range(N):
            node = self.root
            while node.target is None:
                node = node.left if x[i, node.j] < node.t else node.right
            predictions[i] = node.target
        return predictions



# class DecisionTree2(DecisionTree):
#     def fit():
#         pass



# classification: gini impurity or information gain
# regression: variance reduction
# criterion abstraction

# 1. Go through all features j
# 2. For feature j, go through a reasonable number of thresholds t
# 3. Find the feature and threshold improving the criterion the most
# 4. Create child nodes, split index


    