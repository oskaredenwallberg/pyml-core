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

        yi = y[node.index]
        xi = x[node.index]
        n = yi.size

        for j in range(F):
            xij = xi[:, j]
            order = np.argsort(xij)
            x_sorted = xij[order]
            valid = x_sorted[1:] != x_sorted[:-1]
            valid[:self.min_samples-1] = False
            valid[n-self.min_samples:] = False

            k = np.flatnonzero(valid)
            if k.size == 0:
                continue

            lsum = np.cumsum(yi[order])
            lssq = np.cumsum(yi[order] ** 2)
            rsum = lsum[-1] - lsum
            rssq = lssq[-1] - lssq

            sse  = lssq[-1] - lsum[-1] ** 2 / n
            sse1 = lssq[k] - lsum[k] ** 2 / (k+1)
            sse2 = rssq[k] - rsum[k] ** 2 / (n-k-1)

            scores = sse - sse1 - sse2
            argmax = np.argmax(scores)

            if scores[argmax] > best:
                best = scores[argmax]
                j_star = j
                t_star = (x_sorted[k[argmax]] + x_sorted[k[argmax]+1]) / 2

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


# score = self.criterion(yi, y1, y2)



# classification: gini impurity or information gain
# regression: variance reduction
# criterion abstraction

# 1. Go through all features j
# 2. For feature j, go through a reasonable number of thresholds t
# 3. Find the feature and threshold improving the criterion the most
# 4. Create child nodes, split index



# def search_split(self, x: NDArray, y: NDArray, node: Node):
#     N, F = x.shape
#     best = 0.0
#     t_star = None
#     j_star = None

#     if self.skip(node):
#         return j_star, t_star

#     yi = y[node.index]
#     xi = x[node.index]
#     n = yi.size
#     totsum = np.sum(yi)
#     totssq = np.sum(yi ** 2)
#     totsse = totssq - totsum**2 / n

#     for j in range(F):
#         xij = xi[:, j]
#         order = np.argsort(xij)

#         x_sorted = xij[order]
#         y_sorted = yi[order]

#         lsum, rsum = 0.0, totsum
#         lssq, rssq = 0.0, totssq

#         for k in range(n-1):
#             yk = y_sorted[k]
#             lsum, rsum = lsum+yk,    rsum-yk
#             lssq, rssq = lssq+yk**2, rssq-yk**2

#             if x_sorted[k] == x_sorted[k + 1]:
#                 continue

#             n1 = k + 1
#             n2 = n - n1
#             if (n1 < self.min_samples) or (n2 < self.min_samples):
#                 continue

#             lsse = lssq - lsum**2 / n1
#             rsse = rssq - rsum**2 / n2
#             score = totsse - lsse - rsse

#             if score > best:
#                 best = score
#                 j_star = j
#                 t_star = (x_sorted[k] + x_sorted[k + 1]) / 2

#     return j_star, t_star

    