import numpy as np
from numpy.typing import NDArray

from pyml.base import Estimator
from pyml.base import Tree, Node


class DecisionTreeVR(Estimator, Tree):  # Variance Reduction
    def __init__(
            self,
            min_samples: int = 2,
            max_depth: int = 10,
        ):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def skip(self, node: Node) -> bool:
        if node.num_samples < 2 * self.min_samples:
            return True
        if node.depth >= self.max_depth:
            return True
        return False

    def threshold(self, xij: NDArray, yi: NDArray):
        n = yi.size
        order = np.argsort(xij)
        x_sorted = xij[order]
        valid = x_sorted[1:] != x_sorted[:-1]
        valid[:self.min_samples-1] = False
        valid[n-self.min_samples:] = False

        k = np.flatnonzero(valid)
        if k.size == 0:
            return None, 0.0

        lsum = np.cumsum(yi[order])
        lssq = np.cumsum(yi[order] ** 2)
        rsum = lsum[-1] - lsum
        rssq = lssq[-1] - lssq

        sse  = lssq[-1] - lsum[-1] ** 2 / n
        sse1 = lssq[k] - lsum[k] ** 2 / (k+1)
        sse2 = rssq[k] - rsum[k] ** 2 / (n-k-1)

        scores = sse - sse1 - sse2
        argmax = np.argmax(scores)

        score = scores[argmax]
        t = (x_sorted[k[argmax]] + x_sorted[k[argmax]+1]) / 2
        return t, score

    def leaf_target(self, yi: NDArray):
        return yi.mean()





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

    